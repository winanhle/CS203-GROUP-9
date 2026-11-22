"""Run a resumable, SHA-256-verified CM4IR inpainting ablation evaluation."""

import argparse
import csv
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torchvision
import yaml
from filelock import FileLock
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
OFFICIAL_LSUN_CAT_CHECKPOINT_SHA = "e56343f9fce32260396578734e480c2752a45c517920e8a4f20cf7f79ea9333f"
RECORD_FIELDS = (
    "index",
    "input",
    "sha256",
    "restored",
    "restored_sha256",
    "full_psnr_db",
    "full_vgg_lpips",
    "hole_psnr_db",
    "hole_vgg_lpips",
    "observed_max_abs_diff",
)

# Lazily resolved at runtime so tests can monkeypatch run_inpainting_ablation.Diffusion
# without triggering guided_diffusion.diffusion's top-level VGG LPIPS initialization.
Diffusion = None
_LPIPS_CPU_MODEL = None


def digest(path: Path) -> str:
    """Compute SHA-256 hex digest using Python 3.8+-compatible chunked reads."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON with allow_nan=False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write per-image results.csv with empty string for None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in RECORD_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dict2namespace(config: dict[str, Any]) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


class CleanImages:
    """Minimal torch Dataset yielding (tensor_01, 0) matching ImageFolder."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from torchvision.transforms.functional import to_tensor

        with Image.open(self.paths[index]) as image:
            return to_tensor(image.convert("RGB")), 0


def psnr(mse: float) -> Optional[float]:
    """Convert MSE in [0, 1] to PSNR dB; None (JSON null) represents MSE == 0."""
    if mse < 0 or not math.isfinite(mse):
        raise ValueError(f"Invalid MSE value: {mse}")
    return -10.0 * math.log10(mse) if mse > 0.0 else None


def safe_mean(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    """Return mean over rows for key, or None if empty or any entry is None."""
    if not rows:
        return None
    values = [row[key] for row in rows]
    if any(v is None for v in values):
        return None
    return float(mean(values))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _get_lpips_cpu():
    global _LPIPS_CPU_MODEL
    if _LPIPS_CPU_MODEL is None:
        import lpips

        _LPIPS_CPU_MODEL = lpips.LPIPS(net="vgg").eval()
    return _LPIPS_CPU_MODEL


def resolve_config_path(config_arg: str) -> Path:
    candidate = Path(config_arg)
    if candidate.is_file():
        return candidate.resolve()
    under_configs = ROOT / "configs" / config_arg
    if under_configs.is_file():
        return under_configs.resolve()
    raise FileNotFoundError(f"Config file not found: {config_arg}")


def resolve_mask_path(mask_arg: str, exp_dir: Path) -> Path:
    candidate = Path(mask_arg)
    if candidate.is_file():
        return candidate.resolve()
    under_exp = exp_dir / "inpainting_masks" / mask_arg
    if under_exp.is_file():
        return under_exp.resolve()
    under_root = ROOT / "exp" / "inpainting_masks" / mask_arg
    if under_root.is_file():
        return under_root.resolve()
    raise FileNotFoundError(f"Inpainting mask not found: {mask_arg}")


def load_and_validate_mask(mask_path: Path, image_size: int) -> np.ndarray:
    arr = np.load(mask_path, allow_pickle=False)
    if not isinstance(arr, np.ndarray) or arr.shape != (image_size, image_size):
        raise ValueError(
            f"Expected 2D mask of shape {(image_size, image_size)} in {mask_path}, "
            f"got {getattr(arr, 'shape', type(arr).__name__)}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"Mask {mask_path} contains non-finite values")
    if not np.any(arr == 0):
        raise ValueError(f"Mask {mask_path} has no missing pixels (mask == 0)")
    if not np.any(arr != 0):
        raise ValueError(f"Mask {mask_path} has no observed pixels (mask != 0)")
    return arr


def _validate_image_file(path: Path, image_size: int) -> None:
    with Image.open(path) as image:
        if image.size != (image_size, image_size):
            raise ValueError(f"{path}: expected {image_size}x{image_size}, got {image.size}")
        rgb = image.convert("RGB")
        rgb.load()


def inventory(
    dataset_dir: Path,
    expected_count: Optional[int],
    inputs_manifest: Optional[Path] = None,
    image_size: int = 256,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Validate and order input images from dataset_dir (or frozen --inputs-manifest)."""
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory missing: {dataset_dir}")

    paths: list[Path] = []
    rows: list[dict[str, str]] = []
    seen_hashes: set[str] = set()

    if inputs_manifest is not None:
        if not inputs_manifest.is_file():
            raise FileNotFoundError(f"Inputs manifest not found: {inputs_manifest}")
        manifest_data = json.loads(inputs_manifest.read_text(encoding="utf-8"))
        manifest_inputs = manifest_data.get("inputs")
        if not isinstance(manifest_inputs, list) or not manifest_inputs:
            raise ValueError(f"Inputs manifest {inputs_manifest} has empty or invalid 'inputs'")
        manifest_expected = manifest_data.get("expected_count", len(manifest_inputs))
        if len(manifest_inputs) != int(manifest_expected):
            raise ValueError(
                f"Inputs manifest {inputs_manifest} lists {len(manifest_inputs)} inputs "
                f"but expected_count={manifest_expected}"
            )
        if expected_count is not None and len(manifest_inputs) != expected_count:
            raise ValueError(
                f"Expected {expected_count} images, but manifest {inputs_manifest} has {len(manifest_inputs)}"
            )
        for entry in manifest_inputs:
            rel_input = str(entry["input"])
            img_path = (dataset_dir / rel_input).resolve()
            if not img_path.is_file():
                raise FileNotFoundError(f"Manifest input missing on disk: {img_path}")
            _validate_image_file(img_path, image_size=image_size)
            sha = digest(img_path)
            if "sha256" in entry and sha != entry["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {rel_input}: expected {entry['sha256']}, got {sha}"
                )
            if sha in seen_hashes:
                raise ValueError(f"Byte-identical duplicate in manifest: {img_path}")
            seen_hashes.add(sha)
            paths.append(img_path)
            rows.append({"input": rel_input, "sha256": sha})
        return paths, rows

    all_paths = sorted(
        (
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ),
        key=lambda path: path.relative_to(dataset_dir).as_posix(),
    )
    if expected_count is not None and len(all_paths) != expected_count:
        raise ValueError(f"Expected {expected_count} images, found {len(all_paths)} in {dataset_dir}")
    if not all_paths:
        raise ValueError(f"No valid images found in {dataset_dir}")

    for path in all_paths:
        _validate_image_file(path, image_size=image_size)
        sha = digest(path)
        if sha in seen_hashes:
            raise ValueError(f"Byte-identical duplicate: {path}")
        seen_hashes.add(sha)
        paths.append(path)
        rows.append({"input": path.relative_to(dataset_dir).as_posix(), "sha256": sha})
    return paths, rows


def build_imagefolder_index_map(dataset_dir: Path) -> dict[str, int]:
    """Compute exact 0-based torchvision.datasets.ImageFolder index map for dataset_dir."""
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    samples: list[str] = []
    if class_dirs:
        for class_dir in class_dirs:
            files = sorted(
                (p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
                key=lambda p: p.relative_to(dataset_dir).as_posix(),
            )
            samples.extend(p.relative_to(dataset_dir).as_posix() for p in files)
    else:
        files = sorted(
            (p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
            key=lambda p: p.relative_to(dataset_dir).as_posix(),
        )
        samples.extend(p.relative_to(dataset_dir).as_posix() for p in files)
    return {rel: idx for idx, rel in enumerate(samples)}


def parse_deltas(deltas_str: str, t_sampling: int) -> list[float]:
    if not deltas_str or not deltas_str.strip():
        return [0.0] * t_sampling
    values = [float(item.strip()) for item in deltas_str.split(",")]
    if len(values) != t_sampling:
        raise ValueError(
            f"deltas length must equal T_sampling ({t_sampling}); got {len(values)} from {deltas_str!r}"
        )
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"deltas contains non-finite values: {values}")
    return values


def parse_clip_steps(clip_steps_val: Any) -> int:
    if clip_steps_val is None or clip_steps_val == "":
        return 1
    if isinstance(clip_steps_val, int):
        return clip_steps_val
    s = str(clip_steps_val).strip()
    if not s:
        return 1
    if "," in s:
        parts = [int(item.strip()) for item in s.split(",") if item.strip()]
        return max(1, len(parts))
    return int(s)


def build_context_params(args: argparse.Namespace) -> dict[str, Any]:
    if args.context_mode == "none":
        return {}
    if args.context_mode == "telea":
        return {
            "telea_method": str(args.telea_method),
            "telea_radius": float(args.telea_radius),
        }
    if args.context_mode == "spatial":
        return {
            "spatial_sigma": float(args.spatial_sigma),
            "eta_hole": float(args.eta_hole),
            "eta_obs": float(args.eta_obs),
        }
    if args.context_mode == "clip":
        return {
            "clip_model": str(args.clip_model),
            "clip_pretrained": str(args.clip_pretrained),
            "clip_prompt": str(args.clip_prompt),
            "clip_neg_prompt": args.clip_neg_prompt,
            "clip_negative_control": bool(args.clip_negative_control),
            "clip_scale": float(args.clip_scale),
            "clip_steps": parse_clip_steps(args.clip_steps),
        }
    if args.context_mode == "poisson":
        return {
            "eta_obs": float(args.eta_obs),
            "contiguous_radius_threshold_px": 16.0,
            "contiguous_edm_sigmas": [3.0, 1.2, 0.4, 0.05],
        }
    if args.context_mode in ("poisson_bridge", "poisson_bridge_sharp"):
        return {
            "eta_obs": float(args.eta_obs),
            "contiguous_radius_threshold_px": 16.0,
            "contiguous_edm_sigmas": [3.5, 1.6, 0.55, 0.08],
            "bridge_gamma": 5.0,
            "bridge_alphas": [0.40, 0.18, 0.0, 0.0],
            "bridge_blur_sigma": 12.0,
            "contiguous_eta_holes": [1.0, 1.0, 1.0] if args.context_mode == "poisson_bridge_sharp" else [0.95, 1.0, 1.0],
        }
    raise ValueError(f"Unsupported context_mode: {args.context_mode}")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _to_tensor_4d(tensor_like: Any) -> torch.Tensor:
    t = torch.as_tensor(tensor_like, dtype=torch.float32).detach().cpu()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"Expected 3D or 4D image tensor, got shape {tuple(t.shape)}")
    return t


def compute_metrics_from_tensors(
    rec: dict[str, Any],
    mask_2d: np.ndarray,
    skip_lpips: bool,
) -> dict[str, Optional[float]]:
    """Compute any missing metrics from tensor fields in rec using mask == 0 hole semantics."""
    hole_2d = torch.from_numpy(mask_2d == 0)
    obs_2d = torch.from_numpy(mask_2d != 0)

    x_orig_01 = _to_tensor_4d(rec["x_orig_01"]) if "x_orig_01" in rec else None
    x_hat_01 = _to_tensor_4d(rec["x_hat_01"]) if "x_hat_01" in rec else None
    x_orig_m11 = _to_tensor_4d(rec["x_orig_m11"]) if "x_orig_m11" in rec else None
    x_hat_m11 = _to_tensor_4d(rec["x_hat_m11"]) if "x_hat_m11" in rec else None
    apy_01 = _to_tensor_4d(rec["Apy_01"]) if "Apy_01" in rec and rec["Apy_01"] is not None else None

    for name, tensor in (
        ("x_orig_01", x_orig_01),
        ("x_hat_01", x_hat_01),
        ("x_orig_m11", x_orig_m11),
        ("x_hat_m11", x_hat_m11),
        ("Apy_01", apy_01),
    ):
        if tensor is not None and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite values in tensor '{name}' for sample index {rec.get('index')}")

    if x_orig_01 is None and x_orig_m11 is not None:
        x_orig_01 = torch.clamp((x_orig_m11 + 1.0) / 2.0, 0.0, 1.0)
    if x_hat_01 is None and x_hat_m11 is not None:
        x_hat_01 = torch.clamp((x_hat_m11 + 1.0) / 2.0, 0.0, 1.0)
    if x_orig_m11 is None and x_orig_01 is not None:
        x_orig_m11 = x_orig_01 * 2.0 - 1.0
    if x_hat_m11 is None and x_hat_01 is not None:
        x_hat_m11 = x_hat_01 * 2.0 - 1.0

    # 1. full_psnr_db
    full_psnr_db: Optional[float]
    if "full_psnr_db" in rec:
        val = rec["full_psnr_db"]
        if val is None:
            full_psnr_db = None
        elif _is_finite_number(val):
            full_psnr_db = float(val)
        elif x_hat_01 is not None and x_orig_01 is not None and float(torch.mean((x_hat_01 - x_orig_01) ** 2).item()) == 0.0:
            full_psnr_db = None
        else:
            raise ValueError(f"Non-finite full_psnr_db: {val}")
    else:
        if x_hat_01 is None or x_orig_01 is None:
            raise KeyError("Missing full_psnr_db and tensor fallback (x_hat_01, x_orig_01)")
        mse_full = float(torch.mean((x_hat_01 - x_orig_01) ** 2).item())
        full_psnr_db = psnr(mse_full)

    # 2. hole_psnr_db
    hole_psnr_db: Optional[float]
    if "hole_psnr_db" in rec:
        val = rec["hole_psnr_db"]
        if val is None:
            hole_psnr_db = None
        elif _is_finite_number(val):
            hole_psnr_db = float(val)
        elif x_hat_01 is not None and x_orig_01 is not None:
            b, c, h, w = x_hat_01.shape
            hole_4d = hole_2d.view(1, 1, h, w).expand(b, c, h, w)
            if hole_4d.any() and float(torch.mean((x_hat_01[hole_4d] - x_orig_01[hole_4d]) ** 2).item()) == 0.0:
                hole_psnr_db = None
            else:
                raise ValueError(f"Non-finite hole_psnr_db: {val}")
        else:
            raise ValueError(f"Non-finite hole_psnr_db: {val}")
    else:
        if x_hat_01 is None or x_orig_01 is None:
            raise KeyError("Missing hole_psnr_db and tensor fallback (x_hat_01, x_orig_01)")
        if not hole_2d.any():
            hole_psnr_db = None
        else:
            b, c, h, w = x_hat_01.shape
            hole_4d = hole_2d.view(1, 1, h, w).expand(b, c, h, w)
            mse_hole = float(torch.mean((x_hat_01[hole_4d] - x_orig_01[hole_4d]) ** 2).item())
            hole_psnr_db = psnr(mse_hole)

    # 3. full_vgg_lpips & 4. hole_vgg_lpips
    if skip_lpips:
        full_vgg_lpips = None
        hole_vgg_lpips = None
    else:
        if "full_vgg_lpips" in rec and rec["full_vgg_lpips"] is not None:
            val = rec["full_vgg_lpips"]
            if not _is_finite_number(val):
                raise ValueError(f"Non-finite full_vgg_lpips: {val}")
            full_vgg_lpips = float(val)
        else:
            if x_hat_m11 is None or x_orig_m11 is None:
                raise KeyError("Missing full_vgg_lpips and tensor fallback (x_hat_m11, x_orig_m11)")
            lpips_model = _get_lpips_cpu()
            with torch.no_grad():
                full_vgg_lpips = float(lpips_model(x_hat_m11, x_orig_m11).mean().item())

        if "hole_vgg_lpips" in rec and rec["hole_vgg_lpips"] is not None:
            val = rec["hole_vgg_lpips"]
            if not _is_finite_number(val):
                raise ValueError(f"Non-finite hole_vgg_lpips: {val}")
            hole_vgg_lpips = float(val)
        else:
            if not hole_2d.any():
                hole_vgg_lpips = None
            else:
                if x_hat_m11 is None or x_orig_m11 is None:
                    raise KeyError("Missing hole_vgg_lpips and tensor fallback (x_hat_m11, x_orig_m11)")
                b, c, h, w = x_hat_m11.shape
                hole_4d = hole_2d.view(1, 1, h, w).expand(b, c, h, w)
                x_hat_hole_comp = torch.where(hole_4d, x_hat_m11, x_orig_m11)
                lpips_model = _get_lpips_cpu()
                with torch.no_grad():
                    hole_vgg_lpips = float(lpips_model(x_hat_hole_comp, x_orig_m11).mean().item())

    # 5. observed_max_abs_diff
    if "observed_max_abs_diff" in rec and rec["observed_max_abs_diff"] is not None:
        val = rec["observed_max_abs_diff"]
        if not _is_finite_number(val):
            raise ValueError(f"Non-finite observed_max_abs_diff: {val}")
        observed_max_abs_diff = float(val)
    else:
        if not obs_2d.any():
            observed_max_abs_diff = 0.0
        elif x_hat_01 is not None and apy_01 is not None:
            b, c, h, w = x_hat_01.shape
            obs_4d = obs_2d.view(1, 1, h, w).expand(b, c, h, w)
            observed_max_abs_diff = float(torch.max(torch.abs(x_hat_01[obs_4d] - apy_01[obs_4d])).item())
        elif x_hat_01 is not None and x_orig_01 is not None:
            b, c, h, w = x_hat_01.shape
            obs_4d = obs_2d.view(1, 1, h, w).expand(b, c, h, w)
            observed_max_abs_diff = float(torch.max(torch.abs(x_hat_01[obs_4d] - x_orig_01[obs_4d])).item())
        else:
            raise KeyError("Missing observed_max_abs_diff and tensor fallback (x_hat_01, Apy_01/x_orig_01)")

    return {
        "full_psnr_db": full_psnr_db,
        "full_vgg_lpips": full_vgg_lpips,
        "hole_psnr_db": hole_psnr_db,
        "hole_vgg_lpips": hole_vgg_lpips,
        "observed_max_abs_diff": observed_max_abs_diff,
    }


def _is_valid_cached_row(
    row: Any,
    index: int,
    entry: dict[str, str],
    restored_path: Path,
    skip_lpips: bool,
) -> bool:
    if not isinstance(row, dict):
        return False
    if any(field not in row for field in RECORD_FIELDS):
        return False
    if row.get("index") != index:
        return False
    if row.get("input") != entry["input"]:
        return False
    if row.get("sha256") != entry["sha256"]:
        return False
    expected_rel = f"images/{index}_0.png"
    if row.get("restored") != expected_rel:
        return False
    if not restored_path.is_file():
        return False
    if digest(restored_path) != row.get("restored_sha256"):
        return False

    for nullable_key in ("full_psnr_db", "hole_psnr_db", "full_vgg_lpips", "hole_vgg_lpips"):
        val = row.get(nullable_key)
        if val is not None and not _is_finite_number(val):
            return False
    if not _is_finite_number(row.get("observed_max_abs_diff")):
        return False
    if not skip_lpips:
        if row.get("full_vgg_lpips") is None or row.get("hole_vgg_lpips") is None:
            return False
    return True


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _relative_or_str(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_config_path(args.config)
    config = dict2namespace(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config.data.num_workers = int(args.workers)
    image_size = int(getattr(config.data, "image_size", 256))
    t_sampling = int(getattr(config.sampling, "T_sampling", 4))

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint_sha = digest(args.checkpoint)
    if args.verify_official_checkpoint and checkpoint_sha != OFFICIAL_LSUN_CAT_CHECKPOINT_SHA:
        raise ValueError(
            f"Checkpoint SHA-256 {checkpoint_sha} differs from official LSUN Cat checkpoint "
            f"{OFFICIAL_LSUN_CAT_CHECKPOINT_SHA}"
        )

    mask_path = resolve_mask_path(args.mask, args.exp)
    mask_2d = load_and_validate_mask(mask_path, image_size=image_size)
    mask_sha = digest(mask_path)

    paths, inputs = inventory(
        args.dataset,
        expected_count=args.expected_count,
        inputs_manifest=args.inputs_manifest,
        image_size=image_size,
    )

    deltas_list = parse_deltas(args.deltas, t_sampling=t_sampling)
    context_params = build_context_params(args)
    device = _select_device(args.device)

    gpu_name = (
        torch.cuda.get_device_name(0)
        if device.type == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    manifest = {
        "git_commit": get_git_commit(),
        "config_sha256": digest(config_path),
        "checkpoint_sha256": checkpoint_sha,
        "mask_path": _relative_or_str(mask_path, ROOT),
        "mask_sha256": mask_sha,
        "context_mode": str(args.context_mode),
        "sigma_y": float(args.sigma_y),
        "iN": int(args.iN),
        "gamma": float(args.gamma),
        "zeta": float(args.zeta),
        "eta": float(args.eta),
        "deltas": deltas_list,
        "deltas_injection_type": int(args.deltas_injection_type),
        "seed": int(args.seed),
        "seed_rule": "seed + sorted zero-based sample index",
        "context_params": context_params,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda": torch.version.cuda,
            "gpu": gpu_name,
        },
        "inputs": inputs,
    }

    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError("Resume manifest differs. Use the original configuration or a fresh --output")
    else:
        if any(path.name != ".run.lock" for path in args.output.iterdir()):
            raise ValueError("Choose an empty output directory")
        write_json(manifest_path, manifest)

    records_dir = args.output / "records"
    records_dir.mkdir(exist_ok=True)
    images_dir = args.output / "images"
    images_dir.mkdir(exist_ok=True)

    rows: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    for index, entry in enumerate(inputs):
        record_path = records_dir / f"{index}.json"
        restored_path = images_dir / f"{index}_0.png"
        row = None
        if record_path.is_file() and restored_path.is_file():
            try:
                row = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                row = None
        if _is_valid_cached_row(row, index=index, entry=entry, restored_path=restored_path, skip_lpips=args.skip_lpips):
            rows[index] = row
        else:
            pending.append(index)

    if pending:
        diffusion_cls = Diffusion
        if diffusion_cls is None:
            from guided_diffusion.diffusion import Diffusion as _RealDiffusion

            diffusion_cls = _RealDiffusion

        try:
            model_ckpt_rel = args.checkpoint.resolve().relative_to((args.exp / "logs").resolve()).as_posix()
        except ValueError:
            model_ckpt_rel = str(args.checkpoint)

        try:
            mask_rel = mask_path.resolve().relative_to((args.exp / "inpainting_masks").resolve()).as_posix()
        except ValueError:
            mask_rel = str(mask_path)

        settings = SimpleNamespace(
            observed=None,
            subset_start=-1,
            subset_end=-1,
            seed=int(args.seed),
            deltas=str(args.deltas),
            deltas_injection_type=int(args.deltas_injection_type),
            deg="inpainting",
            operator_imp="SVD",
            deg_scale=4.0,
            sigma_y=float(args.sigma_y),
            zeta=float(args.zeta),
            eta=float(args.eta),
            gamma=float(args.gamma),
            iN=int(args.iN),
            save_observed_img=False,
            skip_lpips=bool(args.skip_lpips),
            image_folder=str(images_dir),
            exp=str(args.exp),
            path_y=args.dataset.name,
            dataset_dir=str(args.dataset),
            model_ckpt=model_ckpt_rel,
            checkpoint=str(args.checkpoint),
            cache_model=True,
            inpainting_mask_path=mask_rel,
            sample_ids=list(pending),
            workers=int(args.workers),
            num_workers=int(args.workers),
            context_mode=str(args.context_mode),
            telea_method=str(args.telea_method),
            telea_radius=float(args.telea_radius),
            spatial_sigma=float(args.spatial_sigma),
            eta_hole=float(args.eta_hole),
            eta_obs=float(args.eta_obs),
            clip_model=str(args.clip_model),
            clip_pretrained=str(args.clip_pretrained),
            clip_prompt=str(args.clip_prompt),
            clip_neg_prompt=args.clip_neg_prompt,
            clip_negative_control=bool(args.clip_negative_control),
            clip_scale=float(args.clip_scale),
            clip_steps=parse_clip_steps(args.clip_steps),
        )

        def on_result(
            index: int,
            restored: Any,
            *pos_args: Any,
            **kw_args: Any,
        ) -> None:
            manifest_idx = rec_index_to_manifest.get(int(index), int(index))
            if manifest_idx in rows:
                return
            rec_payload: dict[str, Any] = {"index": int(index), "restored_path": str(restored)}
            rec_payload.update(kw_args)
            restored_p = Path(restored)
            target_p = images_dir / f"{manifest_idx}_0.png"
            if restored_p.resolve() != target_p.resolve():
                shutil.copy2(restored_p, target_p)
            metrics = compute_metrics_from_tensors(rec_payload, mask_2d=mask_2d, skip_lpips=args.skip_lpips)
            entry = inputs[manifest_idx]
            row = {
                "index": manifest_idx,
                "input": entry["input"],
                "sha256": entry["sha256"],
                "restored": f"images/{manifest_idx}_0.png",
                "restored_sha256": digest(target_p),
                **metrics,
            }
            write_json(records_dir / f"{manifest_idx}.json", row)
            rows[manifest_idx] = row

        init_params = inspect.signature(diffusion_cls.__init__).parameters
        if len(init_params) >= 4 or "device" in init_params or any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in init_params.values()
        ):
            runner = diffusion_cls(settings, config, device)
        else:
            runner = diffusion_cls(settings, config)

        sample_params = inspect.signature(runner.sample).parameters
        accepts_dataset = "dataset" in sample_params
        accepts_sample_ids = "sample_ids" in sample_params
        accepts_checkpoint = "checkpoint" in sample_params
        accepts_on_result = "on_result" in sample_params

        if accepts_dataset and accepts_sample_ids:
            settings.sample_ids = list(pending)
            rec_index_to_manifest = {idx: idx for idx in pending}
            sample_kwargs: dict[str, Any] = {
                "dataset": CleanImages([paths[idx] for idx in pending]),
                "sample_ids": list(pending),
            }
            if accepts_checkpoint:
                sample_kwargs["checkpoint"] = str(args.checkpoint)
            if accepts_on_result:
                sample_kwargs["on_result"] = on_result
            runner.sample(logging.getLogger("cm4ir"), **sample_kwargs)
        else:
            imagefolder_map = build_imagefolder_index_map(args.dataset)
            dataset_ids: list[int] = []
            rec_index_to_manifest = {}
            for man_idx in pending:
                rel_input = inputs[man_idx]["input"]
                if rel_input not in imagefolder_map:
                    raise ValueError(f"Input {rel_input} not found in dataset {args.dataset}")
                ds_idx = imagefolder_map[rel_input]
                dataset_ids.append(ds_idx)
                rec_index_to_manifest[ds_idx] = man_idx
            settings.sample_ids = dataset_ids
            sample_kwargs = {}
            if accepts_checkpoint:
                sample_kwargs["checkpoint"] = str(args.checkpoint)
            if accepts_sample_ids:
                sample_kwargs["sample_ids"] = dataset_ids
            if accepts_on_result:
                sample_kwargs["on_result"] = on_result
            runner.sample(logging.getLogger("cm4ir"), **sample_kwargs)

        last_records = getattr(runner, "last_run_records", None) or []
        # Normalize records and perform collision-free two-phase rename/copy when dataset_idx != manifest_idx.
        normalized: list[tuple[int, dict[str, Any], Path]] = []
        target_manifest_indices: set[int] = set()
        for rec in last_records:
            raw_idx = int(rec["index"])
            if raw_idx not in rec_index_to_manifest:
                raise RuntimeError(f"Diffusion returned unexpected sample index {raw_idx}")
            man_idx = rec_index_to_manifest[raw_idx]
            src_path = Path(rec.get("restored_path") or (images_dir / f"{raw_idx}_0.png"))
            if not src_path.is_file():
                raise FileNotFoundError(f"Restored image missing for index {raw_idx}: {src_path}")
            normalized.append((man_idx, rec, src_path))
            target_manifest_indices.add(man_idx)

        # Stage any file whose destination differs from its source path.
        staged_paths: dict[int, Path] = {}
        sources_to_cleanup: list[Path] = []
        for man_idx, _, src_path in normalized:
            dst_path = images_dir / f"{man_idx}_0.png"
            if src_path.resolve() != dst_path.resolve():
                stage_path = images_dir / f".stage_{man_idx}_0.png"
                shutil.copy2(src_path, stage_path)
                staged_paths[man_idx] = stage_path
                if src_path.parent.resolve() == images_dir.resolve():
                    sources_to_cleanup.append(src_path)
        for man_idx, stage_path in staged_paths.items():
            dst_path = images_dir / f"{man_idx}_0.png"
            os.replace(stage_path, dst_path)
        for src_path in sources_to_cleanup:
            if src_path.exists():
                stem_parts = src_path.name.split("_")
                is_active_target = (
                    len(stem_parts) == 2
                    and stem_parts[1] == "0.png"
                    and stem_parts[0].isdigit()
                    and int(stem_parts[0]) in target_manifest_indices
                )
                if not is_active_target:
                    src_path.unlink()

        for man_idx, rec, _ in normalized:
            target_path = images_dir / f"{man_idx}_0.png"
            if not target_path.is_file():
                raise FileNotFoundError(f"Target restored image missing: {target_path}")
            metrics = compute_metrics_from_tensors(rec, mask_2d=mask_2d, skip_lpips=args.skip_lpips)
            entry = inputs[man_idx]
            row = {
                "index": man_idx,
                "input": entry["input"],
                "sha256": entry["sha256"],
                "restored": f"images/{man_idx}_0.png",
                "restored_sha256": digest(target_path),
                **metrics,
            }
            write_json(records_dir / f"{man_idx}.json", row)
            rows[man_idx] = row

    if len(rows) != len(inputs):
        missing = [i for i in range(len(inputs)) if i not in rows]
        raise RuntimeError(f"Only {len(rows)} of {len(inputs)} images completed; missing indices={missing}")

    ordered = [rows[index] for index in range(len(inputs))]
    write_csv(args.output / "results.csv", ordered)

    summary = {
        "completed": len(ordered),
        "mean_full_psnr_db": safe_mean(ordered, "full_psnr_db"),
        "mean_full_vgg_lpips": None if args.skip_lpips else safe_mean(ordered, "full_vgg_lpips"),
        "mean_hole_psnr_db": safe_mean(ordered, "hole_psnr_db"),
        "mean_hole_vgg_lpips": None if args.skip_lpips else safe_mean(ordered, "hole_vgg_lpips"),
        "mean_observed_max_abs_diff": safe_mean(ordered, "observed_max_abs_diff"),
        "max_observed_max_abs_diff": float(max(r["observed_max_abs_diff"] for r in ordered)) if ordered else None,
        "samples": ordered,
    }
    write_json(args.output / "summary.json", summary)
    print(f"Verified {len(ordered)} images. Summary saved in {args.output / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "exp" / "datasets" / "lsun_cat")
    parser.add_argument("--inputs-manifest", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--config", type=str, default="lsun_cat_256.yml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp" / "logs" / "lsun_cat" / "cd_cat256_lpips.pt")
    parser.add_argument("--verify-official-checkpoint", action="store_true")
    parser.add_argument("--mask", "--inpainting-mask-path", dest="mask", type=str, default="random_80_mask.npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exp", type=Path, default=ROOT / "exp")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--skip-lpips", action="store_true", help="Skip VGG-LPIPS evaluation (stores null)")
    parser.add_argument("--sigma-y", type=float, default=0.05)
    parser.add_argument("--iN", type=int, default=150)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--zeta", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--deltas", type=str, default="")
    parser.add_argument("--deltas-injection-type", type=int, default=0)
    parser.add_argument("--context-mode", choices=("none", "telea", "spatial", "clip", "poisson", "poisson_bridge", "poisson_bridge_sharp"), default="none")
    parser.add_argument("--telea-method", choices=("telea", "ns"), default="telea")
    parser.add_argument("--telea-radius", type=float, default=3.0)
    parser.add_argument("--spatial-sigma", type=float, default=8.0)
    parser.add_argument("--eta-hole", type=float, default=0.25)
    parser.add_argument("--eta-obs", type=float, default=0.02)
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--clip-prompt", type=str, default="a photograph of a cat")
    parser.add_argument("--clip-neg-prompt", type=str, default="a photograph of a red fire hydrant in outer space")
    parser.add_argument("--clip-negative-control", action="store_true")
    parser.add_argument("--clip-scale", type=float, default=0.25)
    parser.add_argument("--clip-steps", type=int, default=1)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.expected_count is None and args.inputs_manifest is None:
        args.expected_count = 57
    if args.expected_count is not None and args.expected_count < 1:
        parser.error("--expected-count must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    seed_bound = args.expected_count if args.expected_count is not None else 1
    if not 0 <= args.seed <= 2**63 - 1 - seed_bound:
        parser.error("Invalid --seed range")
    if not math.isfinite(args.sigma_y) or args.sigma_y < 0:
        parser.error("--sigma-y must be finite and non-negative")

    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.exp = args.exp.resolve()
    if args.inputs_manifest is not None:
        args.inputs_manifest = args.inputs_manifest.resolve()

    if args.output.is_relative_to(args.dataset):
        parser.error("Keep --output outside the dataset directory")

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    with FileLock(args.output / ".run.lock", timeout=0):
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
