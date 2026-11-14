"""Evaluate a complete clean LSUN Bedroom split with CM4IR Gaussian deblurring."""

import argparse
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from filelock import FileLock
from PIL import Image

from deblur import PRESETS, ROOT

sys.path.insert(0, str(ROOT))

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CHECKPOINT_SHA = "61238b8de0ba4976ec35a60fd4786a02a862df0960aa8a2554467e9486e71252"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def inventory(root, expected_count):
    if not root.is_dir():
        raise ValueError(f"Dataset directory missing: {root}")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()
         and path.suffix.lower() in EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} images, found {len(paths)} in {root}")
    rows = []
    seen = set()
    for path in paths:
        with Image.open(path) as image:
            if image.size != (256, 256):
                raise ValueError(f"{path}: expected 256×256, got {image.size}")
            image.load()
        sha = digest(path)
        if sha in seen:
            raise ValueError(f"Byte-identical duplicate: {path}")
        seen.add(sha)
        rows.append({"input": path.relative_to(root).as_posix(), "sha256": sha})
    return paths, rows


class CleanImages:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from torchvision.transforms.functional import to_tensor

        with Image.open(self.paths[index]) as image:
            return to_tensor(image.convert("RGB")), 0


def psnr(mse):
    # JSON null represents a mathematically infinite PSNR.
    return -10 * math.log10(mse) if mse else None


def average_psnr(rows, key):
    values = [row[key] for row in rows]
    return None if None in values else mean(values)


def run(args):
    import torch
    import torchvision
    import yaml

    from guided_diffusion.diffusion import Diffusion
    from main import dict2namespace

    paths, inputs = inventory(args.dataset, args.expected_count)
    if digest(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Checkpoint SHA-256 differs from the verified official Bedroom checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    iN, gamma, zeta = PRESETS[args.preset, args.noise]
    config_path = ROOT / "configs/lsun_bedroom_256.yml"
    config = dict2namespace(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config.data.num_workers = args.workers
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": digest(config_path),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "preset": args.preset, "sigma_y": float(args.noise), "iN": int(iN),
        "gamma": float(gamma), "zeta": float(zeta), "eta": 0.1,
        "seed": args.seed, "seed_rule": "seed + sorted zero-based sample index",
        "lpips": args.lpips,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0)},
        "inputs": inputs,
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume manifest differs. Use the original configuration or a fresh --output")
    else:
        if any(path.name != ".run.lock" for path in args.output.iterdir()):
            raise ValueError("Choose an empty output directory")
        write_json(manifest_path, manifest)

    records = args.output / "records"
    records.mkdir(exist_ok=True)
    images = args.output / "images"
    images.mkdir(exist_ok=True)
    rows = {}
    for index, entry in enumerate(inputs):
        record = records / f"{index}.json"
        restored = images / f"{index}_0.png"
        if record.is_file() and restored.is_file():
            row = json.loads(record.read_text(encoding="utf-8"))
            if (row["index"] == index and row["input"] == entry["input"]
                    and row["sha256"] == entry["sha256"]
                    and digest(restored) == row["restored_sha256"]):
                rows[index] = row

    def on_result(index, restored, blurred_mse, restored_mse, lpips):
        entry = inputs[index]
        restored = Path(restored)
        row = {
            "index": index, **entry,
            "restored": restored.relative_to(args.output).as_posix(),
            "restored_sha256": digest(restored),
            "blurred_psnr_db": psnr(blurred_mse),
            "restored_psnr_db": psnr(restored_mse),
            "vgg_lpips": lpips,
        }
        write_json(records / f"{index}.json", row)
        rows[index] = row

    pending = [index for index in range(len(inputs)) if index not in rows]
    if pending:
        settings = SimpleNamespace(
            observed=None, subset_start=-1, subset_end=-1, seed=args.seed,
            deltas="", deltas_injection_type=0, deg="deblur_gauss", operator_imp="SVD",
            sigma_y=float(args.noise), zeta=float(zeta), eta=0.1, gamma=float(gamma),
            iN=int(iN), save_observed_img=False, skip_lpips=not args.lpips,
            image_folder=str(images),
        )
        Diffusion(settings, config, torch.device("cuda:0")).sample(
            logging.getLogger("cm4ir"), checkpoint=args.checkpoint,
            dataset=CleanImages([paths[index] for index in pending]),
            sample_ids=pending, on_result=on_result,
        )
    if len(rows) != len(inputs):
        raise RuntimeError(f"Only {len(rows)} of {len(inputs)} images completed")
    ordered = [rows[index] for index in range(len(inputs))]
    summary = {
        "completed": len(ordered),
        "mean_blurred_psnr_db": average_psnr(ordered, "blurred_psnr_db"),
        "mean_restored_psnr_db": average_psnr(ordered, "restored_psnr_db"),
        "mean_vgg_lpips": mean(row["vgg_lpips"] for row in ordered) if args.lpips else None,
        "samples": ordered,
    }
    write_json(args.output / "summary.json", summary)
    print(f"Verified {len(ordered)} images. Metrics saved in {args.output / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--preset", choices=("repo", "paper"), default="paper")
    parser.add_argument("--noise", choices=("0.025", "0.05"), default="0.025")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()
    if args.expected_count < 1 or args.workers < 0 or not 0 <= args.seed <= 2**63 - 1 - args.expected_count:
        parser.error("Expected a positive count, a nonnegative worker count and a valid seed")
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.dataset):
        parser.error("Keep --output outside the dataset directory")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    with FileLock(args.output / ".run.lock", timeout=0):
        run(args)


if __name__ == "__main__":
    main()
