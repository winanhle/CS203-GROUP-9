"""CPU-only unit and contract tests for M3 masks, resumable runner, and analysis suite."""

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from filelock import FileLock, Timeout
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analyze_inpainting_results as analysis_mod
import scripts.create_masks_and_manifest as mask_mod
import scripts.run_inpainting_ablation as runner_mod


def _reject_nan_constant(token: str):
    raise ValueError(f"Invalid JSON constant detected: {token}")


def _loads_strict(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nan_constant)


def _write_synthetic_png(path: Path, arr_uint8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint8, mode="RGB").save(path, format="PNG")


def _make_synthetic_dataset(root: Path, count: int = 4) -> list[Path]:
    cat_dir = root / "cat"
    cat_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        arr = np.full((256, 256, 3), 40 + i * 35, dtype=np.uint8)
        arr[32:96, 32:96, 0] = min(255, 80 + i * 40)
        p = cat_dir / f"{i}.png"
        _write_synthetic_png(p, arr)
        paths.append(p)
    return paths


def _make_dummy_config(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "data:",
                '    dataset: "lsun_cat"',
                "    image_size: 256",
                "    channels: 3",
                "    num_workers: 4",
                "sampling:",
                "    batch_size: 1",
                "    T_sampling: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _make_dummy_checkpoint(path: Path) -> Path:
    path.write_bytes(b"synthetic-checkpoint-bytes-for-cpu-test")
    return path


def _make_test_env(tmp_path: Path, count: int = 4, mask_name: str = "center_box_mask.npy"):
    dataset_dir = tmp_path / "dataset"
    _make_synthetic_dataset(dataset_dir, count=count)
    exp_dir = tmp_path / "exp"
    masks_dir = exp_dir / "inpainting_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask = mask_mod.build_center_box_mask()
    np.save(masks_dir / mask_name, mask)
    cfg = _make_dummy_config(tmp_path / "cfg.yml")
    ckpt = _make_dummy_checkpoint(tmp_path / "ckpt.pt")
    return dataset_dir, exp_dir, cfg, ckpt, mask


# ---------------------------------------------------------------------------
# 1. Mask & Manifest Tests
# ---------------------------------------------------------------------------


def test_center_box_mask_geometry_and_dtype():
    mask = mask_mod.build_center_box_mask()
    assert mask.shape == (256, 256)
    assert mask.dtype == np.dtype(np.float32)
    assert mask.flags["C_CONTIGUOUS"]
    assert int(np.sum(mask == 0)) == 128 * 128 == 16384
    assert int(np.sum(mask != 0)) == 256 * 256 - 16384 == 49152
    assert np.all(mask[64:192, 64:192] == 0.0)
    outside = mask.copy()
    outside[64:192, 64:192] = 1.0
    assert np.all(outside == 1.0)

    repo_mask_path = ROOT / "exp" / "inpainting_masks" / "center_box_mask.npy"
    if repo_mask_path.is_file():
        info = mask_mod.verify_center_box_mask(repo_mask_path)
        assert info["zero_count"] == 16384
        assert info["nonzero_count"] == 49152
        assert repo_mask_path.stat().st_size == 262272


def test_mask_convention_consistency():
    masks_dir = ROOT / "exp" / "inpainting_masks"
    rand_mask = np.load(masks_dir / "random_80_mask.npy")
    lorem_mask = np.load(masks_dir / "lorem_ipsum_mask.npy")
    box_mask = mask_mod.build_center_box_mask()

    assert rand_mask.shape == (256, 256)
    assert int(np.sum(rand_mask == 0)) == 52428
    assert int(np.sum(rand_mask != 0)) == 13108

    # Crucial: anti-aliased values 1..254 in lorem_ipsum_mask.npy MUST be treated as observed (mask != 0).
    assert lorem_mask.shape == (256, 256)
    assert lorem_mask.dtype == np.dtype(np.uint8)
    assert int(np.sum(lorem_mask == 0)) == 6273
    assert int(np.sum(lorem_mask != 0)) == 59263
    assert int(np.sum(lorem_mask == 1)) < int(np.sum(lorem_mask != 0))

    assert int(np.sum(box_mask == 0)) == 16384
    assert int(np.sum(box_mask != 0)) == 49152


def test_manifest_validator_detects_corruption(tmp_path: Path):
    summary = mask_mod.validate_manifest(check_imagefolder=False)
    assert summary["count"] == 57
    assert summary["unique_hashes"] == 57

    # Copy 2 synthetic PNGs into tmp_path and build valid manifest, then corrupt it in 4 ways.
    ds_dir = tmp_path / "ds"
    paths = _make_synthetic_dataset(ds_dir, count=2)
    entries = []
    for i, p in enumerate(paths):
        entries.append(
            {
                "index": i,
                "input": f"cat/{p.name}",
                "filename": p.name,
                "width": 256,
                "height": 256,
                "mode": "RGB",
                "format": "PNG",
                "bytes": p.stat().st_size,
                "sha256": mask_mod.sha256_file(p),
                "source": "synthetic",
            }
        )
    manifest_payload = {
        "schema_version": "1.0",
        "subset_name": "synthetic_2",
        "expected_count": 2,
        "sort_order": "lexicographical_relative_posix_path",
        "image_spec": {"width": 256, "height": 256, "mode": "RGB", "format": "PNG"},
        "inputs": entries,
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest_payload), encoding="utf-8")
    assert mask_mod.validate_manifest(mpath, ds_dir, check_imagefolder=False)["count"] == 2

    # 1) Tampered image bytes -> SHA/bytes mismatch
    tamper_ds = tmp_path / "ds_tamper"
    shutil.copytree(ds_dir, tamper_ds)
    _write_synthetic_png(tamper_ds / "cat" / "0.png", np.full((256, 256, 3), 200, dtype=np.uint8))
    with pytest.raises(ValueError):
        mask_mod.validate_manifest(mpath, tamper_ds, check_imagefolder=False)

    # 2) Duplicate SHA-256
    dup_ds = tmp_path / "ds_dup"
    shutil.copytree(ds_dir, dup_ds)
    shutil.copy2(dup_ds / "cat" / "0.png", dup_ds / "cat" / "1.png")
    dup_manifest = json.loads(mpath.read_text(encoding="utf-8"))
    dup_manifest["inputs"][1]["sha256"] = dup_manifest["inputs"][0]["sha256"]
    dup_manifest["inputs"][1]["bytes"] = dup_manifest["inputs"][0]["bytes"]
    dup_mpath = tmp_path / "dup_manifest.json"
    dup_mpath.write_text(json.dumps(dup_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate SHA-256"):
        mask_mod.validate_manifest(dup_mpath, dup_ds, check_imagefolder=False)

    # 3) Wrong count
    bad_count = json.loads(mpath.read_text(encoding="utf-8"))
    bad_count["expected_count"] = 3
    bad_count_path = tmp_path / "bad_count.json"
    bad_count_path.write_text(json.dumps(bad_count), encoding="utf-8")
    with pytest.raises(ValueError, match="count mismatch"):
        mask_mod.validate_manifest(bad_count_path, ds_dir, check_imagefolder=False)

    # 4) Missing file on disk
    miss_ds = tmp_path / "ds_miss"
    shutil.copytree(ds_dir, miss_ds)
    (miss_ds / "cat" / "1.png").unlink()
    with pytest.raises(ValueError, match="Expected 2 PNGs"):
        mask_mod.validate_manifest(mpath, miss_ds, check_imagefolder=False)


def test_smoke_8_indices_helper():
    smoke = mask_mod.get_smoke_subset()
    assert smoke["expected_count"] == 8
    assert smoke["sample_ids_in_57"] == [3, 7, 8, 11, 21, 22, 54, 56]
    assert [e["index"] for e in smoke["inputs"]] == list(range(8))
    assert [e["index_in_57"] for e in smoke["inputs"]] == [3, 7, 8, 11, 21, 22, 54, 56]
    assert tuple(e["filename"] for e in smoke["inputs"]) == mask_mod.SMOKE_8_FILENAMES


# ---------------------------------------------------------------------------
# 2. Resumable Runner Tests (Mocked Diffusion)
# ---------------------------------------------------------------------------


def _build_fake_diffusion_kwargs(call_log: list, zero_mse: bool = False, omit_metrics: bool = False):
    class FakeDiffusionKwargs:
        def __init__(self, args, config, device=None):
            self.args = args
            self.config = config
            self.device = device
            self.last_run_records = []

        def sample(self, logger, *, checkpoint=None, dataset=None, sample_ids=None, on_result=None):
            # Simulate legacy diffusion.py in-place sigma_y mutation; manifest must stay unaffected!
            self.args.sigma_y = 2.0 * self.args.sigma_y
            call_log.append(list(sample_ids))
            out_dir = Path(self.args.image_folder)
            out_dir.mkdir(parents=True, exist_ok=True)
            for ds_i, sid in enumerate(sample_ids):
                x_orig_01, _ = dataset[ds_i]
                x_orig_01 = x_orig_01.unsqueeze(0)
                if zero_mse:
                    x_hat_01 = x_orig_01.clone()
                else:
                    x_hat_01 = torch.clamp(x_orig_01 + 0.05 * ((sid + 1) / 10.0), 0.0, 1.0)
                img_u8 = (x_hat_01.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                restored_path = out_dir / f"{sid}_0.png"
                Image.fromarray(img_u8, mode="RGB").save(restored_path, format="PNG")

                x_orig_m11 = x_orig_01 * 2.0 - 1.0
                x_hat_m11 = x_hat_01 * 2.0 - 1.0
                rec = {
                    "index": sid,
                    "x_orig_01": x_orig_01,
                    "x_hat_01": x_hat_01,
                    "x_orig_m11": x_orig_m11,
                    "x_hat_m11": x_hat_m11,
                    "y_obs": None,
                    "Apy_01": x_orig_01.clone(),
                    "restored_path": str(restored_path),
                }
                if not omit_metrics:
                    mse = float(torch.mean((x_hat_01 - x_orig_01) ** 2).item())
                    rec.update(
                        {
                            "full_psnr_db": (-10.0 * math.log10(mse)) if mse > 0 else None,
                            "full_vgg_lpips": None if self.args.skip_lpips else 0.1 + 0.01 * sid,
                            "hole_psnr_db": (-10.0 * math.log10(mse)) if mse > 0 else None,
                            "hole_vgg_lpips": None if self.args.skip_lpips else 0.08 + 0.01 * sid,
                            "observed_max_abs_diff": 0.0 if zero_mse else 0.01 * (sid + 1),
                        }
                    )
                self.last_run_records.append(rec)

    return FakeDiffusionKwargs


def _build_fake_diffusion_settings_only(dataset_dir: Path, call_log: list):
    class FakeDiffusionSettingsOnly:
        def __init__(self, args, config, device=None):
            self.args = args
            self.config = config
            self.device = device
            self.last_run_records = []

        def sample(self, logger):
            sample_ids = list(self.args.sample_ids)
            call_log.append(sample_ids)
            all_files = sorted((dataset_dir / "cat").glob("*.png"), key=lambda p: p.name)
            out_dir = Path(self.args.image_folder)
            out_dir.mkdir(parents=True, exist_ok=True)
            for ds_idx in sample_ids:
                with Image.open(all_files[ds_idx]) as img:
                    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                x_orig_01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                x_hat_01 = torch.clamp(x_orig_01 + 0.04, 0.0, 1.0)
                img_u8 = (x_hat_01.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                restored_path = out_dir / f"{ds_idx}_0.png"
                Image.fromarray(img_u8, mode="RGB").save(restored_path, format="PNG")
                mse = float(torch.mean((x_hat_01 - x_orig_01) ** 2).item())
                self.last_run_records.append(
                    {
                        "index": ds_idx,
                        "x_orig_01": x_orig_01,
                        "x_hat_01": x_hat_01,
                        "x_orig_m11": x_orig_01 * 2.0 - 1.0,
                        "x_hat_m11": x_hat_01 * 2.0 - 1.0,
                        "Apy_01": x_orig_01,
                        "full_psnr_db": -10.0 * math.log10(mse),
                        "full_vgg_lpips": None,
                        "hole_psnr_db": -10.0 * math.log10(mse),
                        "hole_vgg_lpips": None,
                        "observed_max_abs_diff": 0.04,
                        "restored_path": str(restored_path),
                    }
                )

    return FakeDiffusionSettingsOnly


def test_runner_fresh_run_schema_and_no_nan(tmp_path: Path, monkeypatch):
    dataset_dir, exp_dir, cfg, ckpt, _ = _make_test_env(tmp_path, count=4)
    out_dir = tmp_path / "run_out"
    calls = []
    monkeypatch.setattr(runner_mod, "Diffusion", _build_fake_diffusion_kwargs(calls))

    rc = runner_mod.main(
        [
            "--dataset", str(dataset_dir),
            "--expected-count", "4",
            "--config", str(cfg),
            "--checkpoint", str(ckpt),
            "--mask", "center_box_mask.npy",
            "--output", str(out_dir),
            "--exp", str(exp_dir),
            "--device", "cpu",
            "--sigma-y", "0.05",
            "--seed", "1234",
        ]
    )
    assert rc == 0
    assert calls == [[0, 1, 2, 3]]
    manifest = _loads_strict(out_dir / "manifest.json")
    expected_keys = {
        "git_commit",
        "config_sha256",
        "checkpoint_sha256",
        "mask_path",
        "mask_sha256",
        "context_mode",
        "sigma_y",
        "iN",
        "gamma",
        "zeta",
        "eta",
        "deltas",
        "deltas_injection_type",
        "seed",
        "seed_rule",
        "context_params",
        "runtime",
        "inputs",
    }
    assert set(manifest.keys()) == expected_keys
    assert manifest["sigma_y"] == 0.05  # Protected against in-place args.sigma_y mutation
    assert manifest["runtime"]["gpu"] == "cpu"
    assert len(manifest["inputs"]) == 4

    for idx in range(4):
        rec = _loads_strict(out_dir / "records" / f"{idx}.json")
        assert set(rec.keys()) == set(runner_mod.RECORD_FIELDS)
        assert rec["index"] == idx
        assert (out_dir / "images" / f"{idx}_0.png").is_file()
        assert rec["full_psnr_db"] is not None
        assert rec["full_vgg_lpips"] is not None
        assert rec["hole_psnr_db"] is not None
        assert rec["hole_vgg_lpips"] is not None

    with (out_dir / "results.csv").open("r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 4
    summary = _loads_strict(out_dir / "summary.json")
    assert summary["completed"] == 4
    assert len(summary["samples"]) == 4


def test_runner_settings_sample_ids_and_subset_remapping(tmp_path: Path, monkeypatch):
    dataset_dir, exp_dir, cfg, ckpt, _ = _make_test_env(tmp_path, count=4)
    subset_manifest = tmp_path / "subset_manifest.json"
    all_pngs = sorted((dataset_dir / "cat").glob("*.png"), key=lambda p: p.name)
    picked = [all_pngs[1], all_pngs[3]]
    subset_manifest.write_text(
        json.dumps(
            {
                "expected_count": 2,
                "inputs": [
                    {"index": 0, "input": f"cat/{picked[0].name}", "sha256": runner_mod.digest(picked[0])},
                    {"index": 1, "input": f"cat/{picked[1].name}", "sha256": runner_mod.digest(picked[1])},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "subset_out"
    calls = []
    monkeypatch.setattr(runner_mod, "Diffusion", _build_fake_diffusion_settings_only(dataset_dir, calls))

    rc = runner_mod.main(
        [
            "--dataset", str(dataset_dir),
            "--inputs-manifest", str(subset_manifest),
            "--config", str(cfg),
            "--checkpoint", str(ckpt),
            "--mask", "center_box_mask.npy",
            "--output", str(out_dir),
            "--exp", str(exp_dir),
            "--device", "cpu",
            "--skip-lpips",
        ]
    )
    assert rc == 0
    assert calls == [[1, 3]]
    assert (out_dir / "images" / "0_0.png").is_file()
    assert (out_dir / "images" / "1_0.png").is_file()
    assert not (out_dir / "images" / "3_0.png").exists()
    summary = _loads_strict(out_dir / "summary.json")
    assert summary["completed"] == 2
    assert [s["input"] for s in summary["samples"]] == ["cat/1.png", "cat/3.png"]


def test_runner_resume_skips_valid_and_reruns_corrupt(tmp_path: Path, monkeypatch):
    dataset_dir, exp_dir, cfg, ckpt, _ = _make_test_env(tmp_path, count=4)
    out_dir = tmp_path / "resume_out"
    calls = []
    monkeypatch.setattr(runner_mod, "Diffusion", _build_fake_diffusion_kwargs(calls))

    common_args = [
        "--dataset", str(dataset_dir),
        "--expected-count", "4",
        "--config", str(cfg),
        "--checkpoint", str(ckpt),
        "--mask", "center_box_mask.npy",
        "--output", str(out_dir),
        "--exp", str(exp_dir),
        "--device", "cpu",
    ]
    assert runner_mod.main(common_args) == 0
    assert calls == [[0, 1, 2, 3]]

    # Tamper image 1 (SHA mismatch) and corrupt record 2 JSON
    _write_synthetic_png(out_dir / "images" / "1_0.png", np.zeros((256, 256, 3), dtype=np.uint8))
    (out_dir / "records" / "2.json").write_text("{corrupt", encoding="utf-8")
    rec0_before = (out_dir / "records" / "0.json").read_text(encoding="utf-8")
    rec3_before = (out_dir / "records" / "3.json").read_text(encoding="utf-8")

    assert runner_mod.main(common_args) == 0
    assert calls == [[0, 1, 2, 3], [1, 2]]
    assert (out_dir / "records" / "0.json").read_text(encoding="utf-8") == rec0_before
    assert (out_dir / "records" / "3.json").read_text(encoding="utf-8") == rec3_before


def test_runner_rejects_manifest_mismatch(tmp_path: Path, monkeypatch):
    dataset_dir, exp_dir, cfg, ckpt, _ = _make_test_env(tmp_path, count=4)
    out_dir = tmp_path / "mismatch_out"
    calls = []
    monkeypatch.setattr(runner_mod, "Diffusion", _build_fake_diffusion_kwargs(calls))

    base_args = [
        "--dataset", str(dataset_dir),
        "--expected-count", "4",
        "--config", str(cfg),
        "--checkpoint", str(ckpt),
        "--mask", "center_box_mask.npy",
        "--output", str(out_dir),
        "--exp", str(exp_dir),
        "--device", "cpu",
        "--seed", "100",
        "--context-mode", "none",
    ]
    assert runner_mod.main(base_args) == 0
    manifest_before = (out_dir / "manifest.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Resume manifest differs"):
        runner_mod.main(base_args[:-1] + ["telea"])

    with pytest.raises(ValueError, match="Resume manifest differs"):
        runner_mod.main(base_args[:-4] + ["--seed", "101", "--context-mode", "none"])

    assert (out_dir / "manifest.json").read_text(encoding="utf-8") == manifest_before


def test_runner_filelock_prevents_concurrent_run(tmp_path: Path, monkeypatch):
    dataset_dir, exp_dir, cfg, ckpt, _ = _make_test_env(tmp_path, count=4)
    out_dir = tmp_path / "locked_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runner_mod, "Diffusion", _build_fake_diffusion_kwargs([]))

    with FileLock(out_dir / ".run.lock", timeout=0):
        with pytest.raises(Timeout):
            runner_mod.main(
                [
                    "--dataset", str(dataset_dir),
                    "--expected-count", "4",
                    "--config", str(cfg),
                    "--checkpoint", str(ckpt),
                    "--mask", "center_box_mask.npy",
                    "--output", str(out_dir),
                    "--exp", str(exp_dir),
                    "--device", "cpu",
                ]
            )


def test_runner_zero_mse_psnr_is_json_null(tmp_path: Path, monkeypatch):
    dataset_dir, exp_dir, cfg, ckpt, _ = _make_test_env(tmp_path, count=2)
    out_dir = tmp_path / "zero_mse_out"
    calls = []
    monkeypatch.setattr(runner_mod, "Diffusion", _build_fake_diffusion_kwargs(calls, zero_mse=True, omit_metrics=True))

    rc = runner_mod.main(
        [
            "--dataset", str(dataset_dir),
            "--expected-count", "2",
            "--config", str(cfg),
            "--checkpoint", str(ckpt),
            "--mask", "center_box_mask.npy",
            "--output", str(out_dir),
            "--exp", str(exp_dir),
            "--device", "cpu",
            "--skip-lpips",
        ]
    )
    assert rc == 0
    rec0 = _loads_strict(out_dir / "records" / "0.json")
    assert rec0["full_psnr_db"] is None
    assert rec0["hole_psnr_db"] is None
    assert rec0["observed_max_abs_diff"] == 0.0
    summary = _loads_strict(out_dir / "summary.json")
    assert summary["mean_full_psnr_db"] is None
    assert summary["mean_hole_psnr_db"] is None
    with (out_dir / "results.csv").open("r", encoding="utf-8", newline="") as f:
        row0 = next(csv.DictReader(f))
    assert row0["full_psnr_db"] == ""
    assert row0["hole_psnr_db"] == ""


def test_runner_tensor_fallback_metrics_respect_mask_nonzero():
    # Anti-aliased mask: 0=hole, 128=observed (NOT ==1!), 255=observed
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[0:4, 0:4] = 0
    mask[4:6, 4:6] = 128

    x_orig = torch.full((1, 3, 8, 8), 0.2, dtype=torch.float32)
    x_hat = x_orig.clone()
    # Perturb hole (0:4, 0:4) by +0.5 and observed non-255 pixel (4:6, 4:6) by +0.1
    x_hat[:, :, 0:4, 0:4] = 0.7
    x_hat[:, :, 4:6, 4:6] = 0.3

    rec = {
        "index": 0,
        "x_orig_01": x_orig,
        "x_hat_01": x_hat,
        "Apy_01": x_orig,
    }
    metrics = runner_mod.compute_metrics_from_tensors(rec, mask_2d=mask, skip_lpips=True)
    # Hole MSE = 0.5^2 = 0.25 -> PSNR = -10*log10(0.25) = 6.0205999...
    assert metrics["hole_psnr_db"] == pytest.approx(-10.0 * math.log10(0.25), rel=1e-5)
    # Observed max abs diff includes the mask==128 observed region -> 0.1
    assert metrics["observed_max_abs_diff"] == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Analysis & Visualization Tests
# ---------------------------------------------------------------------------


def test_bootstrap_ci_determinism_and_edge_cases():
    empty = analysis_mod.bootstrap_mean_ci([None, float("nan"), float("inf")])
    assert empty == {"n": 0, "mean": None, "std": None, "median": None, "ci_low": None, "ci_high": None}

    single = analysis_mod.bootstrap_mean_ci([2.5, None])
    assert single == {"n": 1, "mean": 2.5, "std": 0.0, "median": 2.5, "ci_low": 2.5, "ci_high": 2.5}

    const = analysis_mod.bootstrap_mean_ci([1.25, 1.25, 1.25, 1.25])
    assert const == {"n": 4, "mean": 1.25, "std": 0.0, "median": 1.25, "ci_low": 1.25, "ci_high": 1.25}

    vals = [0.1, 0.4, -0.2, 0.8, 0.5, 0.3]
    r1 = analysis_mod.bootstrap_mean_ci(vals, n_boot=10000, seed=42)
    r2 = analysis_mod.bootstrap_mean_ci(vals, n_boot=10000, seed=42)
    assert r1 == r2
    assert r1["n"] == 6
    assert r1["ci_low"] <= r1["mean"] <= r1["ci_high"]


def _build_synthetic_analysis_runs(
    tmp_path: Path,
    conditions: list[str],
    masks: list[str],
    seeds: list[int],
    n_images: int = 6,
):
    dataset_root = tmp_path / "dataset"
    cat_dir = dataset_root / "cat"
    cat_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 flat images (0,1,2) and 3 high-edge checkerboard images (3,4,5)
    gt_arrays: list[np.ndarray] = []
    inputs_list = []
    for i in range(n_images):
        if i < n_images // 2:
            arr = np.full((64, 64, 3), 100 + i * 5, dtype=np.uint8)
        else:
            yy, xx = np.indices((64, 64))
            checker = (((yy // 2) + (xx // 2)) % 2) * (160 + i * 10)
            arr = np.stack([checker, checker, checker], axis=-1).astype(np.uint8)
        p = cat_dir / f"img_{i}.png"
        _write_synthetic_png(p, arr)
        gt_arrays.append(arr)
        inputs_list.append({"input": f"cat/img_{i}.png", "sha256": f"sha_{i}"})

    # Masks
    for mname in masks:
        if "random" in mname:
            rng = np.random.default_rng(7)
            m = (rng.random((64, 64)) > 0.8).astype(np.float32)
        else:
            m = np.ones((64, 64), dtype=np.float32)
            m[16:48, 16:48] = 0.0
        np.save(masks_dir / f"{mname}.npy", m)

    # Runs
    results_root = tmp_path / "runs"
    results_root.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []
    for mname in masks:
        for seed in seeds:
            for cond in conditions:
                rdir = results_root / f"{cond}_{mname}_s{seed}"
                (rdir / "records").mkdir(parents=True, exist_ok=True)
                (rdir / "images").mkdir(parents=True, exist_ok=True)
                context_mode = "clip" if cond == "clip_neg_ctrl" else cond
                manifest = {
                    "mask_path": f"exp/inpainting_masks/{mname}.npy",
                    "context_mode": context_mode,
                    "sigma_y": 0.05,
                    "seed": seed,
                    "context_params": {"clip_negative_control": True} if cond == "clip_neg_ctrl" else {},
                    "inputs": inputs_list,
                }
                (rdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

                for idx in range(n_images):
                    # Construct restored image & deterministic metrics
                    offset = {"none": 20, "telea": 12, "spatial": 8, "clip": 5, "clip_neg_ctrl": 25}[cond]
                    restored = np.clip(gt_arrays[idx].astype(np.int32) + offset, 0, 255).astype(np.uint8)
                    _write_synthetic_png(rdir / "images" / f"{idx}_0.png", restored)

                    # Deterministic synthetic deltas vs none:
                    # telea: improves indices 0..4, slightly hurts index 5
                    # clip: improves all, clip_neg_ctrl: worsens all
                    seed_jitter = 0.05 * (seed - seeds[0])
                    if cond == "none":
                        fpsnr = 24.0 + 0.1 * idx + seed_jitter
                        flpips = 0.25 - 0.005 * idx
                    elif cond == "telea":
                        delta_p = 1.8 - 0.4 * idx if idx < 5 else -0.5
                        fpsnr = (24.0 + 0.1 * idx + seed_jitter) + delta_p
                        flpips = (0.25 - 0.005 * idx) + (-0.03 if idx < 5 else 0.02)
                    elif cond == "spatial":
                        fpsnr = (24.0 + 0.1 * idx + seed_jitter) + 1.2
                        flpips = (0.25 - 0.005 * idx) - 0.04
                    elif cond == "clip":
                        fpsnr = (24.0 + 0.1 * idx + seed_jitter) + 2.0
                        flpips = (0.25 - 0.005 * idx) - 0.06
                    else:  # clip_neg_ctrl
                        fpsnr = (24.0 + 0.1 * idx + seed_jitter) - 1.0
                        flpips = (0.25 - 0.005 * idx) + 0.05

                    rec = {
                        "index": idx,
                        "input": inputs_list[idx]["input"],
                        "sha256": inputs_list[idx]["sha256"],
                        "restored": f"images/{idx}_0.png",
                        "restored_sha256": "dummy",
                        "full_psnr_db": float(fpsnr),
                        "full_vgg_lpips": float(flpips),
                        "hole_psnr_db": float(fpsnr - 2.0),
                        "hole_vgg_lpips": float(flpips + 0.02),
                        "observed_max_abs_diff": 0.0,
                    }
                    (rdir / "records" / f"{idx}.json").write_text(json.dumps(rec), encoding="utf-8")
                run_dirs.append(rdir)
    return dataset_root, masks_dir, results_root, run_dirs


def test_paired_delta_sign_convention_and_multiseed_aggregation(tmp_path: Path):
    dataset_root, masks_dir, results_root, _ = _build_synthetic_analysis_runs(
        tmp_path,
        conditions=["none", "telea", "spatial", "clip", "clip_neg_ctrl"],
        masks=["random_80_mask"],
        seeds=[100, 101],
        n_images=6,
    )
    runs = analysis_mod.discover_runs(results_root=results_root)
    labels = {r["condition_label"] for r in runs}
    assert labels == {"none", "telea", "spatial", "clip", "clip_neg_ctrl"}

    out = analysis_mod.analyze_runs(
        runs=runs,
        baseline_mode="none",
        dataset_root=dataset_root,
        masks_dir=masks_dir,
        n_boot=2000,
        bootstrap_seed=42,
    )

    # Image-level aggregation: 6 images (not 12 image-seed rows!)
    telea_summary = out["stratified_summary"]["telea|sigma_y=0.05"]["overall"]["all"]
    assert telea_summary["full_psnr_db"]["n"] == 6
    assert telea_summary["full_psnr_db"]["improved_count"] == 5
    assert telea_summary["full_psnr_db"]["worsened_count"] == 1
    assert telea_summary["full_vgg_lpips"]["improved_count"] == 5  # negative delta = improved
    assert telea_summary["full_vgg_lpips"]["worsened_count"] == 1

    neg_summary = out["stratified_summary"]["clip_neg_ctrl|sigma_y=0.05"]["overall"]["all"]
    assert neg_summary["full_psnr_db"]["mean"] == pytest.approx(-1.0, abs=1e-6)
    assert neg_summary["full_vgg_lpips"]["mean"] == pytest.approx(0.05, abs=1e-6)
    assert neg_summary["full_vgg_lpips"]["worsened_count"] == 6

    assert "telea|random_80_mask|0.05" in out["seed_breakdown"]
    sb = out["seed_breakdown"]["telea|random_80_mask|0.05"]
    assert sb["matched_seeds"] == [100, 101]
    assert sb["pooled_over_image_seeds"]["full_psnr_db"]["n"] == 12


def test_stratified_breakdown_masks_and_texture(tmp_path: Path):
    dataset_root, masks_dir, results_root, _ = _build_synthetic_analysis_runs(
        tmp_path,
        conditions=["none", "telea"],
        masks=["random_80_mask", "center_box_mask"],
        seeds=[100],
        n_images=6,
    )
    runs = analysis_mod.discover_runs(results_root=results_root)
    out = analysis_mod.analyze_runs(
        runs=runs,
        baseline_mode="none",
        dataset_root=dataset_root,
        masks_dir=masks_dir,
        n_boot=1000,
        bootstrap_seed=42,
        top_k=3,
    )

    assert out["mask_groups"]["random_80_mask"] == "scattered"
    assert out["mask_groups"]["center_box_mask"] == "contiguous"

    for idx in range(3):
        assert out["texture_scores"][f"cat/img_{idx}.png"]["texture_stratum"] == "flat"
    for idx in range(3, 6):
        assert out["texture_scores"][f"cat/img_{idx}.png"]["texture_stratum"] == "high_edge"
        assert out["texture_scores"][f"cat/img_{idx}.png"]["laplacian_var"] > 0.0

    strat = out["stratified_summary"]["telea|sigma_y=0.05"]
    assert strat["mask_group"]["scattered"]["full_psnr_db"]["n"] == 6
    assert strat["mask_group"]["contiguous"]["full_psnr_db"]["n"] == 6
    assert strat["texture_stratum"]["flat"]["full_psnr_db"]["n"] == 6
    assert strat["texture_stratum"]["high_edge"]["full_psnr_db"]["n"] == 6
    assert strat["mask_group_x_texture"]["contiguous|high_edge"]["full_psnr_db"]["n"] == 3

    top_entry = out["top_helped_hurt"]["telea|center_box_mask|sigma_y=0.05"]
    assert [e["index"] for e in top_entry["by_full_psnr_db"]["top_helped"]] == [0, 1, 2]
    assert [e["index"] for e in top_entry["by_full_psnr_db"]["top_hurt"]] == [5, 4, 3]


def test_visual_grids_and_error_heatmaps_subset_conditions(tmp_path: Path):
    # First test subset of conditions (none + telea)
    ds1, m1, rroot1, _ = _build_synthetic_analysis_runs(
        tmp_path / "sub",
        conditions=["none", "telea"],
        masks=["center_box_mask"],
        seeds=[100],
        n_images=4,
    )
    out1 = tmp_path / "analysis_sub"
    rc1 = analysis_mod.main(
        [
            "--results-root", str(rroot1),
            "--dataset-root", str(ds1),
            "--masks-dir", str(m1),
            "--output-dir", str(out1),
            "--n-boot", "500",
        ]
    )
    assert rc1 == 0
    _loads_strict(out1 / "analysis_summary.json")
    assert (out1 / "paired_deltas.csv").is_file()
    assert (out1 / "stratified_summary.csv").is_file()
    for fig_name in (
        "comparison_grid_center_box_mask.png",
        "error_heatmap_center_box_mask.png",
        "delta_ci_forest.png",
    ):
        fpath = out1 / "figures" / fig_name
        assert fpath.is_file()
        assert fpath.stat().st_size > 1000
        with Image.open(fpath) as img:
            img.verify()

    # Second test full 5-condition set across two masks
    ds2, m2, rroot2, _ = _build_synthetic_analysis_runs(
        tmp_path / "full",
        conditions=["none", "telea", "spatial", "clip", "clip_neg_ctrl"],
        masks=["random_80_mask", "center_box_mask"],
        seeds=[100],
        n_images=4,
    )
    out2 = tmp_path / "analysis_full"
    rc2 = analysis_mod.main(
        [
            "--results-root", str(rroot2),
            "--dataset-root", str(ds2),
            "--masks-dir", str(m2),
            "--output-dir", str(out2),
            "--n-boot", "500",
        ]
    )
    assert rc2 == 0
    summary2 = _loads_strict(out2 / "analysis_summary.json")
    assert len(summary2["figures"]) == 5
    for rel_fig in summary2["figures"]:
        fpath = out2 / rel_fig
        assert fpath.is_file()
        with Image.open(fpath) as img:
            assert img.size[0] > 100 and img.size[1] > 100
