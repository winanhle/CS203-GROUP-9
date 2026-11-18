#!/usr/bin/env python3
"""Full ablation suite runner for the CM4IR repository."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_inpainting_results as analysis_mod  # noqa: E402
from scripts import run_inpainting_ablation as runner_mod  # noqa: E402

MASKS = [
    ("random_80", "random_80_mask.npy"),
    ("lorem_ipsum", "lorem_ipsum_mask.npy"),
    ("center_box", "center_box_mask.npy"),
]

CONDITION_FLAGS = {
    "none": ["--context-mode", "none"],
    "telea": ["--context-mode", "telea"],
    "spatial": ["--context-mode", "spatial"],
    "clip": ["--context-mode", "clip"],
    "clip_neg_ctrl": ["--context-mode", "clip", "--clip-negative-control"],
}
CONDITIONS = list(CONDITION_FLAGS.items())


def run_one(
    manifest_rel: str,
    mask_file: str,
    cond_flags: list[str],
    sigma_y: float,
    seed: int,
    out_rel: str,
) -> dict[str, Any]:
    """Execute a single ablation configuration and report timing/metrics."""
    argv = [
        "--inputs-manifest", manifest_rel,
        "--mask", mask_file,
        "--sigma-y", str(sigma_y),
        "--seed", str(seed),
        "--output", out_rel,
        "--verify-official-checkpoint",
    ] + cond_flags

    start = time.perf_counter()
    runner_mod.main(argv)
    elapsed_s = time.perf_counter() - start

    summary_path = Path(out_rel) / "summary.json"
    metrics: dict[str, Any] = {}
    if summary_path.is_file():
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        metrics = {"error": f"summary.json not found at {summary_path}"}

    print(
        f"[run_one] {out_rel} | elapsed={elapsed_s:.2f}s | "
        f"full_psnr={metrics.get('mean_full_psnr_db')} | "
        f"full_lpips={metrics.get('mean_full_vgg_lpips')} | "
        f"hole_psnr={metrics.get('mean_hole_psnr_db')} | "
        f"hole_lpips={metrics.get('mean_hole_vgg_lpips')}",
        flush=True,
    )
    return metrics


def _mean_psnr(results_root: str, cond: str, mask_short: str, seed: int) -> float:
    summary_path = Path(results_root) / f"{cond}_{mask_short}_s{seed}" / "summary.json"
    if not summary_path.is_file():
        return float("-inf")
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(data.get("mean_full_psnr_db", float("-inf")))
    except (json.JSONDecodeError, ValueError, TypeError):
        return float("-inf")


def phase_smoke() -> None:
    manifest = "configs/lsun_cat_8_smoke_manifest.json"
    sigma_y = 0.05
    seed = 1234
    results_root = "results/smoke_8"

    for mask_short, mask_file in MASKS:
        for cond_name, cond_flags in CONDITIONS:
            out_rel = str(Path(results_root) / f"{cond_name}_{mask_short}_s{seed}")
            run_one(manifest, mask_file, cond_flags, sigma_y, seed, out_rel)

    analysis_mod.main([
        "--results-root", results_root,
        "--output-dir", f"{results_root}_analysis",
        "--n-boot", "2000",
    ])


def phase_ablation57() -> None:
    manifest = "configs/lsun_cat_57_manifest.json"
    sigma_y = 0.05
    seed = 1234
    results_root = "results/ablation_57"

    for mask_short, mask_file in MASKS:
        for cond_name, cond_flags in CONDITIONS:
            out_rel = str(Path(results_root) / f"{cond_name}_{mask_short}_s{seed}")
            run_one(manifest, mask_file, cond_flags, sigma_y, seed, out_rel)


def phase_multiseed() -> None:
    manifest = "configs/lsun_cat_57_manifest.json"
    sigma_y = 0.05
    results_root = "results/ablation_57"
    candidates = ["telea", "spatial", "clip"]
    extra_seeds = [1235, 1236]

    for mask_short, mask_file in MASKS:
        best_cond = max(candidates, key=lambda c: _mean_psnr(results_root, c, mask_short, 1234))
        conds_to_run = ["none", best_cond]
        if "telea" not in conds_to_run:
            conds_to_run.append("telea")
        for seed in extra_seeds:
            for cond_name in conds_to_run:
                out_rel = str(Path(results_root) / f"{cond_name}_{mask_short}_s{seed}")
                run_one(manifest, mask_file, CONDITION_FLAGS[cond_name], sigma_y, seed, out_rel)


def phase_sigma025() -> None:
    manifest = "configs/lsun_cat_8_smoke_manifest.json"
    sigma_y = 0.025
    seed = 1234
    results_root = "results/smoke_8_sigma025"

    for mask_short, mask_file in MASKS:
        for cond_name, cond_flags in CONDITIONS:
            out_rel = str(Path(results_root) / f"{cond_name}_{mask_short}_s{seed}")
            run_one(manifest, mask_file, cond_flags, sigma_y, seed, out_rel)

    analysis_mod.main([
        "--results-root", results_root,
        "--output-dir", f"{results_root}_analysis",
        "--n-boot", "2000",
    ])


def phase_table5_57() -> None:
    manifest = "configs/lsun_cat_57_manifest.json"
    sigma_y = 0.05
    seed = 1234
    results_root = "results/ablation_57_table5"
    table5_flags = ["--deltas", "0,0,1.0,1.0", "--deltas-injection-type", "1"]

    for mask_short, mask_file in MASKS:
        for cond_name in ("none", "telea", "spatial"):
            out_rel = str(Path(results_root) / f"{cond_name}_{mask_short}_s{seed}")
            run_one(manifest, mask_file, CONDITION_FLAGS[cond_name] + table5_flags, sigma_y, seed, out_rel)

    analysis_mod.main([
        "--results-root", results_root,
        "--output-dir", f"{results_root}_analysis",
        "--n-boot", "10000",
        "--bootstrap-seed", "42",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("smoke", "ablation57", "multiseed", "sigma025", "analyze57", "table5_57", "all"),
        default="all",
    )
    args = parser.parse_args()

    if args.phase in ("smoke", "all"):
        phase_smoke()
    if args.phase in ("ablation57", "all"):
        phase_ablation57()
    if args.phase in ("multiseed", "all"):
        phase_multiseed()
    if args.phase in ("sigma025", "all"):
        phase_sigma025()
    if args.phase == "table5_57":
        phase_table5_57()
    if args.phase in ("ablation57", "multiseed", "analyze57", "all"):
        analysis_mod.main([
            "--results-root", "results/ablation_57",
            "--output-dir", "results/ablation_57_analysis",
            "--n-boot", "10000",
            "--bootstrap-seed", "42",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())