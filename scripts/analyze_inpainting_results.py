"""Analyze paired CM4IR inpainting ablation results, bootstrap CIs, stratification, and figures."""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_CONDITION_ORDER = [
    "none",
    "telea",
    "spatial",
    "poisson",
    "poisson_bridge",
    "poisson_bridge_sharp",
    "clip",
    "clip_neg_ctrl",
]
CONDITION_DISPLAY_NAMES = {
    "none": "CM4IR (none)",
    "telea": "Telea init",
    "spatial": "Spatial feather",
    "poisson": "Poisson BP",
    "poisson_bridge": "Poisson + Bridge (Annealed)",
    "poisson_bridge_sharp": "Poisson + Bridge (Sharp)",
    "clip": "CLIP null-space",
    "clip_neg_ctrl": "CLIP neg ctrl",
}

METRIC_SPECS = {
    "full_psnr_db": {"higher_is_better": True, "label": "Full PSNR (dB)"},
    "full_vgg_lpips": {"higher_is_better": False, "label": "Full VGG-LPIPS"},
    "hole_psnr_db": {"higher_is_better": True, "label": "Hole PSNR (dB)"},
    "hole_vgg_lpips": {"higher_is_better": False, "label": "Hole VGG-LPIPS"},
}


def write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON with allow_nan=False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Atomically write CSV with empty strings for None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _to_finite_float_or_none(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return None
    fval = float(val)
    return fval if math.isfinite(fval) else None


def infer_condition_label(
    manifest: dict[str, Any],
    dir_path: Path,
    label_override: Optional[str] = None,
) -> str:
    """Infer canonical condition label, distinguishing clip vs clip_neg_ctrl."""
    if label_override is not None and label_override.strip():
        return label_override.strip()
    cond_name = manifest.get("condition_name")
    if isinstance(cond_name, str) and cond_name.strip():
        return cond_name.strip()
    context_mode = str(manifest.get("context_mode", "none")).strip() or "none"
    context_params = manifest.get("context_params") or {}
    if context_mode == "clip" and (
        bool(context_params.get("clip_negative_control"))
        or bool(context_params.get("negative_control"))
        or "neg" in dir_path.name.lower()
    ):
        return "clip_neg_ctrl"
    return context_mode


def _parse_csv_records(csv_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            records.append(
                {
                    "index": int(row["index"]),
                    "input": row["input"],
                    "sha256": row.get("sha256", ""),
                    "restored": row.get("restored", f"images/{int(row['index'])}_0.png"),
                    "restored_sha256": row.get("restored_sha256", ""),
                    "full_psnr_db": _to_finite_float_or_none(row.get("full_psnr_db")),
                    "full_vgg_lpips": _to_finite_float_or_none(row.get("full_vgg_lpips")),
                    "hole_psnr_db": _to_finite_float_or_none(row.get("hole_psnr_db")),
                    "hole_vgg_lpips": _to_finite_float_or_none(row.get("hole_vgg_lpips")),
                    "observed_max_abs_diff": _to_finite_float_or_none(row.get("observed_max_abs_diff")),
                }
            )
    records.sort(key=lambda r: int(r["index"]))
    return records


def load_condition_run(run_dir: Path, label_override: Optional[str] = None) -> dict[str, Any]:
    """Load one ablation output directory (manifest.json + records/*.json or results.csv)."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json missing in run directory: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    condition_label = infer_condition_label(manifest, run_dir, label_override=label_override)
    mask_path_str = str(manifest["mask_path"])
    mask_name = Path(mask_path_str).stem
    sigma_y = float(manifest["sigma_y"])
    seed = int(manifest["seed"])

    records_dir = run_dir / "records"
    csv_path = run_dir / "results.csv"
    records: list[dict[str, Any]] = []
    if records_dir.is_dir():
        json_files = sorted(records_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        if json_files:
            for jf in json_files:
                rec = json.loads(jf.read_text(encoding="utf-8"))
                records.append(rec)
            records.sort(key=lambda r: int(r["index"]))
    if not records and csv_path.is_file():
        records = _parse_csv_records(csv_path)
    if not records:
        raise ValueError(f"No records/*.json or results.csv found in {run_dir}")

    required_cols = (
        "index",
        "input",
        "full_psnr_db",
        "full_vgg_lpips",
        "hole_psnr_db",
        "hole_vgg_lpips",
        "observed_max_abs_diff",
    )
    by_index: dict[int, dict[str, Any]] = {}
    for rec in records:
        for col in required_cols:
            if col not in rec:
                raise ValueError(f"Record in {run_dir} missing required field '{col}'")
        idx = int(rec["index"])
        normalized = dict(rec)
        normalized["index"] = idx
        for metric_key in (*METRIC_SPECS.keys(), "observed_max_abs_diff"):
            normalized[metric_key] = _to_finite_float_or_none(rec.get(metric_key))
        rel_restored = rec.get("restored") or f"images/{idx}_0.png"
        normalized["restored_abs_path"] = run_dir / rel_restored
        by_index[idx] = normalized

    return {
        "run_dir": run_dir,
        "condition_label": condition_label,
        "mask_name": mask_name,
        "mask_path": mask_path_str,
        "sigma_y": sigma_y,
        "seed": seed,
        "manifest": manifest,
        "records": [by_index[k] for k in sorted(by_index)],
        "records_by_index": by_index,
    }


def discover_runs(
    run_specs: Optional[list[str]] = None,
    results_root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Discover and load runs from --runs (path or label=path) and/or --results-root."""
    loaded: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()

    if run_specs:
        for spec in run_specs:
            if "=" in spec:
                label_override, path_str = spec.split("=", 1)
            else:
                label_override, path_str = None, spec
            run_dir = Path(path_str).resolve()
            if run_dir not in seen_dirs:
                loaded.append(load_condition_run(run_dir, label_override=label_override))
                seen_dirs.add(run_dir)

    if results_root is not None:
        root = results_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Results root not found: {root}")
        candidates = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
        for cand in candidates:
            has_manifest = (cand / "manifest.json").is_file()
            has_data = (cand / "records").is_dir() or (cand / "results.csv").is_file()
            if has_manifest and has_data and cand.resolve() not in seen_dirs:
                loaded.append(load_condition_run(cand.resolve()))
                seen_dirs.add(cand.resolve())

    if not loaded:
        raise ValueError("No valid ablation run directories discovered")
    return loaded


def bootstrap_mean_ci(
    values: Any,
    n_boot: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Compute seeded percentile bootstrap CI for the mean with edge-case safety."""
    raw = [float(v) for v in values if v is not None and not isinstance(v, bool) and math.isfinite(float(v))]
    n = len(raw)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "ci_low": None,
            "ci_high": None,
        }
    arr = np.asarray(raw, dtype=np.float64)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    if n == 1:
        return {
            "n": 1,
            "mean": mean_val,
            "std": 0.0,
            "median": median_val,
            "ci_low": mean_val,
            "ci_high": mean_val,
        }
    std_val = float(np.std(arr, ddof=1))
    if np.all(arr == arr[0]):
        return {
            "n": n,
            "mean": mean_val,
            "std": 0.0,
            "median": median_val,
            "ci_low": mean_val,
            "ci_high": mean_val,
        }

    rng = np.random.default_rng(seed)
    if n_boot * n <= 5_000_000:
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = arr[idx].mean(axis=1)
    else:
        chunk_size = max(1, 5_000_000 // n)
        chunks = []
        remaining = n_boot
        while remaining > 0:
            cur = min(chunk_size, remaining)
            idx = rng.integers(0, n, size=(cur, n))
            chunks.append(arr[idx].mean(axis=1))
            remaining -= cur
        boot_means = np.concatenate(chunks, axis=0)

    alpha = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "n": int(n),
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def classify_mask_group(mask_name: str, mask_array: Optional[np.ndarray] = None) -> str:
    """Classify mask geometry as 'scattered' or 'contiguous'."""
    lower = mask_name.lower()
    if lower in {"random_80_mask", "lorem_ipsum_mask"} or lower.startswith(("random_", "lorem_")):
        return "scattered"
    if lower == "center_box_mask" or "box" in lower or "center" in lower:
        return "contiguous"
    if mask_array is not None:
        hole = (mask_array == 0).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(hole, connectivity=8)
        num_components = int(num_labels) - 1
        return "contiguous" if num_components == 1 else "scattered"
    return "unknown"


def compute_texture_complexity(gt_rgb_01: np.ndarray) -> dict[str, float]:
    """Compute BT.601 Sobel mean gradient magnitude and Laplacian variance on [0, 1] RGB image."""
    arr = np.asarray(gt_rgb_01, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array, got shape {arr.shape}")
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    sobel_mean_grad_mag = float(np.mean(grad_mag))
    laplacian_var = float(np.var(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)))
    return {
        "sobel_mean_grad_mag": sobel_mean_grad_mag,
        "laplacian_var": laplacian_var,
    }


def resolve_mask_file(mask_path_str: str, mask_name: str, masks_dir: Path) -> Optional[Path]:
    raw_path = Path(mask_path_str)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    candidates.extend(
        [
            masks_dir / f"{mask_name}.npy",
            ROOT / mask_path_str,
            ROOT / "exp" / "inpainting_masks" / f"{mask_name}.npy",
        ]
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def resolve_gt_image_path(rel_input: str, dataset_root: Path) -> Optional[Path]:
    rel = Path(rel_input)
    candidates = [
        dataset_root / rel,
        dataset_root / rel.name,
        dataset_root / "cat" / rel.name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def load_rgb_01(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _order_conditions(labels: set[str] | list[str]) -> list[str]:
    present = set(labels)
    ordered = [c for c in CANONICAL_CONDITION_ORDER if c in present]
    extras = sorted(c for c in present if c not in CANONICAL_CONDITION_ORDER)
    return ordered + extras


def _direction_counts(deltas: list[Optional[float]], higher_is_better: bool) -> dict[str, int]:
    improved = 0
    worsened = 0
    tied = 0
    for d in deltas:
        if d is None or not math.isfinite(d):
            continue
        if d == 0.0:
            tied += 1
        elif (d > 0.0 and higher_is_better) or (d < 0.0 and not higher_is_better):
            improved += 1
        else:
            worsened += 1
    return {
        "improved_count": improved,
        "worsened_count": worsened,
        "tied_count": tied,
    }


def _summarize_group(
    rows: list[dict[str, Any]],
    n_boot: int,
    seed: int,
    ci_level: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for metric_key, spec in METRIC_SPECS.items():
        delta_vals = [r[f"delta_{metric_key}"] for r in rows]
        ci = bootstrap_mean_ci(delta_vals, n_boot=n_boot, seed=seed, ci_level=ci_level)
        counts = _direction_counts(delta_vals, higher_is_better=spec["higher_is_better"])
        base_vals = [r[f"baseline_{metric_key}_mean"] for r in rows if r[f"baseline_{metric_key}_mean"] is not None]
        meth_vals = [r[f"method_{metric_key}_mean"] for r in rows if r[f"method_{metric_key}_mean"] is not None]
        out[metric_key] = {
            **ci,
            **counts,
            "baseline_mean": float(np.mean(base_vals)) if base_vals else None,
            "method_mean": float(np.mean(meth_vals)) if meth_vals else None,
        }
    return out


def analyze_runs(
    runs: list[dict[str, Any]],
    baseline_mode: str = "none",
    dataset_root: Path = ROOT / "exp" / "datasets" / "lsun_cat",
    masks_dir: Path = ROOT / "exp" / "inpainting_masks",
    n_boot: int = 10000,
    bootstrap_seed: int = 42,
    ci_level: float = 0.95,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute paired deltas, seed aggregation, bootstrap CIs, stratification, and top helped/hurt."""
    # 1. Load masks and classify mask groups
    mask_arrays: dict[str, Optional[np.ndarray]] = {}
    mask_groups: dict[str, str] = {}
    for run in runs:
        mname = run["mask_name"]
        if mname not in mask_arrays:
            mfile = resolve_mask_file(run["mask_path"], mname, masks_dir)
            marr = np.load(mfile, allow_pickle=False) if mfile is not None else None
            mask_arrays[mname] = marr
            mask_groups[mname] = classify_mask_group(mname, mask_array=marr)

    # 2. Compute GT image texture complexity (unique across rel_input)
    unique_inputs: dict[str, Optional[Path]] = {}
    for run in runs:
        for rec in run["records"]:
            rel_inp = rec["input"]
            if rel_inp not in unique_inputs:
                unique_inputs[rel_inp] = resolve_gt_image_path(rel_inp, dataset_root)

    texture_scores: dict[str, dict[str, Any]] = {}
    sobel_values: list[float] = []
    for rel_inp, gt_path in sorted(unique_inputs.items()):
        if gt_path is not None and gt_path.is_file():
            tex = compute_texture_complexity(load_rgb_01(gt_path))
            texture_scores[rel_inp] = {
                "gt_path": str(gt_path),
                "sobel_mean_grad_mag": tex["sobel_mean_grad_mag"],
                "laplacian_var": tex["laplacian_var"],
            }
            sobel_values.append(tex["sobel_mean_grad_mag"])

    sobel_median_threshold = float(np.median(sobel_values)) if sobel_values else None
    for rel_inp, tinfo in texture_scores.items():
        tinfo["texture_stratum"] = (
            "high_edge" if tinfo["sobel_mean_grad_mag"] > sobel_median_threshold else "flat"
        )

    # 3. Group runs by (mask_name, sigma_y) and pair against baseline_mode
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["mask_name"], float(run["sigma_y"]))].append(run)

    paired_rows: list[dict[str, Any]] = []
    raw_seed_deltas: list[dict[str, Any]] = []
    seed_breakdown: dict[str, Any] = {}

    for (mask_name, sigma_y), group_runs in sorted(groups.items()):
        by_cond_seed: dict[tuple[str, int], dict[str, Any]] = {}
        for run in group_runs:
            key = (run["condition_label"], run["seed"])
            if key in by_cond_seed:
                raise ValueError(
                    f"Duplicate run for condition={run['condition_label']}, mask={mask_name}, "
                    f"sigma_y={sigma_y}, seed={run['seed']}"
                )
            by_cond_seed[key] = run

        base_seeds = {seed for (cond, seed) in by_cond_seed if cond == baseline_mode}
        other_conds = _order_conditions([cond for (cond, _) in by_cond_seed if cond != baseline_mode])

        for cond in other_conds:
            cond_seeds = sorted(seed for (c, seed) in by_cond_seed if c == cond)
            matched_seeds = [s for s in cond_seeds if s in base_seeds]
            if not matched_seeds:
                continue

            # Map index -> list of per-seed paired dicts
            per_index_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            per_seed_records: dict[int, list[dict[str, Any]]] = defaultdict(list)

            for s in matched_seeds:
                base_run = by_cond_seed[(baseline_mode, s)]
                meth_run = by_cond_seed[(cond, s)]
                common_indices = sorted(
                    set(base_run["records_by_index"]) & set(meth_run["records_by_index"])
                )
                for idx in common_indices:
                    brec = base_run["records_by_index"][idx]
                    mrec = meth_run["records_by_index"][idx]
                    if brec["input"] != mrec["input"]:
                        raise ValueError(
                            f"Input mismatch for ({mask_name}, sigma_y={sigma_y}, seed={s}, index={idx}): "
                            f"{brec['input']} vs {mrec['input']}"
                        )
                    bsha = brec.get("sha256")
                    msha = mrec.get("sha256")
                    if bsha and msha and bsha != msha:
                        raise ValueError(
                            f"SHA-256 mismatch for ({mask_name}, sigma_y={sigma_y}, seed={s}, index={idx}): "
                            f"{bsha} vs {msha}"
                        )
                    pair_entry: dict[str, Any] = {
                        "condition": cond,
                        "baseline_mode": baseline_mode,
                        "mask_name": mask_name,
                        "mask_group": mask_groups[mask_name],
                        "sigma_y": sigma_y,
                        "seed": s,
                        "index": idx,
                        "input": mrec["input"],
                    }
                    for mkey in METRIC_SPECS:
                        bval = brec[mkey]
                        mval = mrec[mkey]
                        dval = (mval - bval) if (bval is not None and mval is not None) else None
                        pair_entry[f"baseline_{mkey}"] = bval
                        pair_entry[f"method_{mkey}"] = mval
                        pair_entry[f"delta_{mkey}"] = dval
                    per_index_seed[idx].append(pair_entry)
                    per_seed_records[s].append(pair_entry)
                    raw_seed_deltas.append(pair_entry)

            # Aggregate per-image across matched seeds
            for idx in sorted(per_index_seed):
                seed_entries = per_index_seed[idx]
                rel_input = seed_entries[0]["input"]
                tex_info = texture_scores.get(rel_input, {})
                agg_row: dict[str, Any] = {
                    "condition": cond,
                    "baseline_mode": baseline_mode,
                    "mask_name": mask_name,
                    "mask_group": mask_groups[mask_name],
                    "sigma_y": sigma_y,
                    "index": idx,
                    "input": rel_input,
                    "n_seeds": len(seed_entries),
                    "seeds": ",".join(str(e["seed"]) for e in seed_entries),
                    "sobel_mean_grad_mag": tex_info.get("sobel_mean_grad_mag"),
                    "laplacian_var": tex_info.get("laplacian_var"),
                    "texture_stratum": tex_info.get("texture_stratum", "unknown"),
                }
                for mkey in METRIC_SPECS:
                    bvals = [e[f"baseline_{mkey}"] for e in seed_entries if e[f"baseline_{mkey}"] is not None]
                    mvals = [e[f"method_{mkey}"] for e in seed_entries if e[f"method_{mkey}"] is not None]
                    dvals = [e[f"delta_{mkey}"] for e in seed_entries if e[f"delta_{mkey}"] is not None]
                    agg_row[f"baseline_{mkey}_mean"] = float(np.mean(bvals)) if bvals else None
                    agg_row[f"method_{mkey}_mean"] = float(np.mean(mvals)) if mvals else None
                    agg_row[f"delta_{mkey}"] = float(np.mean(dvals)) if dvals else None
                    if len(dvals) >= 2:
                        agg_row[f"delta_{mkey}_seed_std"] = float(np.std(dvals, ddof=1))
                    elif len(dvals) == 1:
                        agg_row[f"delta_{mkey}_seed_std"] = 0.0
                    else:
                        agg_row[f"delta_{mkey}_seed_std"] = None
                paired_rows.append(agg_row)

            # Secondary seed breakdown when multiple seeds are matched
            if len(matched_seeds) > 1:
                group_key = f"{cond}|{mask_name}|{sigma_y}"
                per_seed_stats: dict[str, Any] = {}
                all_raw_for_group = []
                for s in matched_seeds:
                    s_rows = per_seed_records[s]
                    all_raw_for_group.extend(s_rows)
                    per_seed_stats[str(s)] = {
                        mkey: bootstrap_mean_ci(
                            [r[f"delta_{mkey}"] for r in s_rows],
                            n_boot=n_boot,
                            seed=bootstrap_seed,
                            ci_level=ci_level,
                        )
                        for mkey in METRIC_SPECS
                    }
                pooled_stats = {
                    mkey: bootstrap_mean_ci(
                        [r[f"delta_{mkey}"] for r in all_raw_for_group],
                        n_boot=n_boot,
                        seed=bootstrap_seed,
                        ci_level=ci_level,
                    )
                    for mkey in METRIC_SPECS
                }
                seed_breakdown[group_key] = {
                    "matched_seeds": matched_seeds,
                    "per_seed": per_seed_stats,
                    "pooled_over_image_seeds": pooled_stats,
                }

    # 4. Stratified summary across (condition, sigma_y)
    stratified_rows: list[dict[str, Any]] = []
    stratified_nested: dict[str, Any] = {}

    cond_sigma_pairs = sorted({(r["condition"], r["sigma_y"]) for r in paired_rows}, key=lambda x: (_order_conditions([x[0]])[0], x[1]))
    for cond, sigma_y in cond_sigma_pairs:
        sub = [r for r in paired_rows if r["condition"] == cond and r["sigma_y"] == sigma_y]
        strata_defs: list[tuple[str, str, list[dict[str, Any]]]] = [
            ("overall", "all", sub),
        ]
        for mname in sorted({r["mask_name"] for r in sub}):
            strata_defs.append(("mask_name", mname, [r for r in sub if r["mask_name"] == mname]))
        for mgroup in ("scattered", "contiguous"):
            grp_rows = [r for r in sub if r["mask_group"] == mgroup]
            if grp_rows:
                strata_defs.append(("mask_group", mgroup, grp_rows))
        if sobel_median_threshold is not None:
            for tname in ("high_edge", "flat"):
                t_rows = [r for r in sub if r["texture_stratum"] == tname]
                strata_defs.append(("texture_stratum", tname, t_rows))
            for mgroup in ("scattered", "contiguous"):
                if not any(r["mask_group"] == mgroup for r in sub):
                    continue
                for tname in ("high_edge", "flat"):
                    x_rows = [
                        r for r in sub
                        if r["mask_group"] == mgroup and r["texture_stratum"] == tname
                    ]
                    strata_defs.append(("mask_group_x_texture", f"{mgroup}|{tname}", x_rows))

        key_cs = f"{cond}|sigma_y={sigma_y}"
        stratified_nested[key_cs] = {}
        for stratum_type, stratum_value, srows in strata_defs:
            summary_metrics = _summarize_group(
                srows, n_boot=n_boot, seed=bootstrap_seed, ci_level=ci_level
            )
            stratified_nested[key_cs].setdefault(stratum_type, {})[stratum_value] = summary_metrics
            for metric_key, mstats in summary_metrics.items():
                stratified_rows.append(
                    {
                        "condition": cond,
                        "baseline_mode": baseline_mode,
                        "sigma_y": sigma_y,
                        "stratum_type": stratum_type,
                        "stratum_value": stratum_value,
                        "metric": metric_key,
                        "n": mstats["n"],
                        "baseline_mean": mstats["baseline_mean"],
                        "method_mean": mstats["method_mean"],
                        "delta_mean": mstats["mean"],
                        "delta_std": mstats["std"],
                        "delta_median": mstats["median"],
                        "ci_low": mstats["ci_low"],
                        "ci_high": mstats["ci_high"],
                        "improved_count": mstats["improved_count"],
                        "worsened_count": mstats["worsened_count"],
                        "tied_count": mstats["tied_count"],
                    }
                )

    # 5. Top helped / hurt indices per (condition, mask_name, sigma_y)
    top_helped_hurt: dict[str, Any] = {}
    cm_pairs = sorted({(r["condition"], r["mask_name"], r["sigma_y"]) for r in paired_rows})
    for cond, mname, sigma_y in cm_pairs:
        sub = [
            r for r in paired_rows
            if r["condition"] == cond and r["mask_name"] == mname and r["sigma_y"] == sigma_y
        ]

        def _entry_view(r: dict[str, Any]) -> dict[str, Any]:
            return {
                "index": r["index"],
                "input": r["input"],
                "delta_full_psnr_db": r["delta_full_psnr_db"],
                "delta_full_vgg_lpips": r["delta_full_vgg_lpips"],
                "texture_stratum": r["texture_stratum"],
            }

        psnr_valid = [r for r in sub if r["delta_full_psnr_db"] is not None]
        lpips_valid = [r for r in sub if r["delta_full_vgg_lpips"] is not None]

        helped_psnr = sorted(psnr_valid, key=lambda r: (-r["delta_full_psnr_db"], r["index"]))[:top_k]
        hurt_psnr = sorted(psnr_valid, key=lambda r: (r["delta_full_psnr_db"], r["index"]))[:top_k]
        helped_lpips = sorted(lpips_valid, key=lambda r: (r["delta_full_vgg_lpips"], r["index"]))[:top_k]
        hurt_lpips = sorted(lpips_valid, key=lambda r: (-r["delta_full_vgg_lpips"], r["index"]))[:top_k]

        top_helped_hurt[f"{cond}|{mname}|sigma_y={sigma_y}"] = {
            "by_full_psnr_db": {
                "top_helped": [_entry_view(r) for r in helped_psnr],
                "top_hurt": [_entry_view(r) for r in hurt_psnr],
            },
            "by_full_vgg_lpips": {
                "top_helped": [_entry_view(r) for r in helped_lpips],
                "top_hurt": [_entry_view(r) for r in hurt_lpips],
            },
        }

    return {
        "metadata": {
            "baseline_mode": baseline_mode,
            "n_boot": n_boot,
            "bootstrap_seed": bootstrap_seed,
            "ci_level": ci_level,
            "top_k": top_k,
            "sobel_median_threshold": sobel_median_threshold,
            "sobel_split_rule": "sobel_mean_grad_mag > median -> high_edge; <= median -> flat",
            "n_runs_loaded": len(runs),
            "n_unique_gt_images": len(texture_scores),
        },
        "mask_arrays": mask_arrays,
        "mask_groups": mask_groups,
        "texture_scores": texture_scores,
        "paired_rows": paired_rows,
        "stratified_rows": stratified_rows,
        "stratified_summary": stratified_nested,
        "seed_breakdown": seed_breakdown,
        "top_helped_hurt": top_helped_hurt,
    }


def _choose_sample_indices(
    mask_name: str,
    paired_rows: list[dict[str, Any]],
    available_indices: list[int],
    explicit_indices: Optional[list[int]],
) -> list[int]:
    if explicit_indices:
        chosen = [idx for idx in explicit_indices if idx in available_indices]
        if chosen:
            return chosen
    candidates = [
        r for r in paired_rows
        if r["mask_name"] == mask_name and r["delta_full_psnr_db"] is not None
    ]
    if candidates:
        ordered = sorted(candidates, key=lambda r: (r["delta_full_psnr_db"], r["index"]))
        hurt = ordered[0]["index"]
        mid = ordered[len(ordered) // 2]["index"]
        helped = ordered[-1]["index"]
        picked: list[int] = []
        for idx in (helped, mid, hurt):
            if idx in available_indices and idx not in picked:
                picked.append(idx)
        if picked:
            return picked
    return available_indices[:3]


def _pick_runs_for_mask(
    runs: list[dict[str, Any]],
    mask_name: str,
    visual_seed: Optional[int],
) -> tuple[Optional[float], Optional[int], dict[str, dict[str, Any]]]:
    mask_runs = [r for r in runs if r["mask_name"] == mask_name]
    if not mask_runs:
        return None, None, {}
    sigma_values = sorted({float(r["sigma_y"]) for r in mask_runs})
    chosen_sigma = sigma_values[0]
    sigma_runs = [r for r in mask_runs if float(r["sigma_y"]) == chosen_sigma]

    seeds_by_cond: dict[str, set[int]] = defaultdict(set)
    for r in sigma_runs:
        seeds_by_cond[r["condition_label"]].add(int(r["seed"]))

    common_seeds = set.intersection(*seeds_by_cond.values()) if seeds_by_cond else set()
    if visual_seed is not None and visual_seed in common_seeds:
        chosen_seed = visual_seed
    elif common_seeds:
        chosen_seed = min(common_seeds)
    elif visual_seed is not None:
        chosen_seed = visual_seed
    else:
        chosen_seed = min(int(r["seed"]) for r in sigma_runs)

    cond_map: dict[str, dict[str, Any]] = {}
    for r in sigma_runs:
        if int(r["seed"]) == chosen_seed:
            cond_map[r["condition_label"]] = r
    return chosen_sigma, chosen_seed, cond_map


def render_figures(
    runs: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
    dataset_root: Path,
    baseline_mode: str = "none",
    sample_indices_override: Optional[list[int]] = None,
    visual_seed: Optional[int] = None,
) -> list[str]:
    """Generate comparison grids, error heatmaps, and delta CI forest plot."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    mask_names = sorted({r["mask_name"] for r in runs})
    mask_arrays: dict[str, Optional[np.ndarray]] = analysis["mask_arrays"]
    mask_groups: dict[str, str] = analysis["mask_groups"]
    paired_rows: list[dict[str, Any]] = analysis["paired_rows"]

    for mask_name in mask_names:
        chosen_sigma, chosen_seed, cond_runs = _pick_runs_for_mask(runs, mask_name, visual_seed)
        if not cond_runs:
            continue
        ordered_conds = _order_conditions(cond_runs.keys())
        common_indices = sorted(
            set.intersection(*(set(r["records_by_index"].keys()) for r in cond_runs.values()))
        )
        if not common_indices:
            continue

        selected_indices = _choose_sample_indices(
            mask_name=mask_name,
            paired_rows=[r for r in paired_rows if r["sigma_y"] == chosen_sigma],
            available_indices=common_indices,
            explicit_indices=sample_indices_override,
        )
        mask_2d = mask_arrays.get(mask_name)

        # 1. Comparison grid
        n_rows = len(selected_indices)
        n_cols = 2 + len(ordered_conds)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.8 * n_cols, 3.0 * n_rows),
            squeeze=False,
        )

        for row_idx, sample_idx in enumerate(selected_indices):
            ref_rec = cond_runs[ordered_conds[0]]["records_by_index"][sample_idx]
            gt_path = resolve_gt_image_path(ref_rec["input"], dataset_root)
            if gt_path is not None and gt_path.is_file():
                gt_01 = load_rgb_01(gt_path)
            else:
                gt_01 = load_rgb_01(ref_rec["restored_abs_path"])

            if mask_2d is not None:
                obs_bool = (mask_2d != 0)[..., None]
                hole_bool = (mask_2d == 0)[..., None]
                masked_01 = gt_01 * obs_bool + 0.5 * hole_bool
            else:
                masked_01 = gt_01

            axes[row_idx, 0].imshow(np.clip(gt_01, 0.0, 1.0))
            axes[row_idx, 0].set_title(
                f"Ground Truth\n#{sample_idx} ({Path(ref_rec['input']).name})",
                fontsize=9,
            )
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(np.clip(masked_01, 0.0, 1.0))
            axes[row_idx, 1].set_title(f"Masked Input\n({mask_name})", fontsize=9)
            axes[row_idx, 1].axis("off")

            base_rec = cond_runs.get(baseline_mode, {}).get("records_by_index", {}).get(sample_idx)
            base_psnr = base_rec.get("full_psnr_db") if base_rec else None

            for col_offset, cond in enumerate(ordered_conds):
                ax = axes[row_idx, 2 + col_offset]
                rec = cond_runs[cond]["records_by_index"][sample_idx]
                restored_01 = load_rgb_01(rec["restored_abs_path"])
                psnr_val = rec.get("full_psnr_db")
                lpips_val = rec.get("full_vgg_lpips")
                psnr_str = f"{psnr_val:.2f}dB" if psnr_val is not None else "inf"
                lpips_str = f"{lpips_val:.3f}" if lpips_val is not None else "n/a"
                delta_str = ""
                if cond != baseline_mode and psnr_val is not None and base_psnr is not None:
                    delta_str = f" ({psnr_val - base_psnr:+.2f})"
                disp_name = CONDITION_DISPLAY_NAMES.get(cond, cond)
                ax.imshow(np.clip(restored_01, 0.0, 1.0))
                ax.set_title(f"{disp_name}\n{psnr_str}{delta_str} / {lpips_str}", fontsize=8.5)
                ax.axis("off")

        fig.suptitle(
            f"Inpainting Comparison — {mask_name} (sigma_y={chosen_sigma}, seed={chosen_seed})",
            fontsize=11,
            y=1.01,
        )
        fig.tight_layout()
        grid_path = figures_dir / f"comparison_grid_{mask_name}.png"
        fig.savefig(grid_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated_files.append(str(grid_path.relative_to(output_dir).as_posix()))

        # 2. Error heatmap
        err_cols = 1 + len(ordered_conds)
        fig_e, axes_e = plt.subplots(
            n_rows,
            err_cols,
            figsize=(2.9 * err_cols + 0.6, 3.0 * n_rows),
            squeeze=False,
        )
        error_maps: list[list[np.ndarray]] = []
        gt_images: list[np.ndarray] = []
        mae_annotations: list[list[tuple[float, Optional[float]]]] = []
        all_errs: list[np.ndarray] = []

        for sample_idx in selected_indices:
            ref_rec = cond_runs[ordered_conds[0]]["records_by_index"][sample_idx]
            gt_path = resolve_gt_image_path(ref_rec["input"], dataset_root)
            gt_01 = load_rgb_01(gt_path) if (gt_path is not None and gt_path.is_file()) else load_rgb_01(ref_rec["restored_abs_path"])
            gt_images.append(gt_01)
            row_errs = []
            row_maes = []
            for cond in ordered_conds:
                rec = cond_runs[cond]["records_by_index"][sample_idx]
                restored_01 = load_rgb_01(rec["restored_abs_path"])
                err_map = np.mean(np.abs(restored_01 - gt_01), axis=2)
                full_mae = float(np.mean(err_map))
                if mask_2d is not None and np.any(mask_2d == 0):
                    hole_mae = float(np.mean(err_map[mask_2d == 0]))
                else:
                    hole_mae = None
                row_errs.append(err_map)
                row_maes.append((full_mae, hole_mae))
                all_errs.append(err_map.ravel())
            error_maps.append(row_errs)
            mae_annotations.append(row_maes)

        vmax = min(1.0, max(0.15, float(np.quantile(np.concatenate(all_errs), 0.995)))) if all_errs else 0.5
        last_im = None
        for row_idx, sample_idx in enumerate(selected_indices):
            axes_e[row_idx, 0].imshow(np.clip(gt_images[row_idx], 0.0, 1.0))
            axes_e[row_idx, 0].set_title(f"Ground Truth\n#{sample_idx}", fontsize=9)
            axes_e[row_idx, 0].axis("off")

            for col_offset, cond in enumerate(ordered_conds):
                ax = axes_e[row_idx, 1 + col_offset]
                emap = error_maps[row_idx][col_offset]
                full_mae, hole_mae = mae_annotations[row_idx][col_offset]
                last_im = ax.imshow(emap, cmap="inferno", vmin=0.0, vmax=vmax)
                if mask_2d is not None and mask_groups.get(mask_name) == "contiguous":
                    hole_bin = (mask_2d == 0).astype(float)
                    ax.contour(hole_bin, levels=[0.5], colors="cyan", linestyles="--", linewidths=0.9)
                disp_name = CONDITION_DISPLAY_NAMES.get(cond, cond)
                hole_str = f"{hole_mae:.3f}" if hole_mae is not None else "n/a"
                ax.set_title(f"{disp_name}\nMAE {full_mae:.3f} / {hole_str}", fontsize=8.5)
                ax.axis("off")

        fig_e.suptitle(
            f"Mean Absolute Error |x_hat - x| — {mask_name} (sigma_y={chosen_sigma}, seed={chosen_seed})",
            fontsize=11,
            y=1.01,
        )
        fig_e.tight_layout()
        if last_im is not None:
            cbar = fig_e.colorbar(last_im, ax=axes_e[:, 1:].ravel().tolist(), fraction=0.02, pad=0.02)
            cbar.set_label("Mean Abs Error |x_hat - x|", fontsize=9)
        heatmap_path = figures_dir / f"error_heatmap_{mask_name}.png"
        fig_e.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close(fig_e)
        generated_files.append(str(heatmap_path.relative_to(output_dir).as_posix()))

    # 3. Delta CI forest plot
    stratified_rows = analysis["stratified_rows"]
    forest_candidates = [
        r for r in stratified_rows
        if r["stratum_type"] in {"overall", "mask_group", "texture_stratum"}
        and r["metric"] in {"full_psnr_db", "full_vgg_lpips"}
        and r["delta_mean"] is not None
    ]
    if forest_candidates:
        labels_ordered = []
        seen_labels = set()
        for r in forest_candidates:
            lbl = f"{r['condition']} (s={r['sigma_y']})\n{r['stratum_type']}:{r['stratum_value']}"
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                labels_ordered.append((r["condition"], r["sigma_y"], r["stratum_type"], r["stratum_value"], lbl))

        fig_f, (ax_psnr, ax_lpips) = plt.subplots(
            1, 2, figsize=(12.5, max(3.5, 0.52 * len(labels_ordered) + 1.2)), sharey=True
        )
        y_positions = np.arange(len(labels_ordered))

        for ax, metric_key, title in (
            (ax_psnr, "full_psnr_db", "Delta Full PSNR (dB) [+ helps]"),
            (ax_lpips, "full_vgg_lpips", "Delta Full VGG-LPIPS [- helps]"),
        ):
            means = []
            err_low = []
            err_high = []
            valid_y = []
            for y_i, (cond, sig, stype, sval, _) in enumerate(labels_ordered):
                match = next(
                    (
                        r for r in forest_candidates
                        if r["condition"] == cond
                        and r["sigma_y"] == sig
                        and r["stratum_type"] == stype
                        and r["stratum_value"] == sval
                        and r["metric"] == metric_key
                    ),
                    None,
                )
                if match is not None and match["delta_mean"] is not None:
                    m = float(match["delta_mean"])
                    lo = float(match["ci_low"]) if match["ci_low"] is not None else m
                    hi = float(match["ci_high"]) if match["ci_high"] is not None else m
                    valid_y.append(y_i)
                    means.append(m)
                    err_low.append(max(0.0, m - lo))
                    err_high.append(max(0.0, hi - m))
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            if valid_y:
                ax.errorbar(
                    means,
                    valid_y,
                    xerr=[err_low, err_high],
                    fmt="o",
                    capsize=3.5,
                    markersize=5,
                    color="#1f77b4" if metric_key == "full_psnr_db" else "#d62728",
                )
            ax.set_title(title, fontsize=10)
            ax.grid(axis="x", linestyle=":", alpha=0.5)

        ax_psnr.set_yticks(y_positions)
        ax_psnr.set_yticklabels([item[4] for item in labels_ordered], fontsize=8.5)
        ax_psnr.invert_yaxis()
        fig_f.suptitle("Paired Differences vs Baseline (95% Bootstrap CIs)", fontsize=11)
        fig_f.tight_layout()
        forest_path = figures_dir / "delta_ci_forest.png"
        fig_f.savefig(forest_path, dpi=200, bbox_inches="tight")
        plt.close(fig_f)
        generated_files.append(str(forest_path.relative_to(output_dir).as_posix()))

    return generated_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=None, help="Run directories or label=path entries")
    parser.add_argument("--results-root", type=Path, default=None, help="Parent directory containing run subdirs")
    parser.add_argument("--baseline-mode", type=str, default="none")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "exp" / "datasets" / "lsun_cat")
    parser.add_argument("--masks-dir", type=Path, default=ROOT / "exp" / "inpainting_masks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None)
    parser.add_argument("--visual-seed", type=int, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.runs and args.results_root is None:
        parser.error("Provide --runs and/or --results-root")
    if args.n_boot < 1:
        parser.error("--n-boot must be >= 1")
    if not 0.0 < args.ci_level < 1.0:
        parser.error("--ci-level must be in (0, 1)")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    output_dir = args.output_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    masks_dir = args.masks_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(run_specs=args.runs, results_root=args.results_root)
    analysis = analyze_runs(
        runs=runs,
        baseline_mode=args.baseline_mode,
        dataset_root=dataset_root,
        masks_dir=masks_dir,
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
        ci_level=args.ci_level,
        top_k=args.top_k,
    )

    paired_fields = [
        "condition",
        "baseline_mode",
        "mask_name",
        "mask_group",
        "sigma_y",
        "index",
        "input",
        "n_seeds",
        "seeds",
        "sobel_mean_grad_mag",
        "laplacian_var",
        "texture_stratum",
    ]
    for mkey in METRIC_SPECS:
        paired_fields.extend(
            [
                f"baseline_{mkey}_mean",
                f"method_{mkey}_mean",
                f"delta_{mkey}",
                f"delta_{mkey}_seed_std",
            ]
        )
    write_csv(output_dir / "paired_deltas.csv", paired_fields, analysis["paired_rows"])

    strat_fields = [
        "condition",
        "baseline_mode",
        "sigma_y",
        "stratum_type",
        "stratum_value",
        "metric",
        "n",
        "baseline_mean",
        "method_mean",
        "delta_mean",
        "delta_std",
        "delta_median",
        "ci_low",
        "ci_high",
        "improved_count",
        "worsened_count",
        "tied_count",
    ]
    write_csv(output_dir / "stratified_summary.csv", strat_fields, analysis["stratified_rows"])

    generated_figures = render_figures(
        runs=runs,
        analysis=analysis,
        output_dir=output_dir,
        dataset_root=dataset_root,
        baseline_mode=args.baseline_mode,
        sample_indices_override=args.sample_indices,
        visual_seed=args.visual_seed,
    )

    summary_payload = {
        "metadata": analysis["metadata"],
        "mask_groups": analysis["mask_groups"],
        "texture_scores": analysis["texture_scores"],
        "stratified_summary": analysis["stratified_summary"],
        "seed_breakdown": analysis["seed_breakdown"],
        "top_helped_hurt": analysis["top_helped_hurt"],
        "figures": generated_figures,
        "paired_count": len(analysis["paired_rows"]),
    }
    write_json(output_dir / "analysis_summary.json", summary_payload)
    print(
        f"Analysis complete: {len(analysis['paired_rows'])} paired rows, "
        f"{len(generated_figures)} figures written to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
