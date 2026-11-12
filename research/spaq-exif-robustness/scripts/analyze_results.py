"""Reproducible paired statistics, subgroup analysis, errors, and figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


EXPERIMENTS = ("e0", "e1", "e2", "e3", "e4")
CONTRASTS = (("e1", "e0"), ("e3", "e1"), ("e2", "e1"), ("e4", "e2"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "srcc": float(spearmanr(y, prediction).statistic),
        "plcc": float(pearsonr(y, prediction).statistic),
        "mae": float(np.mean(np.abs(y - prediction))),
    }


def paired_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    control: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    point_candidate = metrics(y, candidate)
    point_control = metrics(y, control)
    draws = {name: np.empty(resamples, dtype=np.float64) for name in point_candidate}
    for draw in range(resamples):
        index = rng.integers(0, len(y), size=len(y))
        candidate_metrics = metrics(y[index], candidate[index])
        control_metrics = metrics(y[index], control[index])
        for name in draws:
            draws[name][draw] = candidate_metrics[name] - control_metrics[name]
    output: dict[str, dict[str, float]] = {}
    for name, values in draws.items():
        low, high = np.quantile(values, [0.025, 0.975])
        output[name] = {
            "candidate": point_candidate[name],
            "control": point_control[name],
            "delta": point_candidate[name] - point_control[name],
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return output


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.parquet"))
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.manifest)
    predictions: dict[str, pd.DataFrame] = {}
    input_hashes = {"manifest": sha256(args.manifest)}

    for experiment in EXPERIMENTS:
        path = args.runs_root / f"{experiment}_frozen_r50_s{args.seed}" / "predictions.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[frame["split"].eq("test")].sort_values("image_name").reset_index(drop=True)
        if frame["image_name"].duplicated().any() or len(frame) != 2228:
            raise ValueError(f"{experiment}: expected 2,228 unique test predictions")
        predictions[experiment] = frame
        input_hashes[f"{experiment}_predictions"] = sha256(path)

    reference = predictions["e0"][["image_name", "mos"]]
    for experiment, frame in predictions.items():
        if not reference.equals(frame[["image_name", "mos"]]):
            raise ValueError(f"{experiment}: test identifiers or labels do not match E0")

    y = reference["mos"].to_numpy(dtype=np.float64)
    point_metrics = {
        experiment: metrics(y, frame["prediction"].to_numpy(dtype=np.float64))
        for experiment, frame in predictions.items()
    }
    rng = np.random.default_rng(args.seed)
    bootstrap = {}
    bootstrap_rows = []
    for candidate, control in CONTRASTS:
        result = paired_bootstrap(
            y,
            predictions[candidate]["prediction"].to_numpy(dtype=np.float64),
            predictions[control]["prediction"].to_numpy(dtype=np.float64),
            args.bootstrap_resamples,
            rng,
        )
        contrast = f"{candidate.upper()}-{control.upper()}"
        bootstrap[contrast] = result
        for metric_name, values in result.items():
            bootstrap_rows.append({"contrast": contrast, "metric": metric_name, **values})
    bootstrap_path = args.evidence_dir / f"paired_bootstrap_s{args.seed}.csv"
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)

    train = manifest.loc[manifest["split"].eq("train")]
    iso_threshold = float(train["source_iso"].quantile(0.90))
    valid_train_exposure = train.loc[train["source_exposure_valid"], "source_exposure_time"]
    exposure_threshold = float(valid_train_exposure.quantile(0.90))
    test_manifest = manifest.loc[manifest["split"].eq("test")].copy()
    masks = {
        "all_test": np.ones(len(test_manifest), dtype=bool),
        "high_iso": test_manifest["source_iso"].to_numpy() >= iso_threshold,
        "long_exposure": (
            test_manifest["source_exposure_valid"].to_numpy(dtype=bool)
            & (test_manifest["source_exposure_time"].to_numpy() >= exposure_threshold)
        ),
        "night_scene": test_manifest["scene_night"].to_numpy() > 0,
    }
    ordered_test = test_manifest.sort_values("image_name").reset_index(drop=True)
    subgroup_rows = []
    for subgroup, unordered_mask in masks.items():
        selected_names = set(test_manifest.loc[unordered_mask, "image_name"])
        mask = ordered_test["image_name"].isin(selected_names).to_numpy()
        for experiment in EXPERIMENTS:
            result = metrics(y[mask], predictions[experiment].loc[mask, "prediction"].to_numpy())
            subgroup_rows.append(
                {"subgroup": subgroup, "n": int(mask.sum()), "experiment": experiment.upper(), **result}
            )
    subgroup_path = args.evidence_dir / f"subgroup_metrics_s{args.seed}.csv"
    subgroup_frame = pd.DataFrame(subgroup_rows)
    subgroup_frame.to_csv(subgroup_path, index=False)

    selected_model = "e1"  # Highest validation SRCC; selection predates this test error review.
    errors = ordered_test[[
        "image_name", "MOS", "source_iso", "source_exposure_time", "source_exposure_valid",
        "scene_night", "scene_indoor", "make", "model",
    ]].copy()
    errors["prediction"] = predictions[selected_model]["prediction"]
    errors["signed_error"] = errors["prediction"] - errors["MOS"]
    errors["absolute_error"] = errors["signed_error"].abs()
    errors = errors.sort_values(["absolute_error", "image_name"], ascending=[False, True]).head(20)
    error_path = args.evidence_dir / f"top_errors_{selected_model}_s{args.seed}.csv"
    errors.to_csv(error_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.hist(manifest["MOS"], bins=30, color="#4472C4", edgecolor="white")
    ax.set(xlabel="Mean opinion score (MOS)", ylabel="Images", title="SPAQ MOS distribution (n=11,125)")
    save_figure(fig, args.figures_dir / "mos_distribution.png")

    mask_columns = [
        "mask_focal_observed", "mask_aperture_observed", "mask_shutter_observed",
        "mask_sensitivity_observed", "mask_scene_brightness_observed", "mask_flash_observed",
        "mask_time_observed",
    ]
    observed = manifest[mask_columns].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.barh([x.removeprefix("mask_").removesuffix("_observed") for x in observed.index], observed.values)
    ax.set(xlim=(0, 1.02), xlabel="Observed fraction", title="Verified embedded EXIF availability")
    save_figure(fig, args.figures_dir / "exif_availability.png")

    ci_lookup = {(row["contrast"], row["metric"]): row for row in bootstrap_rows}
    metric_experiments = ["e0", "e1", "e2", "e3", "e4"]
    values = [point_metrics[x]["srcc"] for x in metric_experiments]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar([x.upper() for x in metric_experiments], values, color="#5B9BD5")
    ax.set(ylim=(0.80, 0.845), ylabel="Test SRCC", title="Matched frozen-feature experiments")
    for index, value in enumerate(values):
        ax.text(index, value + 0.0005, f"{value:.4f}", ha="center", fontsize=8)
    save_figure(fig, args.figures_dir / "experiment_srcc.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=False)
    contrast_labels = [f"{candidate.upper()}-{control.upper()}" for candidate, control in CONTRASTS]
    for ax, metric_name in zip(axes, ("srcc", "plcc"), strict=True):
        points = np.array([ci_lookup[(label, metric_name)]["delta"] for label in contrast_labels])
        lows = np.array([ci_lookup[(label, metric_name)]["ci95_low"] for label in contrast_labels])
        highs = np.array([ci_lookup[(label, metric_name)]["ci95_high"] for label in contrast_labels])
        ax.errorbar(
            np.arange(len(points)), points, yerr=np.vstack((points - lows, highs - points)),
            fmt="o", capsize=4, color="#2F5597",
        )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_xticks(np.arange(len(points)), contrast_labels, rotation=25)
        ax.set(title=f"Paired delta {metric_name.upper()}", ylabel="Candidate - control")
    fig.suptitle(f"Paired bootstrap 95% intervals ({args.bootstrap_resamples:,} resamples)")
    save_figure(fig, args.figures_dir / "paired_bootstrap_cis.png")

    missingness = json.loads((args.evidence_dir / f"missingness_all_s{args.seed}.json").read_text())
    missing_frame = pd.DataFrame(missingness["aggregate_metrics"])
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for experiment, frame in missing_frame.groupby("experiment"):
        frame = frame.sort_values("missing_fraction")
        ax.plot(frame["missing_fraction"] * 100, frame["srcc_mean"], marker="o", label=experiment)
    ax.set(xlabel="Metadata records missing at inference (%)", ylabel="Test SRCC", title="MCAR metadata-missingness stress")
    ax.legend(ncol=4)
    save_figure(fig, args.figures_dir / "missingness_srcc.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=True)
    for ax, subgroup in zip(axes, ("high_iso", "long_exposure", "night_scene"), strict=True):
        frame = subgroup_frame.loc[subgroup_frame["subgroup"].eq(subgroup)]
        ax.bar(frame["experiment"], frame["srcc"], color="#70AD47")
        ax.set(title=f"{subgroup.replace('_', ' ')}\n(n={int(frame['n'].iloc[0])})", ylim=(0.65, 0.90))
    axes[0].set_ylabel("SRCC")
    save_figure(fig, args.figures_dir / "subgroup_srcc.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for experiment in EXPERIMENTS:
        log_path = args.runs_root / f"{experiment}_frozen_r50_s{args.seed}" / "epochs.jsonl"
        log = pd.read_json(log_path, lines=True)
        ax.plot(log["epoch"], log["val_srcc"], label=experiment.upper(), linewidth=1.2)
    ax.set(xlabel="Epoch", ylabel="Validation SRCC", title="Validation trajectories (test not used)")
    ax.legend(ncol=5)
    save_figure(fig, args.figures_dir / "training_curves.png")

    output_files = [bootstrap_path, subgroup_path, error_path, *sorted(args.figures_dir.glob("*.png"))]
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "test_rows": len(y),
        "paired_identifier_check": "PASS",
        "point_metrics": point_metrics,
        "paired_bootstrap": bootstrap,
        "subgroup_definitions": {
            "high_iso": f"source_iso >= train 90th percentile ({iso_threshold:g})",
            "long_exposure": f"valid source_exposure_time >= train-valid 90th percentile ({exposure_threshold:.12g} s)",
            "night_scene": "authentic scene_night annotation > 0",
            "threshold_policy": "thresholds computed on training partition only",
        },
        "subgroup_counts": {name: int((subgroup_frame["subgroup"].eq(name)).mul(subgroup_frame["n"]).max()) for name in masks},
        "error_analysis_model": "E1 selected because it had the highest validation SRCC",
        "top_error_summary": {
            "count": len(errors),
            "overpredictions": int((errors["signed_error"] > 0).sum()),
            "underpredictions": int((errors["signed_error"] < 0).sum()),
            "median_absolute_error": float(errors["absolute_error"].median()),
            "maximum_absolute_error": float(errors["absolute_error"].max()),
            "selection_rule": "20 largest absolute errors on all held-out E1 predictions; deterministic tie-break by image_name",
            "images_not_redistributed": "Underlying SPAQ image licensing is not explicit enough for report redistribution.",
        },
        "input_sha256": input_hashes,
        "output_sha256": {str(path): sha256(path) for path in output_files},
        "limitations": [
            "CIs quantify paired test-image sampling uncertainty, not training-seed uncertainty.",
            "All learned models use one training seed and a frozen ImageNet feature cache.",
            "Capture-condition subgroups are descriptive and were not used for model selection.",
        ],
    }
    summary_path = args.evidence_dir / f"analysis_summary_s{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "summary": str(summary_path), "bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
