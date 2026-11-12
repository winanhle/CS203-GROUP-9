"""Evaluate EXIF pairing, feature groups, and missingness across Phase 7 seeds."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spaq_iqa.metrics import regression_metrics
from spaq_iqa.models import CachedFeatureRegressor, CachedGatedRegressor


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261109, 20261110, 20261111)
MODELS = ("E1", "E2")
PERMUTATION_SEEDS = (7301, 7302, 7303, 7304, 7305)
MISSINGNESS_SEEDS = (20261110, 20261111, 20261112, 20261113, 20261114)
APEX_COLUMNS = [
    "exif_focal_length", "exif_aperture_value", "exif_shutter_time_value",
    "exif_scene_brightness_value", "exif_sensitivity_value", "exif_flash",
    "exif_time_sin", "exif_time_cos",
]
GROUPS = {"optics": (0, 1), "exposure": (2, 3, 4), "flash": (5,), "time": (6, 7)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(model: str, seed: int) -> Path:
    name = model.lower()
    prefix = "" if seed == 20261109 else "phase7_"
    return ROOT / "artifacts" / "runs" / f"{prefix}{name}_frozen_r50_s{seed}" / "best.pt"


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = CachedGatedRegressor() if payload["config"].get("fusion") == "gated_hidden_256" else CachedFeatureRegressor(metadata=True)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload["config"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = ROOT / "data" / "manifest.parquet"
    cache_path = ROOT / "artifacts" / "features" / "spaq_resnet50_imagenet_v2_abff8b1a.npz"
    manifest = pd.read_parquet(manifest_path).sort_values("image_name", ignore_index=True)
    cache = np.load(cache_path)
    if cache["image_name"].tolist() != manifest["image_name"].tolist():
        raise AssertionError("Feature cache order differs from manifest")
    test_mask = manifest["split"].to_numpy() == "test"
    features = torch.from_numpy(cache["features"][test_mask].astype(np.float32, copy=False)).to(device)
    targets = manifest.loc[test_mask, "MOS"].to_numpy(np.float32)
    names = manifest.loc[test_mask, "image_name"].tolist()
    apex = manifest.loc[test_mask, APEX_COLUMNS].to_numpy(np.float64)

    metric_rows = []
    prediction_rows = []
    checkpoint_hashes = {}
    for seed in SEEDS:
        for model_name in MODELS:
            path = checkpoint_path(model_name, seed)
            model, config = load_model(path, device)
            checkpoint_hashes[f"{model_name}_s{seed}"] = sha256(path)
            median = np.asarray(config["apex_train_median"])
            scale = np.asarray(config["apex_train_iqr"])
            values = ((apex - median) / scale).astype(np.float32)
            metadata = np.concatenate([values, np.ones_like(values)], axis=1).astype(np.float32)

            conditions: list[tuple[str, int | None, np.ndarray]] = [("intact", None, metadata)]
            for permutation_seed in PERMUTATION_SEEDS:
                permutation = np.random.default_rng(permutation_seed).permutation(len(names))
                conditions.append(("whole_record_permutation", permutation_seed, metadata[permutation]))
            for group, indices in GROUPS.items():
                leave_out = metadata.copy()
                group_only = np.zeros_like(metadata)
                columns = list(indices) + [index + 8 for index in indices]
                leave_out[:, columns] = 0
                group_only[:, columns] = metadata[:, columns]
                conditions.append((f"leave_out_{group}", None, leave_out))
                conditions.append((f"group_only_{group}", None, group_only))
            for fraction in (0.25, 0.5, 1.0):
                seeds = MISSINGNESS_SEEDS if fraction in {0.25, 0.5} else MISSINGNESS_SEEDS[:1]
                for mask_seed in seeds:
                    masked = metadata.copy()
                    count = int(round(fraction * len(names)))
                    missing = np.random.default_rng(mask_seed).permutation(len(names))[:count]
                    masked[missing] = 0
                    conditions.append((f"missing_{fraction:.2f}", mask_seed, masked))

            for condition, condition_seed, metadata_array in conditions:
                metadata_tensor = torch.from_numpy(metadata_array).to(device)
                with torch.inference_mode():
                    generic, combined = model(features, metadata_tensor)
                predictions = combined.float().cpu().numpy()
                result = regression_metrics(targets, predictions)
                metric_rows.append(
                    {"training_seed": seed, "model": model_name, "condition": condition, "condition_seed": condition_seed, **result}
                )
                prediction_rows.extend(
                    {
                        "training_seed": seed,
                        "model": model_name,
                        "condition": condition,
                        "condition_seed": condition_seed,
                        "image_name": image_name,
                        "mos": float(target),
                        "prediction": float(prediction),
                        "generic_prediction": float(generic_prediction),
                    }
                    for image_name, target, prediction, generic_prediction in zip(
                        names, targets, predictions, generic.float().cpu().numpy()
                    )
                )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_path = ROOT / "artifacts" / "runs" / "phase7_metadata_diagnostics" / "predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    metrics_path = ROOT / "evidence" / "phase7_metadata_diagnostic_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    permutation = metrics_frame.loc[metrics_frame["condition"].eq("whole_record_permutation")]
    intact = metrics_frame.loc[metrics_frame["condition"].eq("intact")]
    reliability = []
    for seed in SEEDS:
        for model in MODELS:
            intact_row = intact.loc[intact["training_seed"].eq(seed) & intact["model"].eq(model)].iloc[0]
            perm = permutation.loc[permutation["training_seed"].eq(seed) & permutation["model"].eq(model)]
            reliability.append(
                {
                    "training_seed": seed,
                    "model": model,
                    "intact_srcc": float(intact_row["srcc"]),
                    "permuted_srcc_mean": float(perm["srcc"].mean()),
                    "intact_minus_permuted_srcc": float(intact_row["srcc"] - perm["srcc"].mean()),
                    "intact_exceeds_every_permutation": bool((intact_row["srcc"] > perm["srcc"]).all()),
                }
            )
    reliability_frame = pd.DataFrame(reliability)
    aggregate = (
        metrics_frame.groupby(["model", "condition"])[["srcc", "plcc", "mae"]]
        .agg(["count", "mean", "std", "min", "max"])
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index().fillna(0)
    h2 = {}
    for model in MODELS:
        frame = reliability_frame.loc[reliability_frame["model"].eq(model)]
        h2[model] = {
            "mean_intact_minus_permuted_srcc": float(frame["intact_minus_permuted_srcc"].mean()),
            "positive_all_training_seeds": bool((frame["intact_minus_permuted_srcc"] > 0).all()),
            "supports_registered_rule": bool((frame["intact_minus_permuted_srcc"] > 0).all()),
        }

    figures = ROOT / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    missing_rows = aggregate.loc[aggregate["condition"].str.startswith("missing_")].copy()
    intact_rows = aggregate.loc[aggregate["condition"].eq("intact")].copy()
    intact_rows["missing_fraction"] = 0.0
    missing_rows["missing_fraction"] = missing_rows["condition"].str.removeprefix("missing_").astype(float)
    missing_plot = pd.concat([intact_rows, missing_rows], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model, frame in missing_plot.groupby("model"):
        frame = frame.sort_values("missing_fraction")
        ax.errorbar(
            frame["missing_fraction"] * 100,
            frame["srcc_mean"],
            yerr=frame["srcc_std"],
            marker="o",
            capsize=3,
            label=model,
        )
    ax.set(xlabel="Missing EXIF records (%)", ylabel="Test SRCC", title="Phase 7 missingness across seeds and masks")
    ax.legend()
    fig.tight_layout()
    missing_figure = figures / "phase7_multiseed_missingness.png"
    fig.savefig(missing_figure, dpi=180)
    plt.close(fig)

    ablation_conditions = [f"leave_out_{group}" for group in GROUPS]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(ablation_conditions))
    width = 0.34
    for offset, model in enumerate(MODELS):
        intact_mean = float(
            aggregate.loc[aggregate["model"].eq(model) & aggregate["condition"].eq("intact"), "srcc_mean"].iloc[0]
        )
        effects = [
            intact_mean
            - float(
                aggregate.loc[
                    aggregate["model"].eq(model) & aggregate["condition"].eq(condition), "srcc_mean"
                ].iloc[0]
            )
            for condition in ablation_conditions
        ]
        ax.bar(x + (offset - 0.5) * width, effects, width, label=model)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, [condition.removeprefix("leave_out_") for condition in ablation_conditions])
    ax.set(ylabel="Intact SRCC - leave-group-out SRCC", title="Inference-only EXIF group ablation")
    ax.legend()
    fig.tight_layout()
    ablation_figure = figures / "phase7_feature_group_ablation.png"
    fig.savefig(ablation_figure, dpi=180)
    plt.close(fig)

    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "evaluation_type": "fixed-checkpoint inference diagnostics; feature-group tests are not retrained",
        "training_seeds": list(SEEDS),
        "permutation_seeds": list(PERMUTATION_SEEDS),
        "missingness_seeds": list(MISSINGNESS_SEEDS),
        "test_rows": len(names),
        "manifest_sha256": sha256(manifest_path),
        "feature_cache_sha256": sha256(cache_path),
        "checkpoint_sha256": checkpoint_hashes,
        "metrics": metric_rows,
        "aggregate_metrics": aggregate.to_dict(orient="records"),
        "reliability_by_seed": reliability,
        "H2_registered_evaluation": h2,
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "metrics_sha256": sha256(metrics_path),
        "predictions_path": str(predictions_path.relative_to(ROOT)),
        "predictions_sha256": sha256(predictions_path),
        "figure_sha256": {
            str(missing_figure.relative_to(ROOT)): sha256(missing_figure),
            str(ablation_figure.relative_to(ROOT)): sha256(ablation_figure),
        },
        "limitations": [
            "Permutation and feature-group ablations are inference interventions on fixed models, not retrained causal feature-selection experiments.",
            "Permutation can create implausible metadata combinations and is interpreted only as a correct-pair diagnostic.",
        ],
    }
    summary_path = ROOT / "evidence" / "phase7_metadata_diagnostics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "H2": h2, "metric_rows": len(metric_rows)}, indent=2))


if __name__ == "__main__":
    main()
