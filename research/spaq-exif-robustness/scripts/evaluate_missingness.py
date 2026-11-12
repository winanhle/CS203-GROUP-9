"""Evaluate E1/E3 under fixed record-level EXIF availability masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spaq_iqa.metrics import regression_metrics
from spaq_iqa.models import CachedFeatureRegressor, CachedGatedRegressor


APEX_COLUMNS = [
    "exif_focal_length",
    "exif_aperture_value",
    "exif_shutter_time_value",
    "exif_scene_brightness_value",
    "exif_sensitivity_value",
    "exif_flash",
    "exif_time_sin",
    "exif_time_cos",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = (
        CachedGatedRegressor()
        if payload["config"].get("fusion") == "gated_hidden_256"
        else CachedFeatureRegressor(metadata=True)
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload["config"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--e1-checkpoint", type=Path, required=True)
    parser.add_argument("--e3-checkpoint", type=Path, required=True)
    parser.add_argument("--e2-checkpoint", type=Path)
    parser.add_argument("--e4-checkpoint", type=Path)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_parquet(args.manifest).sort_values("image_name", ignore_index=True)
    cache = np.load(args.features)
    if cache["image_name"].tolist() != manifest["image_name"].tolist():
        raise AssertionError("Feature cache order differs from manifest")
    test_mask = manifest["split"].to_numpy() == "test"
    features = torch.from_numpy(cache["features"][test_mask].astype(np.float32, copy=False)).to(device)
    targets = manifest.loc[test_mask, "MOS"].to_numpy(np.float32)
    image_names = manifest.loc[test_mask, "image_name"].tolist()
    apex = manifest.loc[test_mask, APEX_COLUMNS].to_numpy(np.float64)

    checkpoints = [("E1", args.e1_checkpoint), ("E3", args.e3_checkpoint)]
    checkpoints.extend(
        (name, checkpoint)
        for name, checkpoint in (("E2", args.e2_checkpoint), ("E4", args.e4_checkpoint))
        if checkpoint is not None
    )
    models = {}
    for name, checkpoint in checkpoints:
        model, config = load_model(checkpoint, device)
        median = np.asarray(config["apex_train_median"])
        scale = np.asarray(config["apex_train_iqr"])
        values = ((apex - median) / scale).astype(np.float32)
        full_metadata = np.concatenate([values, np.ones_like(values)], axis=1)
        models[name] = (model, torch.from_numpy(full_metadata).to(device))

    fractions = (0.0, 0.25, 0.5, 1.0)
    stochastic_seeds = (20261110, 20261111, 20261112, 20261113, 20261114)
    rows = []
    metrics = []
    for experiment, (model, full_metadata) in models.items():
        for fraction in fractions:
            seeds = stochastic_seeds if fraction in {0.25, 0.5} else stochastic_seeds[:1]
            for seed in seeds:
                rng = np.random.default_rng(seed)
                missing_count = int(round(fraction * len(image_names)))
                missing_indices = rng.permutation(len(image_names))[:missing_count]
                available = np.ones(len(image_names), dtype=np.float32)
                available[missing_indices] = 0
                masked_metadata = full_metadata * torch.from_numpy(available[:, None]).to(device)
                with torch.inference_mode():
                    generic, combined = model(features, masked_metadata)
                predictions = combined.float().cpu().numpy()
                result = regression_metrics(targets, predictions)
                metrics.append(
                    {
                        "experiment": experiment,
                        "missing_fraction": fraction,
                        "mask_seed": seed,
                        "missing_count": missing_count,
                        **result,
                    }
                )
                rows.extend(
                    {
                        "experiment": experiment,
                        "missing_fraction": fraction,
                        "mask_seed": seed,
                        "image_name": image_name,
                        "mos": float(target),
                        "metadata_available": int(flag),
                        "prediction": float(prediction),
                        "generic_prediction": float(generic_prediction),
                    }
                    for image_name, target, flag, prediction, generic_prediction in zip(
                        image_names,
                        targets,
                        available,
                        predictions,
                        generic.float().cpu().numpy(),
                    )
                )

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.predictions_output, index=False, lineterminator="\n")
    metrics_frame = pd.DataFrame(metrics)
    aggregate = (
        metrics_frame.groupby(["experiment", "missing_fraction"])[["srcc", "plcc", "mae"]]
        .agg(["mean", "std", "min", "max"])
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate_records = aggregate.reset_index().fillna(0).to_dict(orient="records")
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "test_rows": len(image_names),
        "mask_unit": "entire EXIF record per image",
        "fractions": list(fractions),
        "stochastic_mask_seeds": list(stochastic_seeds),
        "manifest_sha256": sha256(args.manifest),
        "feature_cache_sha256": sha256(args.features),
        "checkpoint_sha256": {name: sha256(checkpoint) for name, checkpoint in checkpoints},
        "per_run_metrics": metrics,
        "aggregate_metrics": aggregate_records,
        "predictions_path": str(args.predictions_output.resolve()),
        "predictions_sha256": sha256(args.predictions_output),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "per_run_metrics": metrics, "aggregate_metrics": aggregate_records}, indent=2))


if __name__ == "__main__":
    main()
