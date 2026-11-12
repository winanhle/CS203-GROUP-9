"""Train E0/E1/E3 on one frozen, verified ResNet-50 feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate(model, features, metadata, targets) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    with torch.inference_mode():
        _, predictions = model(features, metadata)
    prediction_array = predictions.float().cpu().numpy()
    return regression_metrics(targets.cpu().numpy(), prediction_array), prediction_array


def build_model(experiment: str):
    if experiment in {"E2", "E4"}:
        return CachedGatedRegressor()
    return CachedFeatureRegressor(metadata=experiment in {"E1", "E3"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["E0", "E1", "E2", "E3", "E4"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--metadata-dropout", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.manifest).sort_values("image_name", ignore_index=True)
    cache = np.load(args.features)
    cache_names = cache["image_name"].tolist()
    features_np = cache["features"].astype(np.float32, copy=False)
    if cache_names != manifest["image_name"].tolist() or features_np.shape != (11125, 2048):
        raise AssertionError("Feature cache does not match canonical manifest order")

    train_mask = manifest["split"].to_numpy() == "train"
    apex = manifest[APEX_COLUMNS].to_numpy(np.float64)
    train_apex = apex[train_mask]
    median = np.median(train_apex, axis=0)
    q25, q75 = np.percentile(train_apex, [25, 75], axis=0)
    scale = np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
    standardized_apex = ((apex - median) / scale).astype(np.float32)
    availability = np.ones_like(standardized_apex, dtype=np.float32)
    metadata_np = np.concatenate([standardized_apex, availability], axis=1)

    split_data = {}
    for split in ("train", "val", "test"):
        mask = manifest["split"].to_numpy() == split
        split_data[split] = {
            "names": manifest.loc[mask, "image_name"].tolist(),
            "features": torch.from_numpy(features_np[mask]).to(device),
            "metadata": torch.from_numpy(metadata_np[mask]).to(device),
            "targets": torch.from_numpy(manifest.loc[mask, "MOS"].to_numpy(np.float32)).to(device),
        }

    uses_metadata = args.experiment != "E0"
    dropout_probability = args.metadata_dropout if args.experiment in {"E3", "E4"} else 0.0
    model = build_model(args.experiment).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.L1Loss()
    generator = torch.Generator().manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    config = {
        "experiment": args.experiment,
        "track": "RESOURCE_ADAPTED_FROZEN_RESNET50_IMAGENET_V2",
        "source_commit": args.source_commit,
        "seed": args.seed,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "metadata_dropout": dropout_probability,
        "fusion": (
            "gated_hidden_256"
            if args.experiment in {"E2", "E4"}
            else ("additive_offset" if uses_metadata else "image_only")
        ),
        "manifest_sha256": sha256(args.manifest),
        "feature_cache_sha256": sha256(args.features),
        "apex_columns": APEX_COLUMNS,
        "apex_train_median": median.tolist(),
        "apex_train_iqr": scale.tolist(),
        "metadata_input": "8 train-robust-scaled APEX values + 8 explicit availability mask bits",
        "loss": "E0: L1(combined); metadata variants: 0.5*L1(generic)+0.5*L1(combined)",
        "selection": "maximum validation SRCC; test evaluated once after selection",
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = args.output_dir / "epochs.jsonl"
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    best_srcc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    train = split_data["train"]

    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(train["targets"]), generator=generator)
            losses = []
            for start in range(0, len(permutation), args.batch_size):
                indices = permutation[start : start + args.batch_size].to(device)
                batch_metadata = train["metadata"][indices]
                if dropout_probability:
                    keep = (
                        torch.rand((len(indices), 1), device=device) >= dropout_probability
                    ).to(batch_metadata.dtype)
                    batch_metadata = batch_metadata * keep
                optimizer.zero_grad(set_to_none=True)
                generic, combined = model(train["features"][indices], batch_metadata if uses_metadata else None)
                targets = train["targets"][indices]
                if uses_metadata:
                    loss = 0.5 * criterion(generic, targets) + 0.5 * criterion(combined, targets)
                else:
                    loss = criterion(combined, targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            val_metrics, _ = evaluate(
                model,
                split_data["val"]["features"],
                split_data["val"]["metadata"] if uses_metadata else None,
                split_data["val"]["targets"],
            )
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            improved = val_metrics["srcc"] > best_srcc + 1e-12
            if improved:
                best_srcc = val_metrics["srcc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                atomic_save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": config,
                        "val_metrics": val_metrics,
                        "generator_state": generator.get_state(),
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1
            atomic_save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_srcc": best_srcc,
                    "generator_state": generator.get_state(),
                },
                last_path,
            )
            if epochs_without_improvement >= args.patience:
                break

    elapsed = time.perf_counter() - started
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    selected = build_model(args.experiment).to(device)
    selected.load_state_dict(best_payload["model_state"])
    val_metrics, val_predictions = evaluate(
        selected,
        split_data["val"]["features"],
        split_data["val"]["metadata"] if uses_metadata else None,
        split_data["val"]["targets"],
    )
    test_metrics, test_predictions = evaluate(
        selected,
        split_data["test"]["features"],
        split_data["test"]["metadata"] if uses_metadata else None,
        split_data["test"]["targets"],
    )
    prediction_frames = []
    for split, predictions in (("val", val_predictions), ("test", test_predictions)):
        prediction_frames.append(
            pd.DataFrame(
                {
                    "image_name": split_data[split]["names"],
                    "split": split,
                    "mos": split_data[split]["targets"].cpu().numpy(),
                    "prediction": predictions,
                }
            )
        )
    predictions_path = args.output_dir / "predictions.csv"
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        predictions_path, index=False, lineterminator="\n"
    )
    parameter_count = sum(parameter.numel() for parameter in selected.parameters())
    image_only_parameter_count = sum(
        parameter.numel() for parameter in CachedFeatureRegressor(metadata=False).parameters()
    )
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "experiment": args.experiment,
        "track": config["track"],
        "source_commit": args.source_commit,
        "best_epoch": best_epoch,
        "epochs_completed": epoch,
        "runtime_seconds": elapsed,
        "parameter_count": parameter_count,
        "metadata_parameter_count": parameter_count - image_only_parameter_count,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "artifacts": {
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            "best_checkpoint": {"path": str(best_path.resolve()), "sha256": sha256(best_path)},
            "last_checkpoint": {"path": str(last_path.resolve()), "sha256": sha256(last_path)},
            "epoch_log": {"path": str(log_path.resolve()), "sha256": sha256(log_path)},
            "predictions": {"path": str(predictions_path.resolve()), "sha256": sha256(predictions_path)},
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
