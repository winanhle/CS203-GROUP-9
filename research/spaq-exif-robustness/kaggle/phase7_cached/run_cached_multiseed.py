"""Standalone Kaggle replication of Phase 7 E0/E1/E2 across three seeds."""

from __future__ import annotations

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


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
MANIFEST_SHA = "abff8b1a5f7d2e74d0c8874655fbe40d5e75de75bf61e65753c7085cee15800d"
CACHE_SHA = "b639bb6f2365f31004410af1a71a35529a485de849d70e6924f7fbb2dc3975ed"
SEEDS = (20261109, 20261110, 20261111)
EXPERIMENTS = ("E0", "E1", "E2")
APEX_COLUMNS = [
    "exif_focal_length", "exif_aperture_value", "exif_shutter_time_value",
    "exif_scene_brightness_value", "exif_sensitivity_value", "exif_flash",
    "exif_time_sin", "exif_time_cos",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_one(pattern: str) -> Path:
    matches = sorted(INPUT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern}, found {len(matches)}")
    return matches[0]


class AdditiveModel(nn.Module):
    def __init__(self, metadata: bool) -> None:
        super().__init__()
        self.image_head = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
        self.metadata_head = nn.Linear(16, 1, bias=False) if metadata else None

    def forward(self, features, metadata=None):
        generic = self.image_head(features).squeeze(1)
        if self.metadata_head is None:
            return generic, generic
        combined = generic + self.metadata_head(metadata).squeeze(1)
        return generic, combined


class GatedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(2048)
        self.projection = nn.Linear(2048, 256)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(256, 1)
        self.metadata_modulation = nn.Linear(16, 512)
        nn.init.zeros_(self.metadata_modulation.weight)
        nn.init.zeros_(self.metadata_modulation.bias)

    def score(self, hidden):
        return self.output(self.dropout(self.activation(hidden))).squeeze(1)

    def forward(self, features, metadata):
        hidden = self.projection(self.normalization(features))
        generic = self.score(hidden)
        gamma, beta = self.metadata_modulation(metadata).chunk(2, dim=1)
        return generic, self.score((1.0 + gamma.tanh()) * hidden + beta)


def build_model(experiment):
    return GatedModel() if experiment == "E2" else AdditiveModel(metadata=experiment == "E1")


def metrics(targets, predictions):
    target_rank = pd.Series(targets).rank().to_numpy()
    prediction_rank = pd.Series(predictions).rank().to_numpy()
    return {
        "srcc": float(np.corrcoef(target_rank, prediction_rank)[0, 1]),
        "plcc": float(np.corrcoef(targets, predictions)[0, 1]),
        "mae": float(np.mean(np.abs(targets - predictions))),
    }


def evaluate(model, split, uses_metadata):
    model.eval()
    with torch.inference_mode():
        _, predictions = model(split["features"], split["metadata"] if uses_metadata else None)
    values = predictions.float().cpu().numpy()
    return metrics(split["targets"].cpu().numpy(), values), values


def train_one(experiment, seed, split_data, manifest_hash, cache_hash, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    uses_metadata = experiment != "E0"
    model = build_model(experiment).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss()
    generator = torch.Generator().manual_seed(seed)
    run_id = f"{experiment.lower()}_s{seed}"
    run_dir = OUTPUT / run_id
    run_dir.mkdir(exist_ok=True)
    best_path = run_dir / "best.pt"
    log_path = run_dir / "epochs.jsonl"
    best_srcc, best_epoch, stale = -float("inf"), 0, 0
    started = time.perf_counter()
    train = split_data["train"]
    log_rows = []
    for epoch in range(1, 301):
        model.train()
        permutation = torch.randperm(len(train["targets"]), generator=generator)
        losses = []
        for start in range(0, len(permutation), 256):
            indices = permutation[start:start + 256].to(device)
            optimizer.zero_grad(set_to_none=True)
            generic, combined = model(train["features"][indices], train["metadata"][indices] if uses_metadata else None)
            targets = train["targets"][indices]
            loss = 0.5 * criterion(generic, targets) + 0.5 * criterion(combined, targets) if uses_metadata else criterion(combined, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _ = evaluate(model, split_data["val"], uses_metadata)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{key}": value for key, value in val_metrics.items()}}
        log_rows.append(row)
        if val_metrics["srcc"] > best_srcc + 1e-12:
            best_srcc, best_epoch, stale = val_metrics["srcc"], epoch, 0
            temporary = best_path.with_suffix(".tmp")
            torch.save({"experiment": experiment, "seed": seed, "epoch": epoch, "model_state": model.state_dict(),
                        "validation_metrics": val_metrics, "manifest_sha256": manifest_hash, "feature_cache_sha256": cache_hash}, temporary)
            os.replace(temporary, best_path)
        else:
            stale += 1
            if stale >= 30:
                break
    log_path.write_text("".join(json.dumps(row) + "\n" for row in log_rows), encoding="utf-8")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    selected = build_model(experiment).to(device)
    selected.load_state_dict(checkpoint["model_state"])
    val_metrics, _ = evaluate(selected, split_data["val"], uses_metadata)
    # Held-out test is evaluated once, only after validation selection.
    test_metrics, test_predictions = evaluate(selected, split_data["test"], uses_metadata)
    predictions_path = run_dir / "predictions.csv"
    pd.DataFrame({"image_name": split_data["test"]["names"], "MOS": split_data["test"]["targets"].cpu().numpy(),
                  "prediction": test_predictions}).to_csv(predictions_path, index=False)
    result = {
        "run_id": run_id, "experiment": experiment, "seed": seed, "best_epoch": best_epoch,
        "epochs_completed": len(log_rows), "runtime_seconds": time.perf_counter() - started,
        "validation_metrics": val_metrics, "test_metrics": test_metrics,
        "artifacts": {
            "checkpoint": {"path": str(best_path.relative_to(OUTPUT)), "sha256": sha256(best_path)},
            "epoch_log": {"path": str(log_path.relative_to(OUTPUT)), "sha256": sha256(log_path)},
            "predictions": {"path": str(predictions_path.relative_to(OUTPUT)), "sha256": sha256(predictions_path)},
        },
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    manifest_path = locate_one("**/manifest.parquet")
    cache_path = locate_one("**/spaq_resnet50_imagenet_v2_abff8b1a.npz")
    manifest_hash, cache_hash = sha256(manifest_path), sha256(cache_path)
    if manifest_hash != MANIFEST_SHA or cache_hash != CACHE_SHA:
        raise RuntimeError("Manifest or feature-cache hash mismatch")
    manifest = pd.read_parquet(manifest_path).sort_values("image_name", ignore_index=True)
    cache = np.load(cache_path)
    if cache["image_name"].tolist() != manifest.image_name.tolist() or cache["features"].shape != (11125, 2048):
        raise RuntimeError("Feature cache order/shape mismatch")
    train_mask = manifest.split.to_numpy() == "train"
    apex = manifest[APEX_COLUMNS].to_numpy(np.float64)
    median = np.median(apex[train_mask], axis=0)
    q25, q75 = np.percentile(apex[train_mask], [25, 75], axis=0)
    scale = np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
    metadata = np.concatenate([((apex - median) / scale).astype(np.float32), np.ones_like(apex, dtype=np.float32)], axis=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = cache["features"].astype(np.float32, copy=False)
    split_data = {}
    for split in ("train", "val", "test"):
        mask = manifest.split.to_numpy() == split
        split_data[split] = {
            "names": manifest.loc[mask, "image_name"].tolist(),
            "features": torch.from_numpy(features[mask]).to(device),
            "metadata": torch.from_numpy(metadata[mask]).to(device),
            "targets": torch.from_numpy(manifest.loc[mask, "MOS"].to_numpy(np.float32)).to(device),
        }
    results = [train_one(experiment, seed, split_data, manifest_hash, cache_hash, device)
               for seed in SEEDS for experiment in EXPERIMENTS]
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "SUCCESS",
        "experiment": "PHASE7_CACHED_MULTI_SEED_KAGGLE_REPLICATION",
        "test_policy": "validation SRCC selection; each test prediction emitted once from selected checkpoint",
        "manifest_sha256": manifest_hash, "feature_cache_sha256": cache_hash,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "runs": results,
    }
    (OUTPUT / "cached_multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
