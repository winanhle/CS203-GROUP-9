"""Extract one frozen ResNet-50 feature cache for matched head experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from spaq_iqa.data import SpaqImageDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest).sort_values("image_name", ignore_index=True)
    dataset = SpaqImageDataset(manifest, args.image_root, ResNet50_Weights.IMAGENET1K_V2.transforms())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    feature_batches = []
    names = []
    sample_300_seconds = None
    seen = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model(images)
            feature_batches.append(features.float().cpu().numpy())
            names.extend(batch["image_name"])
            seen += len(batch["image_name"])
            if sample_300_seconds is None and seen >= 300:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                sample_300_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    features = np.concatenate(feature_batches, axis=0)
    if features.shape != (11125, 2048) or names != manifest["image_name"].tolist():
        raise AssertionError("Feature cache shape/order differs from canonical manifest")
    if not np.isfinite(features).all():
        raise AssertionError("Feature cache contains non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, image_name=np.asarray(names), features=features)
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTION_VERIFIED",
        "architecture": "torchvision_resnet50_fc_identity",
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "preprocessing": str(ResNet50_Weights.IMAGENET1K_V2.transforms()),
        "manifest_sha256": sha256(args.manifest),
        "cache_path": str(args.output.resolve()),
        "cache_sha256": sha256(args.output),
        "cache_bytes": args.output.stat().st_size,
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "sample_300_seconds_including_io": sample_300_seconds,
        "all_11125_seconds_including_io": elapsed,
        "images_per_second": len(dataset) / elapsed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
