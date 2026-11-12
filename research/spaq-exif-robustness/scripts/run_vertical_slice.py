"""Run a 16-real-image forward/backward/checkpoint/reload/resume slice."""

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
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from spaq_iqa.data import SpaqImageDataset
from spaq_iqa.metrics import regression_metrics
from spaq_iqa.models import build_resnet50_regressor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        outputs = model(batch["image"].to(device)).squeeze(1).cpu().numpy()
        for image_name, target, prediction in zip(batch["image_name"], batch["mos"], outputs):
            rows.append(
                {"image_name": image_name, "mos": float(target), "prediction": float(prediction)}
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-count", type=int, default=16)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_parquet(args.manifest)
    subset = (
        manifest.loc[manifest["split"] == "train"]
        .sort_values("image_name")
        .head(args.image_count)
        .copy()
    )
    missing = [name for name in subset["image_name"] if not (args.image_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing real images: {missing[:3]}")

    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    dataset = SpaqImageDataset(subset, args.image_root, transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0
    )
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_resnet50_regressor(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses = []
    iterator = iter(train_loader)
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch["image"].to(device)).squeeze(1)
        loss = criterion(predictions, batch["mos"].to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - started

    checkpoint = args.output_dir / f"checkpoint_step{args.steps}.pt"
    checkpoint_payload = {
        "step": args.steps,
        "seed": args.seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "manifest_sha256": sha256(args.manifest),
    }
    atomic_save(checkpoint_payload, checkpoint)
    before_reload = predict(model, eval_loader, device)

    reloaded = build_resnet50_regressor(pretrained=False).to(device)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-4, weight_decay=1e-4)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["model_state"])
    reloaded_optimizer.load_state_dict(payload["optimizer_state"])
    after_reload = predict(reloaded, eval_loader, device)
    if not np.array_equal(before_reload["prediction"].to_numpy(), after_reload["prediction"].to_numpy()):
        raise AssertionError("Checkpoint reload changed predictions")

    # Prove optimizer/model resume with one additional real batch and durable checkpoint.
    resume_batch = next(iter(train_loader))
    reloaded.train()
    reloaded_optimizer.zero_grad(set_to_none=True)
    resume_predictions = reloaded(resume_batch["image"].to(device)).squeeze(1)
    resume_loss = criterion(resume_predictions, resume_batch["mos"].to(device))
    resume_loss.backward()
    reloaded_optimizer.step()
    resumed_checkpoint = args.output_dir / f"checkpoint_step{args.steps + 1}.pt"
    atomic_save(
        {
            **payload,
            "step": args.steps + 1,
            "model_state": reloaded.state_dict(),
            "optimizer_state": reloaded_optimizer.state_dict(),
        },
        resumed_checkpoint,
    )

    predictions_path = args.output_dir / f"predictions_step{args.steps}.csv"
    after_reload.to_csv(predictions_path, index=False, lineterminator="\n")
    metrics = regression_metrics(
        after_reload["mos"].to_numpy(), after_reload["prediction"].to_numpy()
    )
    config = {
        "run_type": "G2_VERTICAL_SLICE_NOT_BASELINE_RESULT",
        "architecture": "torchvision_resnet50_scalar_head",
        "pretrained_weights": "ResNet50_Weights.IMAGENET1K_V2",
        "image_count": args.image_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "loss": "L1",
        "optimizer": "AdamW(lr=1e-4,weight_decay=1e-4)",
        "manifest_sha256": sha256(args.manifest),
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTION_VERIFIED",
        "claim_scope": f"{args.image_count}-image vertical slice only; metrics are resubstitution sanity values",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "elapsed_train_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "resume_loss": float(resume_loss.detach().cpu()),
        "metrics": metrics,
        "reload_predictions_bit_exact": True,
        "image_names": subset["image_name"].tolist(),
        "artifacts": {
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            f"checkpoint_step{args.steps}": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
            f"checkpoint_step{args.steps + 1}": {
                "path": str(resumed_checkpoint.resolve()),
                "sha256": sha256(resumed_checkpoint),
            },
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": sha256(predictions_path),
            },
        },
        "losses": losses,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
