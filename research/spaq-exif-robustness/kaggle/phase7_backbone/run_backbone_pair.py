"""Matched frozen versus layer4-fine-tuned ResNet-50 SPAQ experiment.

The script is standalone for Kaggle and also accepts local paths for smoke tests.
Test data are evaluated only after validation-only checkpoint selection.
"""

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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor


EXPECTED_MANIFEST_SHA256 = "abff8b1a5f7d2e74d0c8874655fbe40d5e75de75bf61e65753c7085cee15800d"
EXPECTED_WEIGHTS_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
SEED = 20261109
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def locate_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern} under {root}, found {len(matches)}")
    return matches[0]


class SpaqImages(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, transform: Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.image_root / str(row.image_name)) as source:
            image = self.transform(source.convert("RGB"))
        return image, torch.tensor(float(row.MOS), dtype=torch.float32), str(row.image_name)


class QualityModel(nn.Module):
    def __init__(self, pretrained_state: dict[str, torch.Tensor], mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.backbone = resnet50(weights=None)
        self.backbone.load_state_dict(pretrained_state)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(2048, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if mode == "layer4_finetuned":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        elif mode != "frozen":
            raise ValueError(mode)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen stages must not update BatchNorm running statistics.
        self.backbone.eval()
        if mode and self.mode == "layer4_finetuned":
            self.backbone.layer4.train()
        self.head.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images)).squeeze(1)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    label_rank = pd.Series(labels).rank(method="average").to_numpy()
    prediction_rank = pd.Series(predictions).rank(method="average").to_numpy()
    return {
        "srcc": float(np.corrcoef(label_rank, prediction_rank)[0, 1]),
        "plcc": float(np.corrcoef(labels, predictions)[0, 1]),
        "mae": float(np.mean(np.abs(labels - predictions))),
    }


def evaluate(model, loader, device, target_mean, target_std, include_names=False):
    model.eval()
    labels, predictions, names = [], [], []
    with torch.inference_mode():
        for images, targets, batch_names in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                normalized = model(images.to(device, non_blocking=True))
            predicted = normalized.float().cpu().numpy() * target_std + target_mean
            predictions.extend(predicted.tolist())
            labels.extend(targets.numpy().tolist())
            if include_names:
                names.extend(batch_names)
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    return metrics(labels_array, predictions_array), names, labels_array, predictions_array


def train_arm(mode, frames, image_root, pretrained_state, weights_hash, output, device, args):
    seed_everything(SEED)
    train_transform = Compose([Resize(256), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(MEAN, STD)])
    eval_transform = Compose([Resize((224, 224)), ToTensor(), Normalize(MEAN, STD)])
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        SpaqImages(frames["train"], image_root, train_transform), batch_size=args.batch_size,
        shuffle=True, generator=loader_generator, num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        SpaqImages(frames["val"], image_root, eval_transform), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        SpaqImages(frames["test"], image_root, eval_transform), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    target_mean = float(frames["train"].MOS.mean())
    target_std = float(frames["train"].MOS.std(ddof=0))
    model = QualityModel(pretrained_state, mode).to(device)
    head_parameters = list(model.head.parameters())
    groups = [{"params": head_parameters, "lr": args.head_lr}]
    if mode == "layer4_finetuned":
        groups.append({"params": list(model.backbone.layer4.parameters()), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_function = nn.SmoothL1Loss()
    best_srcc = -float("inf")
    best_epoch = 0
    stale = 0
    epoch_rows = []
    checkpoint_path = output / f"{mode}_best.pt"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            normalized_targets = ((targets - target_mean) / target_std).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predictions = model(images)
                loss = loss_function(predictions, normalized_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_metrics, _, _, _ = evaluate(model, validation_loader, device, target_mean, target_std)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation_metrics}
        epoch_rows.append(row)
        print(json.dumps({"mode": mode, **row}), flush=True)
        if validation_metrics["srcc"] > best_srcc:
            best_srcc = validation_metrics["srcc"]
            best_epoch = epoch
            stale = 0
            trainable_state = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
                if name.startswith("head.") or (mode == "layer4_finetuned" and name.startswith("backbone.layer4."))
            }
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "mode": mode, "seed": SEED, "epoch": epoch, "state": trainable_state,
                    "pretrained_weights_sha256": weights_hash, "target_mean": target_mean, "target_std": target_std,
                    "validation_metrics": validation_metrics,
                }, temporary,
            )
            os.replace(temporary, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state"], strict=False)
    model.to(device)
    # The held-out test is touched exactly once, after validation selection.
    test_metrics, names, labels, predictions = evaluate(model, test_loader, device, target_mean, target_std, True)
    prediction_path = output / f"{mode}_predictions.csv"
    pd.DataFrame({"image_name": names, "MOS": labels, "prediction": predictions}).to_csv(prediction_path, index=False)
    log_path = output / f"{mode}_epochs.jsonl"
    log_path.write_text("".join(json.dumps(row) + "\n" for row in epoch_rows), encoding="utf-8")
    runtime = time.perf_counter() - started
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "mode": mode, "seed": SEED, "best_epoch": best_epoch, "epochs_completed": len(epoch_rows),
        "runtime_seconds": runtime, "trainable_parameters": trainable_count,
        "validation_metrics": payload["validation_metrics"], "test_metrics": test_metrics,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "artifacts": {
            "checkpoint": {"path": checkpoint_path.name, "sha256": sha256(checkpoint_path)},
            "epoch_log": {"path": log_path.name, "sha256": sha256(log_path)},
            "predictions": {"path": prediction_path.name, "sha256": sha256(prediction_path)},
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or locate_one(args.input_root, "**/manifest.parquet")
    weights_path = args.weights or locate_one(args.input_root, "**/resnet50-11ad3fa6.pth")
    image_root = args.image_root or locate_one(args.input_root, "**/00001.jpg").parent
    manifest_hash = sha256(manifest_path)
    weights_hash = sha256(weights_path)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Manifest SHA-256 mismatch: {manifest_hash}")
    if weights_hash != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(f"Pretrained weight SHA-256 mismatch: {weights_hash}")
    manifest = pd.read_parquet(manifest_path)
    if len(manifest) != 11125 or manifest.image_name.nunique() != 11125:
        raise RuntimeError("Manifest row or identifier uniqueness check failed")
    frames = {split: manifest.loc[manifest.split == split].sort_values("image_name").copy() for split in ("train", "val", "test")}
    if args.smoke:
        frames = {"train": frames["train"].head(32), "val": frames["val"].head(16), "test": frames["test"].head(16)}
        args.epochs = 1
        args.patience = 1
        args.workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    results = []
    for mode in ("frozen", "layer4_finetuned"):
        results.append(train_arm(mode, frames, image_root, pretrained_state, weights_hash, args.output, device, args))
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "SUCCESS",
        "experiment": "H4_MATCHED_BACKBONE_PAIR", "exploratory_single_seed": True,
        "test_evaluation_policy": "each arm evaluated once after validation-only checkpoint selection",
        "controls": ["pretrained weights", "manifest split", "seed", "train/eval transforms", "head", "validation policy"],
        "independent_variable": "frozen backbone versus trainable ResNet-50 layer4",
        "manifest_sha256": manifest_hash, "pretrained_weights_sha256": weights_hash,
        "split_counts": {name: len(frame) for name, frame in frames.items()},
        "config": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "arms": results,
    }
    summary_path = args.output / "backbone_pair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
