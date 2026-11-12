"""Standalone Kaggle GPU/data/checkpoint smoke test for SPAQ."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.transforms import v2


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
EXPECTED_MANIFEST_SHA256 = "abff8b1a5f7d2e74d0c8874655fbe40d5e75de75bf61e65753c7085cee15800d"
EXPECTED_SAMPLE_TREE_SHA256 = "cb2fe0f3a59efda58deefc4ed2af8ad4bcf7ef84fc8c5a8bd6fc7d27c4098f5d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Images(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.transform = v2.Compose(
            [v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.image_name) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(float(row.MOS), dtype=torch.float32)


def locate_one(pattern: str) -> Path:
    matches = sorted(INPUT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} under /kaggle/input, found {len(matches)}")
    return matches[0]


def image_tree_sha256(image_root: Path, image_names: list[str]) -> str:
    """Hash selected filename + per-file digest, independent of container format."""
    digest = hashlib.sha256()
    for image_name in image_names:
        path = image_root / image_name
        digest.update(image_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def main() -> None:
    torch.manual_seed(20261109)
    manifest_path = locate_one("**/manifest.parquet")
    manifest_hash = sha256(manifest_path)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Manifest hash mismatch: {manifest_hash}")
    manifest = pd.read_parquet(manifest_path)
    sample = manifest.loc[manifest.split == "train"].sort_values("image_name").head(300)
    archive_matches = sorted(INPUT.glob("**/spaq.tar"))
    if len(archive_matches) == 1:
        input_mode = "tar_archive"
        archive_path = archive_matches[0]
        image_root = WORKING / "smoke_images"
        image_root.mkdir(exist_ok=True)
        extract_started = time.perf_counter()
        with tarfile.open(archive_path, "r:") as archive:
            for image_name in sample.image_name:
                target = image_root / image_name
                source = archive.extractfile(f"spaq/{image_name}")
                if source is None:
                    raise RuntimeError(f"Missing archive member {image_name}")
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(source.read())
                os.replace(temporary, target)
        extract_seconds = time.perf_counter() - extract_started
        input_image_count = 300
    else:
        input_mode = "kaggle_expanded_tar"
        first_image = locate_one("**/00001.jpg")
        image_root = first_image.parent
        integrity_started = time.perf_counter()
        input_image_count = len(list(image_root.glob("*.jpg")))
        integrity_seconds = time.perf_counter() - integrity_started
        extract_seconds = 0.0
        if input_image_count != 11125:
            raise RuntimeError(f"Expanded image count mismatch: {input_image_count}")

    integrity_started = time.perf_counter()
    sample_tree_hash = image_tree_sha256(image_root, sample.image_name.tolist())
    integrity_seconds = time.perf_counter() - integrity_started
    if sample_tree_hash != EXPECTED_SAMPLE_TREE_SHA256:
        raise RuntimeError(f"300-image sample integrity mismatch: {sample_tree_hash}")

    dataset = Images(sample, image_root)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle smoke requires a CUDA accelerator")
    device = torch.device("cuda")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    torch.cuda.reset_peak_memory_stats()

    inference_started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            model(images.to(device, non_blocking=True))
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    iterator = iter(train_loader)
    train_started = time.perf_counter()
    model.train()
    losses = []
    for _ in range(20):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images.to(device)).squeeze(1)
        loss = criterion(predictions, targets.to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started

    checkpoint = WORKING / "smoke_checkpoint.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 20}, temporary)
    os.replace(temporary, checkpoint)
    reloaded = resnet18(weights=None)
    reloaded.fc = nn.Linear(reloaded.fc.in_features, 1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(payload["model"])
    reloaded.to(device).train()
    resumed_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-4)
    resumed_optimizer.load_state_dict(payload["optimizer"])
    resume_images, resume_targets = next(iter(train_loader))
    resumed_optimizer.zero_grad(set_to_none=True)
    resume_predictions = reloaded(resume_images.to(device)).squeeze(1)
    resume_loss = criterion(resume_predictions, resume_targets.to(device))
    resume_loss.backward()
    resumed_optimizer.step()
    torch.cuda.synchronize()

    disk = shutil.disk_usage(WORKING)
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTION_VERIFIED",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "working_disk_total_bytes": disk.total,
        "working_disk_free_bytes": disk.free,
        "input_mode": input_mode,
        "input_image_count": input_image_count,
        "sample_tree_sha256": sample_tree_hash,
        "expected_sample_tree_sha256": EXPECTED_SAMPLE_TREE_SHA256,
        "integrity_seconds": integrity_seconds,
        "manifest_sha256": manifest_hash,
        "sample_count": 300,
        "extract_seconds": extract_seconds,
        "inference_300_seconds": inference_seconds,
        "train_20_steps_seconds": train_seconds,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload_succeeded": True,
        "optimizer_resume_step_succeeded": True,
        "resume_step_loss": float(resume_loss.detach().cpu()),
        "projected_11125_inference_seconds": inference_seconds * (11125 / 300),
        "projection_with_20_percent_buffer_seconds": inference_seconds * (11125 / 300) * 1.2,
    }
    (WORKING / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
