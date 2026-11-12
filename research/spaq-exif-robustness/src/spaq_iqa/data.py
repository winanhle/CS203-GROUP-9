"""SPAQ image dataset backed by the canonical manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SpaqImageDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: Path,
        transform: Callable | None = None,
    ) -> None:
        required = {"image_name", "MOS", "split", "group_id"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"Manifest is missing {sorted(required - set(manifest.columns))}")
        if manifest["image_name"].duplicated().any():
            raise ValueError("Dataset manifest has duplicate image names")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        image_path = self.image_root / row["image_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "mos": torch.tensor(float(row["MOS"]), dtype=torch.float32),
            "image_name": str(row["image_name"]),
            "group_id": str(row["group_id"]),
        }
