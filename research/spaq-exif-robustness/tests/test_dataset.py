from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from spaq_iqa.data import SpaqImageDataset


def test_dataset_reads_real_rgb_image(tmp_path: Path) -> None:
    image_path = tmp_path / "00001.jpg"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(image_path)
    manifest = pd.DataFrame(
        {"image_name": ["00001.jpg"], "MOS": [50.0], "split": ["train"], "group_id": ["g1"]}
    )
    sample = SpaqImageDataset(manifest, tmp_path)[0]
    assert sample["image"].mode == "RGB"
    assert sample["image"].size == (8, 6)
    assert sample["mos"].item() == 50.0


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "image_name": ["00001.jpg", "00001.jpg"],
            "MOS": [1.0, 2.0],
            "split": ["train", "train"],
            "group_id": ["g1", "g1"],
        }
    )
    with pytest.raises(ValueError):
        SpaqImageDataset(manifest, tmp_path)
