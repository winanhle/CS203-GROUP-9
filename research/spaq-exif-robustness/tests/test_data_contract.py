from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_manifest_contract() -> None:
    manifest = pd.read_parquet(ROOT / "data" / "manifest.parquet")
    assert len(manifest) == 11125
    assert manifest["image_name"].nunique() == 11125
    assert manifest["split"].value_counts().to_dict() == {"train": 7786, "test": 2228, "val": 1111}
    assert manifest["group_id"].nunique() == 10340
    assert manifest["MOS"].min() == 2.0
    assert manifest["MOS"].max() == 96.0
    apex_columns = [
        "exif_focal_length",
        "exif_aperture_value",
        "exif_shutter_time_value",
        "exif_scene_brightness_value",
        "exif_sensitivity_value",
        "exif_flash",
        "exif_time_sin",
        "exif_time_cos",
    ]
    assert manifest[apex_columns].notna().all().all()
    assert (manifest["exif_scene_brightness_provenance"] == "estimated").sum() == 2579
    invalid = manifest.loc[~manifest["source_exposure_valid"], "image_name"].tolist()
    assert invalid == ["03461.jpg"]


def test_every_group_belongs_to_one_split() -> None:
    manifest = pd.read_parquet(ROOT / "data" / "manifest.parquet")
    assert manifest.groupby("group_id")["split"].nunique().max() == 1
