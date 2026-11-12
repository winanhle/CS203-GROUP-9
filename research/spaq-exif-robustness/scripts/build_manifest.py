"""Build the canonical SPAQ row contract from author workbooks and audited assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ATTRIBUTE_RENAMES = {
    "Brightness": "attribute_brightness",
    "Colorfulness": "attribute_colorfulness",
    "Contrast": "attribute_contrast",
    "Noisiness": "attribute_noisiness",
    "Sharpness": "attribute_sharpness",
}
SCENE_RENAMES = {
    "Animal": "scene_animal",
    "Cityscape": "scene_cityscape",
    "Human": "scene_human",
    "Indoor scene": "scene_indoor",
    "Landscape": "scene_landscape",
    "Night scene": "scene_night",
    "Plant": "scene_plant",
    "Still-life": "scene_still_life",
    "Others": "scene_others",
}
SOURCE_RENAMES = {
    "Focal length": "source_focal_length",
    "F-number": "source_f_number",
    "Exposure time": "source_exposure_time",
    "ISO": "source_iso",
    "Brightness": "source_brightness_value",
    "Flash": "source_flash",
    "Time": "source_time_sin",
    "Unnamed: 8": "source_time_cos",
}
# The workbook labels the fourth and fifth APEX fields in reverse semantic order.
# Values and the paper's exposure equation show that `Sensitivity vaue` is scene
# brightness and `Brightness value` is ISO sensitivity. Preserve this correction
# explicitly rather than propagating misleading source headers.
APEX_RENAMES = {
    "Focal length": "exif_focal_length",
    "Aperture value": "exif_aperture_value",
    "Time value": "exif_shutter_time_value",
    "Sensitivity vaue": "exif_scene_brightness_value",
    "Brightness value": "exif_sensitivity_value",
    "Flash": "exif_flash",
    "Time": "exif_time_sin",
    "Unnamed: 8": "exif_time_cos",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mos", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--exif", type=Path, required=True)
    parser.add_argument("--embedded-audit", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    mos = pd.read_excel(args.mos).rename(columns={"Image name": "image_name", **ATTRIBUTE_RENAMES})
    scene = pd.read_excel(args.scene).rename(columns={"Image name": "image_name", **SCENE_RENAMES})
    source = pd.read_excel(args.exif, sheet_name="Source EXIF tags").rename(
        columns={"Image name": "image_name", **SOURCE_RENAMES}
    )
    apex = pd.read_excel(args.exif, sheet_name="APEX EXIF tags").rename(
        columns={"Image name": "image_name", **APEX_RENAMES}
    )
    embedded = pd.read_csv(args.embedded_audit)
    split = pd.read_csv(args.split)
    frames = [mos, scene, source, apex, embedded, split]
    for frame in frames:
        if len(frame) != 11125 or frame["image_name"].nunique() != 11125:
            raise AssertionError("Every contract input must contain 11,125 unique image identifiers")

    manifest = mos
    for frame in (scene, source, apex, embedded, split):
        manifest = manifest.merge(frame, on="image_name", how="inner", validate="1:1")
    if len(manifest) != 11125:
        raise AssertionError("Canonical manifest join lost rows")
    manifest.insert(1, "qalign_member_path", "spaq/" + manifest["image_name"])
    manifest.insert(2, "original_member_path", "SPAQ/TestImage/" + manifest["image_name"])
    manifest["exif_scene_brightness_provenance"] = manifest["apex_brightness_provenance"]
    manifest["mask_focal_observed"] = manifest["embedded_focal_length"].notna()
    manifest["mask_aperture_observed"] = manifest["embedded_f_number"].notna()
    manifest["mask_shutter_observed"] = manifest["embedded_shutter_speed_value"].notna() | manifest[
        "embedded_exposure_time"
    ].notna()
    manifest["mask_sensitivity_observed"] = manifest["embedded_iso"].notna()
    manifest["mask_scene_brightness_observed"] = manifest["embedded_brightness_value"].notna()
    manifest["mask_flash_observed"] = manifest["embedded_flash_raw"].notna()
    manifest["mask_time_observed"] = manifest["date_time"].notna()
    manifest["source_exposure_observed"] = manifest["embedded_exposure_time"].notna()
    manifest["source_exposure_valid"] = manifest["source_exposure_time"].between(
        0, 60, inclusive="neither"
    )
    manifest = manifest.sort_values("image_name", ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output, index=False, compression="zstd")
    split_counts = manifest["split"].value_counts().to_dict()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.output.resolve()),
        "manifest_sha256": sha256(args.output),
        "rows": int(len(manifest)),
        "columns": int(len(manifest.columns)),
        "unique_image_names": int(manifest["image_name"].nunique()),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "group_count": int(manifest["group_id"].nunique()),
        "mos_range": [float(manifest["MOS"].min()), float(manifest["MOS"].max())],
        "apex_feature_null_counts": {
            column: int(manifest[column].isna().sum()) for column in APEX_RENAMES.values()
        },
        "observed_mask_counts": {
            column: int(manifest[column].sum()) for column in manifest if column.startswith("mask_")
        },
        "estimated_scene_brightness_rows": int(
            (manifest["exif_scene_brightness_provenance"] == "estimated").sum()
        ),
        "invalid_source_exposure_rows": manifest.loc[
            ~manifest["source_exposure_valid"], "image_name"
        ].tolist(),
        "input_sha256": {
            "mos_attributes_workbook": sha256(args.mos),
            "scene_workbook": sha256(args.scene),
            "exif_workbook": sha256(args.exif),
            "embedded_audit_csv": sha256(args.embedded_audit),
            "split_csv": sha256(args.split),
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
