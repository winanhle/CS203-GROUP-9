"""Audit a SPAQ metadata table against an image tar archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.metadata)
    key = "Image name"
    if key not in frame.columns:
        raise ValueError(f"Missing required key column {key!r}")

    with tarfile.open(args.archive, "r:*") as tar:
        archive_names = [Path(m.name).name for m in tar if m.isfile()]

    label_names = frame[key].astype(str).tolist()
    archive_set = set(archive_names)
    label_set = set(label_names)
    split_columns = [name for name in frame if "split" in name]
    exif_columns = [
        "Focal length",
        "F-number",
        "Exposure time",
        "ISO",
        "Brightness.1",
        "Flash",
        "Time0",
        "Time1",
    ]
    attribute_columns = ["Brightness", "Colorfulness", "Contrast", "Noisiness", "Sharpness"]
    scene_columns = [
        "Animal",
        "Cityscape",
        "Human",
        "Indoor scene",
        "Landscape",
        "Night scene",
        "Plant",
        "Still-life",
        "Others",
    ]

    numerical = ["MOS", *attribute_columns, *scene_columns, *exif_columns]
    summary = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(args.metadata.resolve()),
        "metadata_bytes": args.metadata.stat().st_size,
        "metadata_sha256": sha256_file(args.metadata),
        "archive_path": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256_file(args.archive),
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "columns": frame.columns.tolist(),
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "duplicate_label_keys": int(frame[key].duplicated(keep=False).sum()),
        "unique_label_keys": len(label_set),
        "archive_file_count": len(archive_names),
        "unique_archive_filenames": len(archive_set),
        "matched_key_count": len(label_set & archive_set),
        "labels_without_image_count": len(label_set - archive_set),
        "images_without_label_count": len(archive_set - label_set),
        "labels_without_image_examples": sorted(label_set - archive_set)[:20],
        "images_without_label_examples": sorted(archive_set - label_set)[:20],
        "null_counts": {name: int(frame[name].isna().sum()) for name in frame.columns},
        "numeric_min": {name: scalar(frame[name].min()) for name in numerical},
        "numeric_max": {name: scalar(frame[name].max()) for name in numerical},
        "numeric_unique_counts": {name: int(frame[name].nunique(dropna=True)) for name in numerical},
        "exif_columns": exif_columns,
        "attribute_columns": attribute_columns,
        "scene_columns": scene_columns,
        "split_columns": split_columns,
        "split_value_counts": {
            name: {str(k): int(v) for k, v in frame[name].value_counts(dropna=False).items()}
            for name in split_columns
        },
        "filename_min": min(label_names),
        "filename_max": max(label_names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

