"""Cross-check original SPAQ annotation workbooks against flattened metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workbook_record(path: Path, frame: pd.DataFrame) -> dict:
    excel = pd.ExcelFile(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "sheet_names": excel.sheet_names,
        "rows": int(len(frame)),
        "columns": [str(value) for value in frame.columns],
        "dtypes": [str(value) for value in frame.dtypes],
        "null_counts": [int(value) for value in frame.isna().sum()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mos", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--exif", type=Path, required=True)
    parser.add_argument("--flattened", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mos = pd.read_excel(args.mos)
    scene = pd.read_excel(args.scene)
    exif = pd.read_excel(args.exif)
    flattened = pd.read_csv(args.flattened)
    if not (len(mos) == len(scene) == len(exif) == len(flattened)):
        raise AssertionError("Annotation row counts do not agree")
    id_columns = [frame.iloc[:, 0].astype(str) for frame in (mos, scene, exif)]
    identifier_mismatches = {
        "mos_vs_scene": int((id_columns[0] != id_columns[1]).sum()),
        "mos_vs_exif": int((id_columns[0] != id_columns[2]).sum()),
        "mos_vs_flattened": int((id_columns[0] != flattened.iloc[:, 0].astype(str)).sum()),
    }

    reconstructed = pd.concat([mos, scene.iloc[:, 1:], exif.iloc[:, 1:]], axis=1)
    expected = flattened.iloc[:, : reconstructed.shape[1]]
    column_records = []
    total_value_mismatches = 0
    for index in range(reconstructed.shape[1]):
        raw_series = reconstructed.iloc[:, index]
        flat_series = expected.iloc[:, index]
        if index == 0:
            matches = raw_series.astype(str).to_numpy() == flat_series.astype(str).to_numpy()
            maximum_difference = None
        else:
            raw_values = pd.to_numeric(raw_series, errors="coerce").to_numpy(float)
            flat_values = pd.to_numeric(flat_series, errors="coerce").to_numpy(float)
            matches = np.isclose(raw_values, flat_values, rtol=0, atol=1e-12, equal_nan=True)
            finite = np.isfinite(raw_values) & np.isfinite(flat_values)
            maximum_difference = (
                float(np.max(np.abs(raw_values[finite] - flat_values[finite]))) if finite.any() else None
            )
        mismatch_count = int((~matches).sum())
        total_value_mismatches += mismatch_count
        column_records.append(
            {
                "position": index,
                "raw_name": str(reconstructed.columns[index]),
                "flattened_name": str(expected.columns[index]),
                "mismatch_count": mismatch_count,
                "maximum_absolute_difference": maximum_difference,
            }
        )

    outlier_rows = {}
    for image_name in ("03461.jpg", "05293.jpg", "00914.jpg"):
        locations = np.flatnonzero(id_columns[2].to_numpy() == image_name)
        outlier_rows[image_name] = (
            exif.iloc[int(locations[0])].to_dict() if len(locations) == 1 else {"match_count": len(locations)}
        )

    result = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "flattened_path": str(args.flattened.resolve()),
        "flattened_sha256": sha256(args.flattened),
        "workbooks": {
            "mos_attributes": workbook_record(args.mos, mos),
            "scene": workbook_record(args.scene, scene),
            "exif": workbook_record(args.exif, exif),
        },
        "identifier_mismatches": identifier_mismatches,
        "reconstructed_column_count": int(reconstructed.shape[1]),
        "flattened_compared_column_count": int(expected.shape[1]),
        "total_value_mismatches": total_value_mismatches,
        "column_comparison": column_records,
        "selected_exif_rows": outlier_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if any(identifier_mismatches.values()) or total_value_mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
