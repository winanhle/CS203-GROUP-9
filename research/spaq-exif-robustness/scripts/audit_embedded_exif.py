"""Audit embedded EXIF in all original SPAQ JPEGs against author workbooks."""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ExifTags, Image


EXIF_IFD = 0x8769


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value).replace("\x00", "").strip() or None


def finite_float(value: object) -> float | None:
    try:
        result = float(value)  # Handles Pillow IFDRational.
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) else None


def approximately(left: float | None, right: object, tolerance: float = 1e-5) -> bool | None:
    if left is None:
        return None
    right_float = finite_float(right)
    return right_float is not None and abs(left - right_float) <= tolerance * max(1.0, abs(left))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--exif-workbook", type=Path, required=True)
    parser.add_argument("--row-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_excel(args.exif_workbook, sheet_name="Source EXIF tags").set_index("Image name")
    apex = pd.read_excel(args.exif_workbook, sheet_name="APEX EXIF tags").set_index("Image name")
    rows = []
    with tarfile.open(args.archive, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {member.name}")
            data = extracted.read()
            with Image.open(io.BytesIO(data)) as image:
                root = image.getexif()
                nested = root.get_ifd(EXIF_IFD)
                root_named = {ExifTags.TAGS.get(key, key): value for key, value in root.items()}
                nested_named = {ExifTags.TAGS.get(key, key): value for key, value in nested.items()}
            image_name = Path(member.name).name
            if image_name not in source.index or image_name not in apex.index:
                raise AssertionError(f"Embedded image has no workbook row: {image_name}")
            source_row, apex_row = source.loc[image_name], apex.loc[image_name]
            exposure = finite_float(nested_named.get("ExposureTime"))
            brightness = finite_float(nested_named.get("BrightnessValue"))
            date_time = clean_text(nested_named.get("DateTimeOriginal") or root_named.get("DateTime"))
            flash_raw = finite_float(nested_named.get("Flash"))
            flash_fired = int(flash_raw) & 1 if flash_raw is not None else None
            rows.append(
                {
                    "image_name": image_name,
                    "make": clean_text(root_named.get("Make")),
                    "model": clean_text(root_named.get("Model")),
                    "device_id": "|".join(
                        value or "UNKNOWN"
                        for value in (clean_text(root_named.get("Make")), clean_text(root_named.get("Model")))
                    ),
                    "date_time": date_time,
                    "orientation": finite_float(root_named.get("Orientation")),
                    "embedded_focal_length": finite_float(nested_named.get("FocalLength")),
                    "embedded_f_number": finite_float(nested_named.get("FNumber")),
                    "embedded_exposure_time": exposure,
                    "embedded_iso": finite_float(
                        nested_named.get("PhotographicSensitivity", nested_named.get("ISOSpeedRatings"))
                    ),
                    "embedded_brightness_value": brightness,
                    "embedded_flash_raw": flash_raw,
                    "embedded_flash_fired": flash_fired,
                    "embedded_shutter_speed_value": finite_float(nested_named.get("ShutterSpeedValue")),
                    "embedded_aperture_value": finite_float(nested_named.get("ApertureValue")),
                    "source_focal_matches": approximately(
                        finite_float(nested_named.get("FocalLength")), source_row["Focal length"]
                    ),
                    "source_f_number_matches": approximately(
                        finite_float(nested_named.get("FNumber")), source_row["F-number"]
                    ),
                    "source_exposure_matches": approximately(exposure, source_row["Exposure time"]),
                    "source_iso_matches": approximately(
                        finite_float(nested_named.get("PhotographicSensitivity", nested_named.get("ISOSpeedRatings"))),
                        source_row["ISO"],
                    ),
                    "source_brightness_matches_when_observed": approximately(brightness, source_row["Brightness"]),
                    "source_flash_matches": (
                        bool(flash_fired == int(source_row["Flash"])) if flash_fired is not None else None
                    ),
                    # The workbook's APEX column labels are retained verbatim here. Paper text and
                    # observed values show that its "Sensitivity vaue" column carries observed or
                    # estimated scene brightness, while "Brightness value" is ISO sensitivity APEX.
                    "apex_sensitivity_vaue": finite_float(apex_row["Sensitivity vaue"]),
                    "apex_brightness_value": finite_float(apex_row["Brightness value"]),
                    "apex_brightness_provenance": "observed" if brightness is not None else "estimated",
                }
            )

    frame = pd.DataFrame(rows).sort_values("image_name", ignore_index=True)
    if len(frame) != 11125 or frame["image_name"].nunique() != 11125:
        raise AssertionError("Embedded EXIF audit did not cover 11,125 unique images")
    args.row_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.row_output, index=False, lineterminator="\n")

    comparison_columns = [column for column in frame if column.endswith("_matches") or "matches_when" in column]
    comparison_summary = {}
    for column in comparison_columns:
        values = frame[column]
        comparison_summary[column] = {
            "compared": int(values.notna().sum()),
            "matched": int((values == True).sum()),  # noqa: E712 - explicit nullable comparison
            "mismatched": int((values == False).sum()),  # noqa: E712
            "unavailable_embedded": int(values.isna().sum()),
        }
    device_counts = frame["device_id"].value_counts()
    invalid_exposure_rows = source.index[
        ~np.isfinite(pd.to_numeric(source["Exposure time"], errors="coerce"))
        | (pd.to_numeric(source["Exposure time"], errors="coerce") <= 0)
        | (pd.to_numeric(source["Exposure time"], errors="coerce") > 60)
    ].tolist()
    result = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "row_output": str(args.row_output.resolve()),
        "unique_device_id_count": int(frame["device_id"].nunique()),
        "unknown_device_row_count": int(frame["device_id"].str.contains("UNKNOWN").sum()),
        "device_row_counts": {str(key): int(value) for key, value in device_counts.items()},
        "embedded_field_presence": {
            column: int(frame[column].notna().sum())
            for column in [
                "embedded_focal_length",
                "embedded_f_number",
                "embedded_exposure_time",
                "embedded_iso",
                "embedded_brightness_value",
                "embedded_flash_raw",
                "date_time",
            ]
        },
        "comparison_summary": comparison_summary,
        "apex_brightness_provenance_counts": dict(
            Counter(frame["apex_brightness_provenance"].astype(str))
        ),
        "source_exposure_values_outside_0_to_60_seconds": invalid_exposure_rows,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
