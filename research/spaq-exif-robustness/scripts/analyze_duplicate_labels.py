"""Measure label consistency and split leakage for duplicate SPAQ images."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LABEL_COLUMNS = [
    "MOS",
    "Brightness",
    "Colorfulness",
    "Contrast",
    "Noisiness",
    "Sharpness",
    "Animal",
    "Cityscape",
    "Human",
    "Indoor scene",
    "Landscape",
    "Night scene",
    "Plant",
    "Still-life",
    "Others",
    "Focal length",
    "F-number",
    "Exposure time",
    "ISO",
    "Brightness.1",
    "Flash",
    "Time0",
    "Time1",
]


def filename(value: str) -> str:
    return Path(value).name


def differences(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        result[column] = float(numeric.max() - numeric.min())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    duplicate_sha = hashlib.sha256(args.duplicates.read_bytes()).hexdigest()
    metadata_sha = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    audit = json.loads(args.duplicates.read_text(encoding="utf-8"))
    metadata = pd.read_csv(args.metadata).set_index("Image name")
    if not metadata.index.is_unique:
        raise ValueError("Metadata image identifiers are not unique")

    group_records = []
    conflict_counts = {column: 0 for column in LABEL_COLUMNS}
    qalign_cross_split = 0
    legacy_cross_split = {f"ratio802_seed123_split_{i:02d}": 0 for i in range(1, 11)}

    for raw_group in audit["exact_duplicate_groups"]:
        names = [filename(item) for item in raw_group]
        group = metadata.loc[names]
        deltas = differences(group, LABEL_COLUMNS)
        differing = [column for column, delta in deltas.items() if delta != 0]
        for column in differing:
            conflict_counts[column] += 1
        qalign_values = sorted(group["qalign_split"].astype(str).unique().tolist())
        if len(qalign_values) > 1:
            qalign_cross_split += 1
        legacy_conflicts = []
        for column in legacy_cross_split:
            values = sorted(group[column].astype(str).unique().tolist())
            if len(values) > 1:
                legacy_cross_split[column] += 1
                legacy_conflicts.append(column)
        group_records.append(
            {
                "images": names,
                "differing_columns": differing,
                "label_ranges": {key: value for key, value in deltas.items() if value != 0},
                "qalign_splits": qalign_values,
                "legacy_cross_split_columns": legacy_conflicts,
                "rows": group.reset_index()[["Image name", *LABEL_COLUMNS, "qalign_split"]].to_dict(
                    orient="records"
                ),
            }
        )

    near_nonexact = []
    exact_pairs = {
        frozenset(filename(item) for item in group)
        for group in audit["exact_duplicate_groups"]
        if len(group) == 2
    }
    for pair in audit["near_pairs"]:
        names = [filename(pair["left"]), filename(pair["right"])]
        if frozenset(names) in exact_pairs:
            continue
        group = metadata.loc[names]
        deltas = differences(group, LABEL_COLUMNS)
        near_nonexact.append(
            {
                "images": names,
                "dhash_distance": pair["dhash_distance"],
                "mos_absolute_difference": deltas["MOS"],
                "differing_columns": [key for key, value in deltas.items() if value != 0],
            }
        )

    result = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "duplicate_audit": str(args.duplicates.resolve()),
            "duplicate_audit_sha256": duplicate_sha,
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": metadata_sha,
        },
        "exact_duplicate_group_count": len(group_records),
        "exact_duplicate_groups_with_any_label_conflict": sum(
            bool(record["differing_columns"]) for record in group_records
        ),
        "conflicting_group_count_by_column": conflict_counts,
        "qalign_cross_split_group_count": qalign_cross_split,
        "legacy_cross_split_group_count_by_column": legacy_cross_split,
        "exact_duplicate_groups": group_records,
        "nonexact_near_pair_count": len(near_nonexact),
        "nonexact_near_pairs": near_nonexact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"exact_duplicate_groups", "nonexact_near_pairs"}}, indent=2))


if __name__ == "__main__":
    main()
