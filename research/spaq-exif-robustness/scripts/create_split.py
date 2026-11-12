"""Create a deterministic 70/10/20 SPAQ manifest with perceptual groups intact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from spaq_iqa.splitting import assert_group_safe, assign_groups, build_groups, materialize_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--duplicate-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    duplicate_audit = json.loads(args.duplicate_audit.read_text(encoding="utf-8"))
    edges = [(pair["left"], pair["right"]) for pair in duplicate_audit["near_pairs"]]
    groups = build_groups(metadata, edges)
    assignments = assign_groups(groups, args.seed)
    manifest = materialize_manifest(groups, assignments)
    assert_group_safe(manifest, edges)
    if set(manifest["image_name"]) != set(metadata["Image name"]):
        raise AssertionError("Manifest coverage differs from metadata")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False, lineterminator="\n")
    merged = manifest.merge(
        metadata[["Image name", "MOS"]], left_on="image_name", right_on="Image name", validate="1:1"
    )
    split_summary = {}
    for split, frame in merged.groupby("split", sort=True):
        split_summary[split] = {
            "rows": int(len(frame)),
            "groups": int(frame["group_id"].nunique()),
            "mos_min": float(frame["MOS"].min()),
            "mos_max": float(frame["MOS"].max()),
            "mos_mean": float(frame["MOS"].mean()),
            "mos_std": float(frame["MOS"].std()),
        }
    grouped_sizes = manifest.groupby("group_id").size()
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "ratios": {"train": 0.70, "val": 0.10, "test": 0.20},
        "metadata_sha256": sha256(args.metadata),
        "duplicate_audit_sha256": sha256(args.duplicate_audit),
        "manifest_sha256": sha256(args.manifest),
        "row_count": int(len(manifest)),
        "group_count": int(manifest["group_id"].nunique()),
        "multi_image_group_count": int((grouped_sizes > 1).sum()),
        "largest_group_size": int(grouped_sizes.max()),
        "candidate_edge_count": len(edges),
        "candidate_cross_split_count": 0,
        "split_summary": split_summary,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
