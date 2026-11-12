"""Stream-inspect the author-recommended SPAQ tgz without full extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANNOTATION_SUFFIXES = {".csv", ".xls", ".xlsx", ".json", ".txt"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_leaf(name: str) -> str:
    path = Path(name)
    return "__".join(path.parts[-3:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-image-ids", nargs="*", default=["03461", "05293", "00914", "00001", "11125"]
    )
    args = parser.parse_args()

    suffix_counts: Counter[str] = Counter()
    member_type_counts: Counter[str] = Counter()
    member_names: list[str] = []
    image_names: list[str] = []
    annotation_records = []
    sample_records = []
    uncompressed_file_bytes = 0
    duplicate_names = []
    seen_names: set[str] = set()
    args.selected_output_dir.mkdir(parents=True, exist_ok=True)

    # Streaming mode bounds memory and does not extract the 34.8 GB archive.
    with tarfile.open(args.archive, mode="r|gz") as archive:
        for member in archive:
            member_names.append(member.name)
            if member.name in seen_names:
                duplicate_names.append(member.name)
            seen_names.add(member.name)
            if member.isdir():
                member_type_counts["directory"] += 1
                continue
            if not member.isfile():
                member_type_counts["other"] += 1
                continue
            member_type_counts["file"] += 1
            uncompressed_file_bytes += member.size
            suffix = Path(member.name).suffix.lower()
            suffix_counts[suffix or "<none>"] += 1
            is_image = suffix in IMAGE_SUFFIXES
            is_annotation = suffix in ANNOTATION_SUFFIXES
            if is_image:
                image_names.append(member.name)
            basename = Path(member.name).stem
            should_sample = is_image and basename in set(args.sample_image_ids)
            if not is_annotation and not should_sample:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read selected member {member.name}")
            data = extracted.read()
            target = args.selected_output_dir / safe_leaf(member.name)
            target.write_bytes(data)
            record = {
                "archive_path": member.name,
                "size": len(data),
                "sha256": digest(data),
                "selected_path": str(target.resolve()),
            }
            if is_annotation:
                annotation_records.append(record)
            else:
                sample_records.append(record)

    image_basenames = [Path(name).name for name in image_names]
    image_stems = [Path(name).stem for name in image_names]
    result = {
        "inspected_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(args.archive.resolve()),
        "archive_size": args.archive.stat().st_size,
        "member_count": len(member_names),
        "member_type_counts": dict(sorted(member_type_counts.items())),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "uncompressed_file_bytes": uncompressed_file_bytes,
        "duplicate_member_names": duplicate_names,
        "image_count": len(image_names),
        "unique_image_paths": len(set(image_names)),
        "unique_image_basenames": len(set(image_basenames)),
        "unique_image_stems": len(set(image_stems)),
        "image_name_first_20": sorted(image_names)[:20],
        "image_name_last_20": sorted(image_names)[-20:],
        "annotation_files": annotation_records,
        "selected_image_samples": sample_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
