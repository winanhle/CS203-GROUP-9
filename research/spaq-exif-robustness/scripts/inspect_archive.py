"""Inspect SPAQ tar archives without extracting them.

The command records member names, extension counts, compressed member sizes,
and (optionally) verifies every image through Pillow while checking embedded
EXIF. It never writes archive contents to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path, verify_images: bool) -> dict[str, object]:
    extensions: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    top_levels: Counter[str] = Counter()
    filenames: list[str] = []
    image_count = 0
    image_bytes = 0
    readable_images = 0
    unreadable_images: list[dict[str, str]] = []
    images_with_exif = 0

    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            top_levels[member.name.split("/", 1)[0]] += 1
            if not member.isfile():
                continue
            filenames.append(member.name)
            suffix = Path(member.name).suffix.lower() or "<none>"
            extensions[suffix] += 1
            image_bytes += member.size
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            image_count += 1
            if not verify_images:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                unreadable_images.append({"name": member.name, "error": "extractfile returned None"})
                continue
            try:
                with Image.open(extracted) as image:
                    image.verify()
                extracted = archive.extractfile(member)
                assert extracted is not None
                with Image.open(extracted) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
                    if len(image.getexif()) > 0:
                        images_with_exif += 1
                readable_images += 1
            except Exception as exc:  # evidence should retain every failure
                unreadable_images.append({"name": member.name, "error": repr(exc)})

    stems = [Path(name).stem for name in filenames]
    return {
        "archive_path": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "inspected_utc": datetime.now(timezone.utc).isoformat(),
        "verify_images": verify_images,
        "file_count": len(filenames),
        "image_count": image_count,
        "total_member_file_bytes": image_bytes,
        "extension_counts": dict(sorted(extensions.items())),
        "top_level_member_counts": dict(sorted(top_levels.items())),
        "unique_filename_count": len(set(filenames)),
        "unique_stem_count": len(set(stems)),
        "first_members": filenames[:20],
        "last_members": filenames[-20:],
        "readable_images": readable_images if verify_images else None,
        "unreadable_images": unreadable_images if verify_images else None,
        "dimension_counts": dict(dimensions.most_common()) if verify_images else None,
        "images_with_embedded_exif": images_with_exif if verify_images else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.archive, args.verify_images)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

