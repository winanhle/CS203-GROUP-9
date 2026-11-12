"""Compare selected original SPAQ images with Q-Align's re-encoded variant."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    result = 0
    for row in range(8):
        for column in range(8):
            result = (result << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return result


def properties(data: bytes) -> tuple[dict, int]:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        return (
            {
                "bytes": len(data),
                "sha256": sha256(data),
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "embedded_exif_entries": len(image.getexif()),
            },
            dhash(image),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qalign-archive", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    with tarfile.open(args.qalign_archive, mode="r:") as archive:
        for image_id in args.ids:
            image_name = f"{image_id}.jpg"
            original_path = args.original_dir / f"SPAQ__TestImage__{image_name}"
            original_data = original_path.read_bytes()
            qalign_member = archive.getmember(f"spaq/{image_name}")
            extracted = archive.extractfile(qalign_member)
            if extracted is None:
                raise RuntimeError(f"Could not read {qalign_member.name}")
            qalign_data = extracted.read()
            original, original_hash = properties(original_data)
            qalign, qalign_hash = properties(qalign_data)
            records.append(
                {
                    "image_name": image_name,
                    "original": original,
                    "qalign": qalign,
                    "byte_identical": original_data == qalign_data,
                    "dhash_distance": (original_hash ^ qalign_hash).bit_count(),
                }
            )
    result = {
        "compared_utc": datetime.now(timezone.utc).isoformat(),
        "qalign_archive": str(args.qalign_archive.resolve()),
        "original_dir": str(args.original_dir.resolve()),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
