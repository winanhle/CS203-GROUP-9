"""Scan a tar archive for exact and perceptually near-duplicate images."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def dhash(image: Image.Image, size: int = 8) -> int:
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            value = (value << 1) | (pixels[offset + col] > pixels[offset + col + 1])
    return value


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, str, dict[int, object]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, name: str) -> None:
        node = (value, name, {})
        if self.root is None:
            self.root = node
            return
        current = self.root
        while True:
            distance = self.distance(value, current[0])
            children = current[2]
            if distance not in children:
                children[distance] = node
                return
            current = children[distance]  # type: ignore[assignment]

    def query(self, value: int, radius: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        results: list[tuple[int, str]] = []
        stack = [self.root]
        while stack:
            current = stack.pop()
            distance = self.distance(value, current[0])
            if distance <= radius:
                results.append((distance, current[1]))
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in current[2].items() if low <= edge <= high)
        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    byte_hash_groups: dict[str, list[str]] = defaultdict(list)
    perceptual_hash_groups: dict[int, list[str]] = defaultdict(list)
    near_pairs: list[dict[str, object]] = []
    tree = BKTree()
    count = 0

    with tarfile.open(args.archive, "r:*") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            extracted = archive.extractfile(member)
            assert extracted is not None
            payload = extracted.read()
            byte_hash_groups[hashlib.sha256(payload).hexdigest()].append(member.name)
            with Image.open(__import__("io").BytesIO(payload)) as image:
                value = dhash(image)
            for distance, prior_name in tree.query(value, args.radius):
                near_pairs.append({"left": prior_name, "right": member.name, "dhash_distance": distance})
            tree.add(value, member.name)
            perceptual_hash_groups[value].append(member.name)
            count += 1

    exact_groups = [names for names in byte_hash_groups.values() if len(names) > 1]
    same_dhash_groups = [names for names in perceptual_hash_groups.values() if len(names) > 1]
    result = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(args.archive.resolve()),
        "image_count": count,
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_image_count": sum(len(group) for group in exact_groups),
        "exact_duplicate_groups": exact_groups,
        "dhash_radius": args.radius,
        "identical_dhash_group_count": len(same_dhash_groups),
        "identical_dhash_groups": same_dhash_groups,
        "near_pair_count": len(near_pairs),
        "near_pairs": near_pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"near_pairs", "identical_dhash_groups", "exact_duplicate_groups"}}, indent=2))


if __name__ == "__main__":
    main()

