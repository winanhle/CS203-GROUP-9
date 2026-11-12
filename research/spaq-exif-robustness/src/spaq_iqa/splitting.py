"""Deterministic, group-safe SPAQ split construction."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


@dataclass(frozen=True)
class Group:
    group_id: str
    images: tuple[str, ...]
    mean_mos: float


def canonical_name(value: str) -> str:
    return Path(value).name


def build_groups(metadata: pd.DataFrame, edges: list[tuple[str, str]]) -> list[Group]:
    names = metadata["Image name"].astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError("Image names must be unique")
    known = set(names)
    union_find = UnionFind(names)
    for raw_left, raw_right in edges:
        left, right = canonical_name(raw_left), canonical_name(raw_right)
        if left not in known or right not in known:
            raise ValueError(f"Unknown image in duplicate edge: {left}, {right}")
        union_find.union(left, right)

    members: dict[str, list[str]] = defaultdict(list)
    for name in names:
        members[union_find.find(name)].append(name)
    mos = metadata.set_index("Image name")["MOS"]
    groups = []
    for images in members.values():
        ordered = tuple(sorted(images))
        digest = hashlib.sha256("\n".join(ordered).encode()).hexdigest()[:16]
        groups.append(Group(f"grp_{digest}", ordered, float(mos.loc[list(ordered)].mean())))
    return sorted(groups, key=lambda group: group.group_id)


def assign_groups(
    groups: list[Group], seed: int, ratios: dict[str, float] | None = None
) -> dict[str, str]:
    ratios = ratios or SPLIT_RATIOS
    if not groups or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Groups must be nonempty and split ratios must sum to one")

    frame = pd.DataFrame(
        {"group_id": [g.group_id for g in groups], "mean_mos": [g.mean_mos for g in groups]}
    )
    frame["stratum"] = pd.qcut(frame["mean_mos"], q=10, labels=False, duplicates="drop")
    group_lookup = {group.group_id: group for group in groups}
    assignments: dict[str, str] = {}

    for stratum, stratum_frame in frame.groupby("stratum", sort=True):
        group_ids = stratum_frame["group_id"].tolist()
        random.Random(f"{seed}:{stratum}").shuffle(group_ids)
        total_rows = sum(len(group_lookup[group_id].images) for group_id in group_ids)
        target = {split: total_rows * ratio for split, ratio in ratios.items()}
        current = {split: 0 for split in ratios}
        for group_id in group_ids:
            # Largest remaining row deficit wins; stable split-name tie break.
            split = max(sorted(ratios), key=lambda name: target[name] - current[name])
            assignments[group_id] = split
            current[split] += len(group_lookup[group_id].images)
    return assignments


def materialize_manifest(groups: list[Group], assignments: dict[str, str]) -> pd.DataFrame:
    rows = [
        {"image_name": image, "split": assignments[group.group_id], "group_id": group.group_id}
        for group in groups
        for image in group.images
    ]
    return pd.DataFrame(rows).sort_values("image_name", ignore_index=True)


def assert_group_safe(manifest: pd.DataFrame, edges: list[tuple[str, str]]) -> None:
    if manifest["image_name"].duplicated().any():
        raise AssertionError("Manifest has duplicate image names")
    indexed = manifest.set_index("image_name")
    for raw_left, raw_right in edges:
        left, right = canonical_name(raw_left), canonical_name(raw_right)
        if indexed.at[left, "split"] != indexed.at[right, "split"]:
            raise AssertionError(f"Candidate pair crosses splits: {left}, {right}")
        if indexed.at[left, "group_id"] != indexed.at[right, "group_id"]:
            raise AssertionError(f"Candidate pair has inconsistent groups: {left}, {right}")
