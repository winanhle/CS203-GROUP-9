"""Generate and verify the contiguous center-box mask and LSUN Cat manifests."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK_PATH = ROOT / "exp" / "inpainting_masks" / "center_box_mask.npy"
DEFAULT_MANIFEST_PATH = ROOT / "configs" / "lsun_cat_57_manifest.json"
DEFAULT_DATASET_DIR = ROOT / "exp" / "datasets" / "lsun_cat"

SMOKE_8_FILENAMES = (
    "109.png",
    "133.png",
    "14.png",
    "169.png",
    "240.png",
    "245.png",
    "86.png",
    "92.png",
)
SMOKE_8_INDICES_IN_57 = [3, 7, 8, 11, 21, 22, 54, 56]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest using 8 MiB chunked reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_center_box_mask(image_size: int = 256, box_size: int = 128) -> np.ndarray:
    """Build a contiguous center-box mask (0.0 = missing hole, 1.0 = observed)."""
    if image_size <= 0 or box_size <= 0 or box_size >= image_size:
        raise ValueError(
            f"Require 0 < box_size < image_size; got box_size={box_size}, image_size={image_size}"
        )
    if (image_size - box_size) % 2 != 0:
        raise ValueError(
            f"(image_size - box_size) must be even for exact centering; got {image_size} - {box_size}"
        )
    start = (image_size - box_size) // 2
    end = start + box_size
    mask = np.ones((image_size, image_size), dtype=np.float32)
    mask[start:end, start:end] = 0.0
    return np.ascontiguousarray(mask, dtype=np.float32)


def verify_center_box_mask(
    mask_path: Path = DEFAULT_MASK_PATH,
    image_size: int = 256,
    box_size: int = 128,
) -> dict[str, Any]:
    """Validate an existing center_box_mask.npy against exact geometry and dtype."""
    if not mask_path.is_file():
        raise FileNotFoundError(f"Center box mask not found: {mask_path}")
    loaded = np.load(mask_path, allow_pickle=False)
    expected = build_center_box_mask(image_size=image_size, box_size=box_size)
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"Expected ndarray in {mask_path}, got {type(loaded).__name__}")
    if loaded.shape != expected.shape:
        raise ValueError(f"Mask shape mismatch in {mask_path}: expected {expected.shape}, got {loaded.shape}")
    if loaded.dtype != np.dtype(np.float32):
        raise ValueError(f"Mask dtype mismatch in {mask_path}: expected float32, got {loaded.dtype}")
    if not loaded.flags["C_CONTIGUOUS"]:
        raise ValueError(f"Mask in {mask_path} is not C-contiguous")
    if not np.isfinite(loaded).all():
        raise ValueError(f"Mask in {mask_path} contains non-finite values")
    if not np.array_equal(loaded, expected):
        raise ValueError(f"Mask values in {mask_path} do not match expected center box geometry")

    zero_count = int(np.sum(loaded == 0))
    nonzero_count = int(np.sum(loaded != 0))
    return {
        "path": str(mask_path),
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
        "c_contiguous": bool(loaded.flags["C_CONTIGUOUS"]),
        "zero_count": zero_count,
        "nonzero_count": nonzero_count,
        "missing_fraction": float(zero_count / loaded.size),
        "sha256": sha256_file(mask_path),
    }


def ensure_center_box_mask(
    mask_path: Path = DEFAULT_MASK_PATH,
    image_size: int = 256,
    box_size: int = 128,
    check_only: bool = False,
) -> dict[str, Any]:
    """Verify or idempotently create center_box_mask.npy atomically."""
    if check_only:
        return verify_center_box_mask(mask_path=mask_path, image_size=image_size, box_size=box_size)

    if mask_path.is_file():
        try:
            return verify_center_box_mask(mask_path=mask_path, image_size=image_size, box_size=box_size)
        except ValueError:
            pass

    mask = build_center_box_mask(image_size=image_size, box_size=box_size)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = mask_path.with_name(f"{mask_path.name}.tmp")
    with tmp_path.open("wb") as stream:
        np.save(stream, mask, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, mask_path)
    return verify_center_box_mask(mask_path=mask_path, image_size=image_size, box_size=box_size)


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    check_imagefolder: bool = True,
) -> dict[str, Any]:
    """Validate frozen LSUN Cat manifest against on-disk PNGs and ImageFolder ordering."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required_top = ("schema_version", "subset_name", "expected_count", "sort_order", "image_spec", "inputs")
    for key in required_top:
        if key not in manifest:
            raise ValueError(f"Manifest {manifest_path} missing required top-level key: {key}")

    inputs = manifest["inputs"]
    expected_count = int(manifest["expected_count"])
    if not isinstance(inputs, list) or len(inputs) != expected_count:
        raise ValueError(
            f"Manifest input count mismatch: expected {expected_count}, got {len(inputs) if isinstance(inputs, list) else type(inputs).__name__}"
        )

    image_spec = manifest["image_spec"]
    spec_width = int(image_spec["width"])
    spec_height = int(image_spec["height"])
    spec_mode = str(image_spec["mode"])
    spec_format = str(image_spec["format"])

    cat_dir = dataset_dir / "cat"
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"Dataset class directory not found: {cat_dir}")
    disk_pngs = sorted(cat_dir.glob("*.png"), key=lambda p: p.name)
    if len(disk_pngs) != expected_count:
        raise ValueError(f"Expected {expected_count} PNGs in {cat_dir}, found {len(disk_pngs)}")

    seen_hashes: set[str] = set()
    smoke_indices: list[int] = []
    smoke_inputs: list[str] = []

    for expected_idx, (entry, disk_path) in enumerate(zip(inputs, disk_pngs)):
        if entry.get("index") != expected_idx:
            raise ValueError(f"Manifest index mismatch at position {expected_idx}: got {entry.get('index')}")
        rel_posix = f"cat/{disk_path.name}"
        if entry.get("input") != rel_posix:
            raise ValueError(f"Manifest order mismatch at index {expected_idx}: expected {entry.get('input')}, disk has {rel_posix}")
        if entry.get("filename") is not None and entry.get("filename") != disk_path.name:
            raise ValueError(f"Manifest filename mismatch at index {expected_idx}: {entry.get('filename')} vs {disk_path.name}")

        if "bytes" in entry:
            actual_bytes = disk_path.stat().st_size
            if actual_bytes != int(entry["bytes"]):
                raise ValueError(f"Byte size mismatch for {rel_posix}: expected {entry['bytes']}, got {actual_bytes}")

        with Image.open(disk_path) as img:
            img_format = img.format
            img_size = img.size
            img_mode = img.mode
            img.load()
        if img_size != (spec_width, spec_height):
            raise ValueError(f"Image dimensions mismatch for {rel_posix}: expected {(spec_width, spec_height)}, got {img_size}")
        if img_mode != spec_mode:
            raise ValueError(f"Image mode mismatch for {rel_posix}: expected {spec_mode}, got {img_mode}")
        if img_format != spec_format:
            raise ValueError(f"Image format mismatch for {rel_posix}: expected {spec_format}, got {img_format}")

        actual_sha = sha256_file(disk_path)
        if actual_sha != entry.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {rel_posix}: expected {entry.get('sha256')}, got {actual_sha}")
        if actual_sha in seen_hashes:
            raise ValueError(f"Duplicate SHA-256 detected in manifest: {actual_sha} ({rel_posix})")
        seen_hashes.add(actual_sha)

        if "repo_tracked" in str(entry.get("source", "")):
            smoke_indices.append(expected_idx)
            smoke_inputs.append(rel_posix)

    if check_imagefolder:
        try:
            import torchvision.datasets
        except ImportError:
            torchvision = None
        if torchvision is not None:
            ds = torchvision.datasets.ImageFolder(str(dataset_dir))
            if len(ds.samples) != len(inputs):
                raise ValueError(f"ImageFolder length {len(ds.samples)} != manifest length {len(inputs)}")
            for idx, ((sample_path, _), entry) in enumerate(zip(ds.samples, inputs)):
                rel = Path(sample_path).resolve().relative_to(dataset_dir.resolve()).as_posix()
                if rel != entry["input"]:
                    raise ValueError(f"ImageFolder order mismatch at {idx}: {rel} != {entry['input']}")

    return {
        "manifest_path": str(manifest_path),
        "dataset_dir": str(dataset_dir),
        "count": len(inputs),
        "unique_hashes": len(seen_hashes),
        "manifest_sha256": sha256_file(manifest_path),
        "smoke_indices": smoke_indices,
        "smoke_inputs": smoke_inputs,
    }


def get_smoke_subset(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Return the deterministic 8-image repo-tracked smoke subset from the 57-image manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs", [])

    selected = [
        entry for entry in inputs
        if "repo_tracked" in str(entry.get("source", ""))
        or Path(str(entry.get("input", ""))).name in SMOKE_8_FILENAMES
    ]
    selected.sort(key=lambda e: int(e["index"]))

    filenames = tuple(Path(str(e["input"])).name for e in selected)
    if filenames != SMOKE_8_FILENAMES:
        raise ValueError(f"Smoke filenames mismatch: expected {SMOKE_8_FILENAMES}, got {filenames}")

    sample_ids_in_57 = [int(e["index"]) for e in selected]
    if sample_ids_in_57 != SMOKE_8_INDICES_IN_57:
        raise ValueError(
            f"Smoke indices in 57-image manifest mismatch: expected {SMOKE_8_INDICES_IN_57}, got {sample_ids_in_57}"
        )

    smoke_inputs: list[dict[str, Any]] = []
    for smoke_idx, entry in enumerate(selected):
        copied = dict(entry)
        copied["index_in_57"] = int(entry["index"])
        copied["index"] = smoke_idx
        smoke_inputs.append(copied)

    return {
        "schema_version": str(manifest.get("schema_version", "1.0")),
        "subset_name": "lsun_cat_8_smoke_repo_tracked",
        "parent_manifest": "configs/lsun_cat_57_manifest.json",
        "dataset_root": manifest.get("dataset_root", "exp/datasets/lsun_cat"),
        "expected_count": len(smoke_inputs),
        "sort_order": manifest.get("sort_order", "lexicographical_relative_posix_path"),
        "image_spec": manifest.get("image_spec", {"width": 256, "height": 256, "mode": "RGB", "format": "PNG"}),
        "sample_ids_in_57": sample_ids_in_57,
        "inputs": smoke_inputs,
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--check", action="store_true", help="Verify mask and manifest without modifying files")
    parser.add_argument("--smoke-manifest-out", type=Path, default=None, help="Optional path to write 8-image smoke JSON")
    parser.add_argument("--print-smoke-ids", action="store_true", help="Print 8-image smoke indices and filenames")
    parser.add_argument("--no-imagefolder", action="store_true", help="Skip torchvision ImageFolder cross-check")
    args = parser.parse_args(argv)

    mask_info = ensure_center_box_mask(mask_path=args.mask_path, check_only=args.check)
    manifest_info = validate_manifest(
        manifest_path=args.manifest,
        dataset_dir=args.dataset_dir,
        check_imagefolder=not args.no_imagefolder,
    )
    smoke = get_smoke_subset(manifest_path=args.manifest)

    if args.smoke_manifest_out is not None:
        _write_json_atomic(args.smoke_manifest_out, smoke)

    print(
        f"Mask OK: {mask_info['path']} shape={tuple(mask_info['shape'])} "
        f"dtype={mask_info['dtype']} zeros={mask_info['zero_count']} "
        f"nonzeros={mask_info['nonzero_count']} sha256={mask_info['sha256']}"
    )
    print(
        f"Manifest OK: {manifest_info['count']} PNGs ({manifest_info['unique_hashes']} unique SHA-256) "
        f"manifest_sha256={manifest_info['manifest_sha256']}"
    )
    if args.print_smoke_ids:
        smoke_files = [entry["input"] for entry in smoke["inputs"]]
        print(f"Smoke sample_ids_in_57={smoke['sample_ids_in_57']} files={smoke_files}")
    if args.smoke_manifest_out is not None:
        print(f"Wrote smoke manifest: {args.smoke_manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
