"""Verify downloaded Phase 7 Kaggle backbone outputs and emit tracked evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MANIFEST = "abff8b1a5f7d2e74d0c8874655fbe40d5e75de75bf61e65753c7085cee15800d"
EXPECTED_WEIGHTS = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "srcc": float(np.corrcoef(pd.Series(labels).rank(), pd.Series(predictions).rank())[0, 1]),
        "plcc": float(np.corrcoef(labels, predictions)[0, 1]),
        "mae": float(np.mean(np.abs(labels - predictions))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/kaggle_download/phase7_backbone_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/phase7_kaggle_backbone_v1.json")
    args = parser.parse_args()
    summary_path = args.input / "backbone_pair_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "SUCCESS":
        raise AssertionError("Kaggle summary is not successful")
    if summary["manifest_sha256"] != EXPECTED_MANIFEST or summary["pretrained_weights_sha256"] != EXPECTED_WEIGHTS:
        raise AssertionError("Input hash contract differs")
    if summary["split_counts"] != {"train": 7786, "val": 1111, "test": 2228}:
        raise AssertionError("Split counts differ")

    manifest = pd.read_parquet(ROOT / "data/manifest.parquet")
    expected_test = manifest.loc[manifest.split == "test", ["image_name", "MOS"]].sort_values("image_name").reset_index(drop=True)
    verified_arms = []
    for arm in summary["arms"]:
        mode = arm["mode"]
        for artifact in arm["artifacts"].values():
            if sha256(args.input / artifact["path"]) != artifact["sha256"]:
                raise AssertionError(f"{mode} artifact hash mismatch: {artifact['path']}")
        predictions = pd.read_csv(args.input / arm["artifacts"]["predictions"]["path"])
        if predictions.image_name.tolist() != expected_test.image_name.tolist():
            raise AssertionError(f"{mode} prediction identifiers/order differ from frozen test split")
        # Dataset targets are intentionally converted to float32 tensors before
        # inference, so compare to the canonical values after the same cast.
        canonical_labels = expected_test.MOS.to_numpy().astype(np.float32).astype(np.float64)
        if not np.array_equal(predictions.MOS.to_numpy(), canonical_labels):
            raise AssertionError(f"{mode} labels differ from canonical manifest")
        recomputed = metrics(predictions.MOS.to_numpy(), predictions.prediction.to_numpy())
        for name, value in recomputed.items():
            if not np.isclose(value, arm["test_metrics"][name], rtol=0, atol=1e-12):
                raise AssertionError(f"{mode} recomputed {name} differs")
        checkpoint = torch.load(args.input / arm["artifacts"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
        if checkpoint["mode"] != mode or checkpoint["seed"] != 20261109:
            raise AssertionError(f"{mode} checkpoint identity differs")
        if checkpoint["pretrained_weights_sha256"] != EXPECTED_WEIGHTS:
            raise AssertionError(f"{mode} checkpoint pretrained hash differs")
        state_keys = list(checkpoint["state"])
        allowed = all(key.startswith("head.") or (mode == "layer4_finetuned" and key.startswith("backbone.layer4.")) for key in state_keys)
        if not allowed or not any(key.startswith("head.") for key in state_keys):
            raise AssertionError(f"{mode} checkpoint contains unexpected state")
        log_lines = [json.loads(line) for line in (args.input / arm["artifacts"]["epoch_log"]["path"]).read_text().splitlines()]
        if len(log_lines) != arm["epochs_completed"]:
            raise AssertionError(f"{mode} epoch log count differs")
        verified_arms.append({
            "mode": mode,
            "prediction_rows": len(predictions),
            "prediction_ids_match_frozen_test": True,
            "labels_match_manifest": True,
            "recomputed_test_metrics": recomputed,
            "checkpoint_state_tensor_count": len(state_keys),
            "checkpoint_trainable_state_only": True,
            "epoch_log_rows": len(log_lines),
        })

    all_files = sorted(path for path in args.input.iterdir() if path.is_file())
    evidence = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTION_VERIFIED",
        "kernel": "longltrnhong/spaq-phase-7-matched-backbone-pair",
        "kernel_version": 1,
        "source_commit": "b092e64768c65de5443018f0d3e2f3a50ad7829a",
        "summary": summary,
        "verified_arms": verified_arms,
        "downloaded_artifacts": {path.name: sha256(path) for path in all_files},
        "interpretation": "Single-seed exploratory matched comparison; not a multi-seed confirmatory backbone claim.",
    }
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "verified_files": len(all_files), "verified_arms": verified_arms}, indent=2))


if __name__ == "__main__":
    main()
