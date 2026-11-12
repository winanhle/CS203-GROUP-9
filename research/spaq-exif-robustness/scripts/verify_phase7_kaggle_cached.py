"""Verify Kaggle cached-head multiseed training outputs."""

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
MANIFEST_SHA = "abff8b1a5f7d2e74d0c8874655fbe40d5e75de75bf61e65753c7085cee15800d"
CACHE_SHA = "b639bb6f2365f31004410af1a71a35529a485de849d70e6924f7fbb2dc3975ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(labels, predictions):
    return {
        "srcc": float(np.corrcoef(pd.Series(labels).rank(), pd.Series(predictions).rank())[0, 1]),
        "plcc": float(np.corrcoef(labels, predictions)[0, 1]),
        "mae": float(np.mean(np.abs(labels - predictions))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/kaggle_download/phase7_cached_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/phase7_kaggle_cached_v1.json")
    args = parser.parse_args()
    summary_path = args.input / "cached_multiseed_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "SUCCESS" or summary["manifest_sha256"] != MANIFEST_SHA or summary["feature_cache_sha256"] != CACHE_SHA:
        raise AssertionError("Kaggle summary/input contract failed")
    if len(summary["runs"]) != 9:
        raise AssertionError("Expected nine experiment/seed runs")
    manifest = pd.read_parquet(ROOT / "data/manifest.parquet")
    expected = manifest.loc[manifest.split == "test", ["image_name", "MOS"]].sort_values("image_name").reset_index(drop=True)
    canonical_labels = expected.MOS.to_numpy().astype(np.float32).astype(np.float64)
    verified_runs = []
    by_key = {}
    for run in summary["runs"]:
        key = (run["seed"], run["experiment"])
        if key in by_key:
            raise AssertionError(f"Duplicate run {key}")
        by_key[key] = run
        for artifact in run["artifacts"].values():
            path = args.input / artifact["path"]
            if sha256(path) != artifact["sha256"]:
                raise AssertionError(f"Artifact hash mismatch: {path}")
        frame = pd.read_csv(args.input / run["artifacts"]["predictions"]["path"])
        label_max_abs_error = float(np.max(np.abs(frame.MOS.to_numpy() - canonical_labels)))
        # CSV uses a shortest decimal representation of float32 labels; parsing
        # it back as float64 can differ by a few 1e-6 from the binary float32.
        if frame.image_name.tolist() != expected.image_name.tolist() or label_max_abs_error > 5e-6:
            raise AssertionError(f"Frozen test ID/label mismatch: {key}")
        recomputed = metrics(frame.MOS.to_numpy(), frame.prediction.to_numpy())
        for metric, value in recomputed.items():
            if not np.isclose(value, run["test_metrics"][metric], atol=2e-6, rtol=0):
                raise AssertionError(f"Recomputed metric mismatch: {key} {metric}")
        payload = torch.load(args.input / run["artifacts"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
        if payload["experiment"] != run["experiment"] or payload["seed"] != run["seed"]:
            raise AssertionError(f"Checkpoint identity mismatch: {key}")
        if payload["manifest_sha256"] != MANIFEST_SHA or payload["feature_cache_sha256"] != CACHE_SHA:
            raise AssertionError(f"Checkpoint input contract mismatch: {key}")
        logs = (args.input / run["artifacts"]["epoch_log"]["path"]).read_text().splitlines()
        if len(logs) != run["epochs_completed"]:
            raise AssertionError(f"Epoch count mismatch: {key}")
        verified_runs.append({"run_id": run["run_id"], "prediction_rows": len(frame),
                              "label_max_abs_error_from_float32_manifest": label_max_abs_error,
                              "recomputed_test_metrics": recomputed, "checkpoint_identity_verified": True,
                              "epoch_log_rows": len(logs)})

    contrasts = []
    for seed in (20261109, 20261110, 20261111):
        e0, e1, e2 = (by_key[(seed, experiment)]["test_metrics"]["srcc"] for experiment in ("E0", "E1", "E2"))
        contrasts.append({"seed": seed, "E1_minus_E0_srcc": e1 - e0, "E2_minus_E1_srcc": e2 - e1})
    e10 = [row["E1_minus_E0_srcc"] for row in contrasts]
    e21 = [row["E2_minus_E1_srcc"] for row in contrasts]
    replication = {
        "per_seed_contrasts": contrasts,
        "E1_minus_E0_mean_srcc": float(np.mean(e10)),
        "E1_minus_E0_positive_all_seeds": all(value > 0 for value in e10),
        "E2_minus_E1_mean_srcc": float(np.mean(e21)),
        "E2_minus_E1_positive_all_seeds": all(value > 0 for value in e21),
        "H3_registered_rule_supported_on_kaggle": float(np.mean(e21)) > 0 and all(value > 0 for value in e21),
    }
    files = sorted(path for path in args.input.rglob("*") if path.is_file())
    evidence = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "EXECUTION_VERIFIED",
        "kernel": "longltrnhong/spaq-phase-7-cached-multiseed", "kernel_version": 1,
        "source_commit": "3fd4008155429a834d01da88570c4514915a5263",
        "summary": summary, "verified_runs": verified_runs, "replication_contrasts": replication,
        "downloaded_artifacts": {str(path.relative_to(args.input)).replace("\\", "/"): sha256(path) for path in files},
        "interpretation": "Independent environment replication; numerical equality with local PyTorch/CUDA is not expected or claimed.",
    }
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "files": len(files), "replication": replication}, indent=2))


if __name__ == "__main__":
    main()
