"""Evaluate KonIQ-trained TOPIQ-NR on SPAQ without SPAQ fine-tuning.

Validation predictions fit one affine score-to-MOS calibration. The test split is
then inferred once; raw rank correlation and validation-calibrated metrics are
reported as exploratory cross-dataset evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyiqa
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = "abff8b1a5f7d2e74d0c8874655fbe40d5e75de75bf61e65753c7085cee15800d"
CHECKPOINT_SHA = "9a73138beb3748dd6a83365610ac88fa1cd53225342c77fc9ed44a3be461354c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_split(metric, frame, image_root, output_path):
    rows = []
    started = time.perf_counter()
    for index, row in enumerate(frame.itertuples(index=False), 1):
        with torch.inference_mode():
            score = float(metric(str(image_root / row.image_name)).detach().cpu().reshape(-1)[0])
        rows.append({"image_name": row.image_name, "MOS": float(row.MOS), "raw_score": score})
        if index % 100 == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(json.dumps({"split": str(output_path.stem), "completed": index, "total": len(frame)}), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result, time.perf_counter() - started


def metrics(labels, predictions):
    return {
        "srcc": float(spearmanr(labels, predictions).statistic),
        "plcc": float(pearsonr(labels, predictions).statistic),
        "mae": float(np.mean(np.abs(np.asarray(labels) - np.asarray(predictions)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.parquet")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/q_align/extracted/spaq")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/torch/hub/pyiqa/cfanet_nr_koniq_res50-9a73138b.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/runs/phase7_topiq_nr_koniq")
    parser.add_argument("--evidence-output", type=Path, default=ROOT / "evidence/phase7_topiq_nr_koniq.json")
    args = parser.parse_args()
    if sha256(args.manifest) != MANIFEST_SHA or sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("Manifest or TOPIQ checkpoint hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.manifest).sort_values("image_name", ignore_index=True)
    validation = manifest.loc[manifest.split == "val", ["image_name", "MOS"]]
    test = manifest.loc[manifest.split == "test", ["image_name", "MOS"]]
    metric = pyiqa.create_metric("topiq_nr", device="cuda" if torch.cuda.is_available() else "cpu")

    validation_path = args.output_dir / "validation_predictions.csv"
    validation_predictions, validation_seconds = infer_split(metric, validation, args.image_root, validation_path)
    design = np.column_stack([validation_predictions.raw_score.to_numpy(), np.ones(len(validation_predictions))])
    slope, intercept = np.linalg.lstsq(design, validation_predictions.MOS.to_numpy(), rcond=None)[0]
    validation_calibrated = slope * validation_predictions.raw_score.to_numpy() + intercept
    validation_metrics = metrics(validation_predictions.MOS, validation_calibrated)

    # Test inference occurs only after the validation-only calibration is fixed.
    test_path = args.output_dir / "test_predictions.csv"
    test_predictions, test_seconds = infer_split(metric, test, args.image_root, test_path)
    test_calibrated = slope * test_predictions.raw_score.to_numpy() + intercept
    test_predictions["validation_affine_calibrated_prediction"] = test_calibrated
    test_predictions.to_csv(test_path, index=False)
    test_metrics = metrics(test_predictions.MOS, test_calibrated)
    raw_test_metrics = metrics(test_predictions.MOS, test_predictions.raw_score)

    evidence = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "SUCCESS",
        "experiment": "TOPIQ_NR_KONIQ_CROSS_DATASET_EXPLORATORY",
        "role": "stronger image-only IQA baseline explored only after H1-H4 core tests",
        "not_used": "topiq_nr-spaq checkpoint, because SPAQ training overlap would invalidate held-out comparison",
        "model": "pyiqa topiq_nr / CFANet ResNet-50 / KonIQ-trained default",
        "pyiqa_version": pyiqa.__version__, "checkpoint_sha256": CHECKPOINT_SHA,
        "manifest_sha256": MANIFEST_SHA, "validation_rows": len(validation), "test_rows": len(test),
        "selection_and_calibration": "single affine raw-score-to-MOS mapping fit on validation only; no model fitting; test inferred once",
        "affine_calibration": {"slope": float(slope), "intercept": float(intercept)},
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "raw_test_metrics": raw_test_metrics,
        "runtime_seconds": {"validation": validation_seconds, "test": test_seconds},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "artifacts": {
            "validation_predictions": {"path": str(validation_path.relative_to(ROOT)), "sha256": sha256(validation_path)},
            "test_predictions": {"path": str(test_path.relative_to(ROOT)), "sha256": sha256(test_path)},
        },
        "limitations": [
            "Cross-dataset pretrained inference is not a matched training-budget comparison.",
            "TOPIQ sees compact re-encoded SPAQ images and uses its official default inference preprocessing.",
            "This analysis is exploratory and was initiated only after the preregistered core hypotheses were tested.",
        ],
    }
    args.evidence_output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
