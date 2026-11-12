"""Verify that the completed frozen-feature baseline remains intact."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("e0", "e1", "e2", "e3", "e4")
EXPECTED_MANIFEST = "abff8b1a5f7d2e74d0c8874655fbe40d5e75de75bf61e65753c7085cee15800d"
EXPECTED_CACHE = "b639bb6f2365f31004410af1a71a35529a485de849d70e6924f7fbb2dc3975ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    baseline_commit = git("rev-list", "-n", "1", "baseline-v0.1")
    manifest = ROOT / "data" / "manifest.parquet"
    cache = ROOT / "artifacts" / "features" / "spaq_resnet50_imagenet_v2_abff8b1a.npz"
    if sha256(manifest) != EXPECTED_MANIFEST or sha256(cache) != EXPECTED_CACHE:
        raise AssertionError("Frozen manifest or feature cache changed")

    models = {}
    for model in MODELS:
        evidence_path = ROOT / "evidence" / f"{model}_frozen_r50_s20261109.json"
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if payload["status"] != "SUCCESS":
            raise AssertionError(f"{model}: baseline status is not SUCCESS")
        verified = {}
        for artifact_name in ("config", "best_checkpoint", "last_checkpoint", "epoch_log", "predictions"):
            artifact = payload["artifacts"][artifact_name]
            path = Path(artifact["path"])
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                raise AssertionError(f"{model}: {artifact_name} missing or hash mismatch")
            verified[artifact_name] = artifact["sha256"]
        models[model.upper()] = {
            "experiment": payload["experiment"],
            "track": payload["track"],
            "test_metrics": payload["test_metrics"],
            "validation_metrics": payload["validation_metrics"],
            "parameter_count": payload["parameter_count"],
            "metadata_parameter_count": payload["metadata_parameter_count"],
            "artifact_sha256": verified,
        }

    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "baseline_version": "v0.1",
        "baseline_tag": "baseline-v0.1",
        "baseline_commit": baseline_commit,
        "protocol": "resource-adapted frozen torchvision ResNet-50 ImageNet1K-V2 feature cache",
        "not_full_finetuning": True,
        "manifest_sha256": EXPECTED_MANIFEST,
        "feature_cache_sha256": EXPECTED_CACHE,
        "models": models,
        "branch_at_audit": git("branch", "--show-current"),
    }
    output = ROOT / "evidence" / "phase7_baseline_v01_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
