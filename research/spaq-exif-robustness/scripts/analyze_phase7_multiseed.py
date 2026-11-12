"""Aggregate preregistered multi-seed and capture-condition contrasts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261109, 20261110, 20261111)
MODELS = ("E0", "E1", "E2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "srcc": float(spearmanr(y, prediction).statistic),
        "plcc": float(pearsonr(y, prediction).statistic),
        "mae": float(np.mean(np.abs(y - prediction))),
    }


def prediction_path(model: str, seed: int) -> Path:
    name = model.lower()
    if seed == 20261109:
        return ROOT / "artifacts" / "runs" / f"{name}_frozen_r50_s{seed}" / "predictions.csv"
    return ROOT / "artifacts" / "runs" / f"phase7_{name}_frozen_r50_s{seed}" / "predictions.csv"


def main() -> None:
    manifest_path = ROOT / "data" / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    train = manifest.loc[manifest["split"].eq("train")]
    iso_threshold = float(train["source_iso"].quantile(0.90))
    exposure_threshold = float(
        train.loc[train["source_exposure_valid"], "source_exposure_time"].quantile(0.90)
    )
    test = manifest.loc[manifest["split"].eq("test")].sort_values("image_name").reset_index(drop=True)
    subgroup_masks = {
        "all_test": np.ones(len(test), dtype=bool),
        "high_iso": test["source_iso"].to_numpy() >= iso_threshold,
        "long_exposure": (
            test["source_exposure_valid"].to_numpy(dtype=bool)
            & (test["source_exposure_time"].to_numpy() >= exposure_threshold)
        ),
        "night_scene": test["scene_night"].to_numpy() > 0,
    }

    frames: dict[tuple[int, str], pd.DataFrame] = {}
    input_hashes = {}
    for seed in SEEDS:
        for model in MODELS:
            path = prediction_path(model, seed)
            frame = pd.read_csv(path)
            frame = frame.loc[frame["split"].eq("test")].sort_values("image_name").reset_index(drop=True)
            if frame["image_name"].tolist() != test["image_name"].tolist():
                raise AssertionError(f"{model}/{seed}: test identifiers differ")
            if not np.allclose(frame["mos"], test["MOS"]):
                raise AssertionError(f"{model}/{seed}: MOS differs")
            frames[(seed, model)] = frame
            input_hashes[f"{model}_s{seed}"] = sha256(path)

    metric_rows = []
    contrast_rows = []
    for seed in SEEDS:
        for subgroup, mask in subgroup_masks.items():
            y = test.loc[mask, "MOS"].to_numpy(np.float64)
            per_model = {}
            for model in MODELS:
                prediction = frames[(seed, model)].loc[mask, "prediction"].to_numpy(np.float64)
                result = metrics(y, prediction)
                per_model[model] = result
                metric_rows.append(
                    {"seed": seed, "subgroup": subgroup, "n": int(mask.sum()), "model": model, **result}
                )
            for candidate, control in (("E1", "E0"), ("E2", "E1")):
                contrast_rows.append(
                    {
                        "seed": seed,
                        "subgroup": subgroup,
                        "n": int(mask.sum()),
                        "contrast": f"{candidate}-{control}",
                        **{name: per_model[candidate][name] - per_model[control][name] for name in ("srcc", "plcc", "mae")},
                    }
                )

    metrics_frame = pd.DataFrame(metric_rows)
    contrasts_frame = pd.DataFrame(contrast_rows)
    aggregate = (
        contrasts_frame.groupby(["subgroup", "contrast", "n"])[["srcc", "plcc", "mae"]]
        .agg(["mean", "std", "min", "max"])
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    sign_rows = []
    for (subgroup, contrast), frame in contrasts_frame.groupby(["subgroup", "contrast"]):
        sign_rows.append(
            {
                "subgroup": subgroup,
                "contrast": contrast,
                "srcc_positive_all_seeds": bool((frame["srcc"] > 0).all()),
                "srcc_negative_all_seeds": bool((frame["srcc"] < 0).all()),
            }
        )

    all_mean = float(
        aggregate.loc[
            aggregate["subgroup"].eq("all_test") & aggregate["contrast"].eq("E1-E0"), "srcc_mean"
        ].iloc[0]
    )
    h1 = {}
    for subgroup in ("high_iso", "long_exposure", "night_scene"):
        row = aggregate.loc[
            aggregate["subgroup"].eq(subgroup) & aggregate["contrast"].eq("E1-E0")
        ].iloc[0]
        per_seed = contrasts_frame.loc[
            contrasts_frame["subgroup"].eq(subgroup) & contrasts_frame["contrast"].eq("E1-E0"), "srcc"
        ]
        h1[subgroup] = {
            "mean_delta_srcc": float(row["srcc_mean"]),
            "exceeds_all_test_mean": bool(row["srcc_mean"] > all_mean),
            "positive_all_seeds": bool((per_seed > 0).all()),
            "supports_registered_rule": bool(row["srcc_mean"] > all_mean and (per_seed > 0).all()),
        }
    h3_seed = contrasts_frame.loc[
        contrasts_frame["subgroup"].eq("all_test") & contrasts_frame["contrast"].eq("E2-E1"), "srcc"
    ]
    h3 = {
        "mean_delta_srcc": float(h3_seed.mean()),
        "positive_all_seeds": bool((h3_seed > 0).all()),
        "supports_registered_rule": bool(h3_seed.mean() > 0 and (h3_seed > 0).all()),
    }

    evidence_dir = ROOT / "evidence"
    metrics_path = evidence_dir / "phase7_multiseed_metrics.csv"
    contrasts_path = evidence_dir / "phase7_multiseed_contrasts.csv"
    aggregate_path = evidence_dir / "phase7_multiseed_aggregate.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    contrasts_frame.to_csv(contrasts_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    figures = ROOT / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(SEEDS))
    width = 0.24
    for offset, model in enumerate(MODELS):
        values = [
            metrics_frame.loc[
                metrics_frame["seed"].eq(seed)
                & metrics_frame["subgroup"].eq("all_test")
                & metrics_frame["model"].eq(model),
                "srcc",
            ].iloc[0]
            for seed in SEEDS
        ]
        ax.bar(x + (offset - 1) * width, values, width, label=model)
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set(ylim=(0.81, 0.845), xlabel="Training seed", ylabel="Test SRCC", title="Phase 7 matched multi-seed results")
    ax.legend()
    fig.tight_layout()
    multiseed_figure = figures / "phase7_multiseed_srcc.png"
    fig.savefig(multiseed_figure, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    delta_frame = contrasts_frame.loc[contrasts_frame["contrast"].eq("E1-E0")]
    for subgroup, frame in delta_frame.groupby("subgroup"):
        ax.plot(frame["seed"].astype(str), frame["srcc"], marker="o", label=subgroup)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set(xlabel="Training seed", ylabel="Delta SRCC (E1-E0)", title="Conditional EXIF utility by seed")
    ax.legend()
    fig.tight_layout()
    subgroup_figure = figures / "phase7_conditional_delta_srcc.png"
    fig.savefig(subgroup_figure, dpi=180)
    plt.close(fig)

    output_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (metrics_path, contrasts_path, aggregate_path, multiseed_figure, subgroup_figure)
    }
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "test_rows": len(test),
        "subgroup_counts": {name: int(mask.sum()) for name, mask in subgroup_masks.items()},
        "subgroup_thresholds": {"high_iso": iso_threshold, "long_exposure_seconds": exposure_threshold},
        "input_prediction_sha256": input_hashes,
        "per_seed_metrics": metric_rows,
        "per_seed_contrasts": contrast_rows,
        "aggregate_contrasts": aggregate.to_dict(orient="records"),
        "sign_consistency": sign_rows,
        "H1_registered_evaluation": {"all_test_mean_delta_srcc": all_mean, "subgroups": h1},
        "H3_registered_evaluation": h3,
        "output_sha256": output_hashes,
        "limitations": [
            "Three seeds quantify limited optimization variability; they do not support a population-level random-effects claim.",
            "Capture-condition groups overlap and describe conditional association, not causal metadata utility.",
        ],
    }
    summary_path = evidence_dir / "phase7_multiseed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "H1": summary["H1_registered_evaluation"], "H3": h3}, indent=2))


if __name__ == "__main__":
    main()
