# Baseline v0.1 versus Phase 7 extension

| Dimension | Baseline v0.1 (preserved) | Phase 7 extension |
|---|---|---|
| Git boundary | Tag `baseline-v0.1` | Branch `phase7-hypothesis-extension` |
| Visual representation | One frozen ResNet-50 ImageNet-V2 feature cache | Same cache for confirmatory multi-seed work; matched frozen/fine-tuned image pipeline executed |
| Training seeds | One: 20261109 | 20261109, 20261110, 20261111 for E0/E1/E2 |
| Metadata tests | Presence, fusion, dropout, record missingness | Correct-pair permutation and feature-group diagnostics |
| Capture conditions | One-seed descriptive subgroup table | Per-seed paired subgroup deltas with fixed train-derived definitions |
| Fusion claim | E2 approximately tied with E1 in one seed | Test whether E2-E1 is positive in all three seeds |
| Backbone claim | No full fine-tuning | One-seed exploratory matched frozen versus layer4-fine-tuned comparison; not full-network tuning |
| Kaggle | GPU smoke only | Nine cached-head runs plus matched backbone pair, with retrieved/verified logs, checkpoints, and predictions |
| Test policy | Validation-only selection; test once | Same policy; no threshold or hyperparameter changes after Phase 7 test inspection |

Baseline artifacts remain under their original paths and hashes. New runs use distinct `phase7_*` run IDs and evidence files. No Phase 7 result may overwrite an E0–E4 baseline artifact.
