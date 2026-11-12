# TASKS

Legend: `TODO`, `IN_PROGRESS`, `DONE`, `FAILED`, `BLOCKED`, `NOT_APPLICABLE`. Every completed task cites real evidence.

## Phase 0 — repository and source audit

- [x] **G0-001 DONE** Read controlling documents and audit primary papers/repositories/data endpoints. Evidence: `docs/SOURCE_AUDIT.md`, pinned SHAs in `EVIDENCE_LOG.md`.
- [x] **G0-002 DONE** Create repository, durable state ledger, test runner skeleton, and initial clean commit. Evidence: commit `2297c0539e14056182f40dbbada4054b79bbb5ea`.
- [x] **G0-003 DONE** Probe Python/Git/GPU/GitHub/Kaggle without exposing secrets. Kaggle auth was initially absent and was later established through OAuth. Evidence: `EVIDENCE_LOG.md`.

## Phase 1 — data and provenance

- [x] **G1-001 DONE** Retrieved both compact and author-recommended archives; verified hashes, 11,125 original images, and three authentic workbooks. Evidence: `evidence/q_align_spaq_archive.json`, `evidence/iqa_pytorch_spaq_archive.json`.
- [x] **G1-002 DONE** Audited schemas, joins, embedded EXIF, per-field provenance, duplicates, terms, and preprocessing differences. Evidence: `evidence/spaq_original_workbook_audit.json`, `evidence/spaq_embedded_exif_audit.json`, `data/provenance.json`.
- [x] **G1-003 DONE** Frozen seed-20261109 group-safe split; 1,069 candidate edges cross zero partitions. Manifest SHA-256 `abff8b1a...5800d`; contract tests pass. Evidence: `data/manifest.parquet`, `evidence/spaq_split_seed20261109_audit.json`.

## Phase 2 — vertical slice and compute smoke

- [x] **G2-001 DONE** Ran 16-image ResNet-50 forward/backward, 20 updates, atomic checkpoint, bit-exact reload evaluation, optimizer resume, and four-image overfit sanity; 12 tests pass. Evidence: `evidence/g2_vertical_slice.json`, `evidence/g2_tiny_overfit.json`.
- [x] **G2-002 DONE** Kaggle GPU smoke v2 completed on Tesla T4; verified 300-image sample integrity, timing, VRAM, 20 train steps, checkpoint reload/resume, output retrieval, and hashes. Evidence: `evidence/kaggle_smoke_v2.json`.

## Phase 3 — matched baselines

- [x] **G3-001 DONE** E0 resource-adapted frozen-ResNet50 run completed; test SRCC/PLCC/MAE 0.83048/0.83063/9.3489 with checkpoint, logs, and predictions. Evidence: `evidence/e0_frozen_r50_s20261109.json`.
- [x] **G3-002 DONE** Matched E1 additive EXIF run completed; test 0.83613/0.83572/9.1942. Evidence: `evidence/e1_frozen_r50_s20261109.json`.

## Phase 4 — controlled extensions

- [x] **G4-001 DONE** E3 with 50% record-level metadata dropout completed; test 0.83450/0.83452/9.2534. Missingness curve showed no robustness advantage over E1. Evidence: `evidence/e3_frozen_r50_s20261109.json`, `evidence/missingness_e1_e3_s20261109.json`.
- [x] **G4-002 DONE** E2 gated fusion completed; test SRCC/PLCC/MAE 0.83602/0.83779/9.1179. Evidence: `evidence/e2_frozen_r50_s20261109.json`.
- [x] **G4-003 DONE** E4 gated plus 50% metadata dropout completed; test 0.83208/0.83337/9.3265. Dropout reduced full- and missing-metadata performance. Evidence: `evidence/e4_frozen_r50_s20261109.json`, `evidence/missingness_all_s20261109.json`.

## Phase 5 — analysis

- [x] **G5-001 DONE** Completed 10,000-resample paired bootstrap, train-defined capture-condition subgroups, deterministic top-error review, and seven provenance-hashed figures. Evidence: `evidence/analysis_summary_s20261109.json`.

## Phase 6 — deliverable

- [x] **G6-001 DONE** Completed README reproduction guide, experiment report, acceptance audit, release validator, Kaggle evidence retrieval, and context-independent handoff. Only optional GitHub publication remains. Evidence: `evidence/release_validation.json`, `docs/ACCEPTANCE_AUDIT.md`.

## Phase 7 — hypothesis-driven extension

- [x] **G7-001 DONE** Preserved completed E0-E4 as tag `baseline-v0.1`; verified every baseline config/checkpoint/log/prediction hash and documented the frozen-feature limitation. Evidence: `evidence/phase7_baseline_v01_audit.json`.
- [x] **G7-002 DONE** Preregistered H1-H4 seeds, variables, controls, subgroup definitions, permutation seeds, metrics, and acceptance rules before new test execution. Evidence: `configs/phase7_experiments.json`, `docs/RESEARCH_DIRECTION.md`.
- [x] **G7-003 DONE** Executed matched E0/E1/E2 on three training seeds and aggregated preregistered contrasts. Evidence: `evidence/phase7_multiseed_summary.json`.
- [x] **G7-004 DONE** Executed correct-pair EXIF permutation plus inference-only feature-group leave-out/only diagnostics on all E1/E2 checkpoints. Evidence: `evidence/phase7_metadata_diagnostics.json`.
- [x] **G7-005 DONE** Aggregated missingness and train-defined capture-condition results across all three seeds. Evidence: `evidence/phase7_multiseed_summary.json`, `evidence/phase7_metadata_diagnostics.json`.
- [x] **G7-006 DONE** Ran the matched frozen versus layer4-fine-tuned ResNet-50 pair on Kaggle Tesla T4 and independently verified all downloaded logs, predictions, and checkpoints. Evidence: `evidence/phase7_kaggle_backbone_v1.json`.
- [x] **G7-007 DONE** Completed and verified nine-run Kaggle replication, Phase 7 report/figures, KonIQ-trained TOPIQ exploration, release validation, persistent handoff, and branch push. Evidence: `evidence/phase7_kaggle_cached_v1.json`, `evidence/phase7_topiq_nr_koniq.json`, `evidence/release_validation.json`.

