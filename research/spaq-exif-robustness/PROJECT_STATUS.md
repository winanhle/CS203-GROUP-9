# PROJECT STATUS

**Updated UTC:** 2026-11-09T12:53:00Z
**Phase:** Phase 7 — hypothesis-driven extension
**Last successful gate:** G7-007 (Phase 7 complete)
**Git commit / remote:** `2a378d824106c0c69f9d6c718376b544b7db3561` / `origin` (GitHub), branch `phase7-hypothesis-extension`
**Kaggle kernel/version/status:** `longltrnhong/spaq-phase-7-cached-multiseed` / v1 / COMPLETE

## Current objective

Phase 7 is execution-complete. Preserve the verified release and use the documented evidence for reporting or a separately preregistered follow-up.

## Execution-verified achievements

- Handoff package hashes match `PACKAGE_MANIFEST.md` for all eight controlling/source documents checked.
- Local Git 2.50.1 is available and a `main` repository was initialized.
- Python 3.13.5 is available through `py`; the `python` alias is not usable.
- Local NVIDIA RTX 4050 Laptop GPU is visible with 6141 MiB VRAM; this is not Kaggle evidence.
- `gh` is not installed. Kaggle CLI 2.2.5 is authenticated through OAuth as `longltrnhong`.
- Q-Align `spaq.tar` downloaded: 2,626,897,920 bytes; SHA-256 matches the Hugging Face linked ETag.
- The archive contains exactly 11,125 unique, readable JPEGs and no annotation files; none contains embedded EXIF.
- IQA-PyTorch's pinned metadata release has 11,125 unique SPAQ rows with MOS, attributes, scene annotations, EXIF-derived columns, and no nulls.
- Exact filename join between that CSV and Q-Align archive is 11,125/11,125 with no orphans or collisions.
- A full byte-hash audit found 322 exact duplicate pairs (644 files). All 322 have conflicting annotation fields; 321 have different MOS, while all eight EXIF-derived fields agree within every pair. The inherited Q-Align split separates 103 pairs, so it is not acceptable for this project.
- Author-recommended `spaq.tgz` is 34,841,101,349 bytes; SHA-256 `54e0cb49...7418f7` matches its LFS object. It contains 11,125 originals and three workbooks.
- All 11,125 workbook IDs and 24 MOS/attribute/scene/source-EXIF values match the flattened mirror exactly.
- Embedded EXIF was audited for all originals: scene brightness is observed for 8,546 rows and author-estimated for 2,579.
- The frozen split groups all 1,069 repeated-content edges: 7,786/1,111/2,228 rows and zero edge crossings.
- Canonical `data/manifest.parquet` has 11,125 rows and SHA-256 `abff8b1a...5800d`.
- Python 3.12.10/PyTorch 2.14.1+cu130 completed a CUDA operation on the RTX 4050.
- The 16-image ResNet-50 slice completed 20 updates, bit-exact checkpoint reload, optimizer resume, and step-21 checkpoint with 1.20 GB peak allocation.
- Four-image overfit sanity reduced L1 from 54.29 to 1.49 (minimum 0.17) over 200 updates. Twelve tests pass.
- The standalone Kaggle smoke completed on Tesla T4 and its outputs were retrieved and verified.
- Frozen ResNet-50 features for all 11,125 rows were extracted once at 92.89 images/s; cache SHA-256 is `b639bb6f...975ed`.
- E0 test SRCC/PLCC/MAE: 0.83048/0.83063/9.3489. E1 additive EXIF: 0.83613/0.83572/9.1942. E3 additive plus dropout: 0.83450/0.83452/9.2534.
- Missingness stress shows a very small metadata contribution: E1 SRCC falls only 0.00034 from full to zero EXIF; E3 falls 0.00018 and does not outperform E1 at any availability level.
- E2 gated fusion completed with 8,704 metadata parameters: test SRCC/PLCC/MAE 0.83602/0.83779/9.1179. It is effectively tied with E1 on SRCC while improving PLCC and MAE in this seed.
- E4 gated plus 50% metadata dropout completed at 0.83208/0.83337/9.3265, below E2 and E1. E2 SRCC declines to 0.83290 at 100% metadata missingness; E4 declines to 0.82813.
- A 10,000-resample paired bootstrap found E1-E0 delta SRCC 0.00565, 95% CI [0.00132, 0.01013]. E2-E1 crosses zero on SRCC/PLCC/MAE; no reliable gated advantage was detected.
- Capture-condition subsets contain 235 high-ISO, 316 long-exposure, and 191 night-scene test images. E1 leads the first two by SRCC; E2 narrowly leads night scenes. These are descriptive subgroup results.
- Seven figures, paired tables, and a deterministic 20-case E1 error audit are hash-indexed in `evidence/analysis_summary_s20261109.json`.
- Release validation passed: canonical manifest/hash/counts/group isolation, eight evidence statuses, ten analysis-output hashes, three Kaggle-output hashes, ignore policy, required documents, 16 tests, and dependency consistency.
- Kaggle OAuth is established for `longltrnhong`. Private dataset `longltrnhong/spaq-compact-verified` contains the verified manifest and all 11,125 expanded compact images.
- Kaggle smoke v2 completed on Tesla T4: 300-image inference 11.09 s, 20 train steps 2.39 s, 580 MB peak allocation, checkpoint reload and optimizer resume successful. Downloaded artifacts match recorded hashes.
- Baseline v0.1 is preserved at tag `baseline-v0.1`; all E0-E4 artifacts revalidated by SHA-256. Existing results are explicitly frozen-feature, not full fine-tuning.
- Phase 7 hypotheses and acceptance criteria were preregistered before new test execution in `configs/phase7_experiments.json`.
- E0/E1/E2 now each have three matched training seeds. Mean E1-E0 test delta SRCC is +0.001878, but one seed is negative; the effect is small and not sign-consistent.
- H1 is not supported under its registered rule: high-ISO, long-exposure, and night subsets have larger mean E1-E0 SRCC than all-test, but every subgroup has at least one non-positive seed.
- H2's registered pairing test is supported with small effects: intact-minus-permuted mean SRCC is +0.000351 for E1 and +0.002593 for E2, positive for all three training seeds. This is pairing sensitivity, not causal proof.
- H3 is not supported: mean E2-E1 test delta SRCC is -0.001317 and E2 is lower on all three seeds.
- Missingness and feature-group evaluations were executed from fixed checkpoints. Feature-group results are inference-only diagnostics, not retrained causal ablations.
- H4 completed on Kaggle Tesla T4: frozen versus layer4-fine-tuned test SRCC 0.85134 versus 0.88187. This +0.03052 result is exploratory because it has one training seed.
- Kaggle independently retrained all nine E0/E1/E2 seed combinations. Mean E1-E0 SRCC is -0.000057 and mean E2-E1 is -0.001646; neither comparison is positive across all seeds.
- All 8 H4 and 29 cached-replication downloaded files match their recorded hashes; checkpoints, logs, predictions, IDs, labels, and recomputed metrics were verified.
- Official KonIQ-trained TOPIQ-NR was explored only after core tests. Validation-only calibrated test SRCC/PLCC/MAE is 0.82005/0.82685/9.0969. The SPAQ-trained TOPIQ checkpoint was excluded.
- Final Phase 7 release validation passes with 16 tests and all manifest, split, evidence, output-hash, ignore, and documentation contracts.

## Current task

- No active task. G7-001 through G7-007 are complete.

## Actual blockers

- No active blocker remains.

## Observed results

G0-G6 REMAIN COMPLETE AND PRESERVED AS BASELINE V0.1. PHASE 7 IS COMPLETE. H1 AND H3 FAILED THEIR REGISTERED SUPPORT RULES; H2 PASSED WITH A SMALL PAIRING-SENSITIVITY EFFECT. H4 FAVORED LAYER4 TUNING IN ONE EXPLORATORY SEED. KAGGLE REPLICATION AND TOPIQ EXPLORATION ARE VERIFIED.

## Next actions

1. Re-run `python -m pytest -q` and `scripts/validate_release.py` before publishing or merging.
2. If extending H4, preregister at least two additional seeds before inspecting their test results.
3. Treat external-dataset validation or device-OOD work as a new phase; it is not implied by Phase 7 completion.

## Relevant evidence and file paths

- `evidence/q_align_spaq_archive.json`
- `evidence/spaq_qalign_join_audit.json`
- `evidence/q_align_duplicate_audit.json`
- `evidence/q_align_duplicate_label_audit.json`
- `evidence/iqa_pytorch_spaq_archive.json`
- `evidence/spaq_original_workbook_audit.json`
- `evidence/spaq_embedded_exif_audit.json`
- `evidence/spaq_manifest_audit.json`
- `evidence/g2_vertical_slice.json`
- `evidence/g2_tiny_overfit.json`
- `evidence/e0_frozen_r50_s20261109.json`
- `evidence/e1_frozen_r50_s20261109.json`
- `evidence/e3_frozen_r50_s20261109.json`
- `evidence/e2_frozen_r50_s20261109.json`
- `evidence/e4_frozen_r50_s20261109.json`
- `evidence/missingness_all_s20261109.json`
- `evidence/analysis_summary_s20261109.json`
- `evidence/paired_bootstrap_s20261109.csv`
- `evidence/subgroup_metrics_s20261109.csv`
- `evidence/release_validation.json`
- `docs/EXPERIMENT_REPORT.md`
- `docs/ACCEPTANCE_AUDIT.md`
- `evidence/kaggle_smoke_v2.json`
- `evidence/phase7_baseline_v01_audit.json`
- `configs/phase7_experiments.json`
- `docs/RESEARCH_DIRECTION.md`
- `evidence/phase7_multiseed_summary.json`
- `evidence/phase7_metadata_diagnostics.json`
- `docs/PHASE7_EXPERIMENT_REPORT.md`
- `evidence/phase7_kaggle_backbone_v1.json`
- `evidence/phase7_kaggle_cached_v1.json`
- `evidence/phase7_topiq_nr_koniq.json`
- `docs/DATASET_RECOVERY.md`
- `docs/SOURCE_AUDIT.md`
- `EVIDENCE_LOG.md`

