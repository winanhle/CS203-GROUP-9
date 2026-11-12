# SESSION HANDOFF

**Updated UTC:** 2026-11-09T12:53:00Z
**Repo path/commit:** `D:\HCMUS\Year2\Scientific Method\SC\spaq-exif-robustness` / `2a378d824106c0c69f9d6c718376b544b7db3561`, branch `phase7-hypothesis-extension`
**Most advanced completed gate:** G7-007; Phase 7 complete (G0-G6 preserved at `baseline-v0.1`)
**Active Kaggle kernel/version/status:** `longltrnhong/spaq-phase-7-cached-multiseed` / v1 / COMPLETE
**Last valid checkpoint:** `artifacts/kaggle_download/phase7_backbone_v1/layer4_finetuned_best.pt`

## Verified completed tasks

- Read all controlling local documents and both research reports.
- Verified the handoff package hashes against its manifest.
- Initialized the project repository and durable progress ledger; committed the G0/recovery milestone.
- Installed Kaggle CLI 2.2.5 and established OAuth for `longltrnhong`.
- Downloaded and verified Q-Align `spaq.tar`: 11,125 readable JPEGs, no annotations or embedded EXIF.
- Retrieved pinned IQA-PyTorch metadata and matched all 11,125 rows to archive filenames with no orphans/collisions.
- Hashed every Q-Align image and identified 322 exact duplicate pairs. All pairs conflict on at least one subjective field, 321 conflict on MOS, all agree on EXIF, and 103 cross the Q-Align train/test boundary.
- Downloaded and hash-verified the 34.8 GB author-recommended archive; streamed it without full extraction and recovered the original workbooks.
- Proved exact 11,125-row workbook/mirror equality and audited embedded EXIF for every original JPEG.
- Froze the repeated-content-grouped split and built `data/manifest.parquet` with observed/estimated masks.
- Installed `.venv312`; local RTX 4050 CUDA operation and 10 tests pass.
- Extracted all 11,125 compact images and ran the 16-image/20-step ResNet-50 vertical slice with atomic checkpoint, bit-exact reload, optimizer resume, and 1.20 GB peak allocation.
- Four-image overfit sanity reduced L1 from 54.29 to 1.49 (minimum 0.17) in 200 steps; 12 tests pass.
- Packaged and executed the standalone Kaggle smoke runner against a private verified dataset.
- Extracted one verified 11,125x2,048 frozen ResNet-50 cache and ran E0/E1/E3 with identical split, seed, visual features, budget, and validation-only checkpoint selection.
- E1 improves test SRCC by 0.00565 and MAE by 0.1547 over E0 in this one seed. E3 is below E1 at full metadata.
- Evaluated 0/25/50/100% record-level metadata missingness with five fixed stochastic masks at 25/50%; E3 dropout gives no measured robustness advantage.
- Completed matched E2 gated and E4 gated-plus-dropout runs. E2 test SRCC/PLCC/MAE is 0.83602/0.83779/9.1179; E4 is 0.83208/0.83337/9.3265.
- Extended missingness evaluation to all metadata models; E2 falls to 0.83290 SRCC and E4 to 0.82813 at zero available EXIF.
- Completed 10,000-resample paired inference: E1-E0 SRCC delta 0.00565 [0.00132, 0.01013]; all E2-E1 metric intervals cross zero.
- Completed train-defined high-ISO (n=235), long-exposure (n=316), and night-scene (n=191) analysis, deterministic top-error audit, and seven hashed figures.
- Completed README, experiment report, Kaggle runbook, acceptance audit, and automated release validation. Sixteen tests and dependency checks pass.
- Established Kaggle OAuth and uploaded the verified compact dataset privately. Kaggle expanded the tar into 11,125 images; v2 verifies the deterministic tree hash of the exact 300-image smoke sample.
- Kaggle v1 timed out during the overly broad full-tree integrity scan. Controlled v2 verified the exact 300-image smoke sample, completed on Tesla T4, and produced downloaded/hash-verified checkpoint, summary, and log artifacts.
- Tagged and re-audited Baseline v0.1; all E0-E4 artifacts match their recorded hashes. Preregistered Phase 7 before new test execution.
- Completed all nine E0/E1/E2 seed/model combinations without changing the baseline seed artifacts; hashes and metrics are registered in `EXPERIMENTS.csv` and per-run evidence JSON.
- Aggregated H1/H3 exactly as preregistered: H1 is not supported because subgroup signs vary by seed; H3 is not supported because E2-E1 SRCC is negative in all three seeds.
- Completed five-seed whole-record permutation, missingness, and inference-only feature-group diagnostics for all E1/E2 checkpoints. H2's rule passes, but its measured effect is small and is not interpreted causally.
- Executed H4 on Kaggle Tesla T4. Frozen test SRCC is 0.85134; layer4-fine-tuned is 0.88187. All eight outputs and trainable-only checkpoint contents were independently verified.
- Independently retrained E0/E1/E2 across all three seeds on Kaggle. All 29 outputs verify; H3 remains unsupported and additive EXIF remains seed-dependent.
- Ran official KonIQ-trained TOPIQ-NR only after the core study. Test SRCC/PLCC/validation-calibrated MAE is 0.82005/0.82685/9.0969. Never use `topiq_nr-spaq` for this held-out comparison.
- Updated the reproducible README, Phase 7 report, experiment registry, evidence ledger, and release validator. Final validator and 16 tests pass.

## Last operation, command and result

- Last operation: completed G7-007, ran final tests/release validation, and prepared the final ledger commit. No Phase 7 experiment remains active.

## Failed/blocked steps

- `python --version` resolves to a nonfunctional Windows Store alias; use `py`.
- `gh` is absent, but HTTPS authentication succeeded and `main` tracks GitHub `origin/main`.
- Source row `03461.jpg` contains exposure `1,000,000`, while embedded exposure is invalid/non-finite. Never invent a replacement; retain the author's distinct APEX shutter value with provenance.

## Data provenance and frozen split references

- Q-Align archive SHA-256: `d312242f10af983d5ceaac433c76b7c2b3473e2a2f87782f9db8ee1537dc2194`.
- Flattened metadata SHA-256: `17d4eaedeff3c2724b8f6fee9f9dbfef65ac8824bd31b774760d86b53a719e8f`.
- Canonical manifest exact join: 11,125/11,125; SHA-256 `abff8b1a...5800d`.
- Frozen split SHA-256 `91a8d1b2...a108`; train/val/test 7,786/1,111/2,228.

## Exact next commands

1. `.\.venv312\Scripts\python.exe -m pytest -q`
2. `.\.venv312\Scripts\python.exe scripts\validate_release.py`
3. `git status --short` (expected clean after the final ledger commit)

## Warnings

- Do not report paper metrics as project results.
- Do not accept the Hugging Face archive license as proof of underlying SPAQ image rights.
- Do not use `topiq_nr-spaq` as an unseen SPAQ comparator.
- Local and Kaggle PyTorch/CUDA versions differ; do not claim bitwise numerical replication.
- H4 is one seed. Do not promote it to a confirmatory general claim without preregistered replication.

