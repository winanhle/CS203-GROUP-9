# DECISIONS — append only

## D-0001 — 2026-11-08 — Separate documentary and execution evidence

- **Decision:** Use the statuses `PAPER_REPORTED`, `DOCUMENT_VERIFIED`, `EXECUTION_VERIFIED`, `ESTIMATED`, `UNVERIFIED`, and `FAILED` for claims.
- **Reason:** The handoff contains proposals and literature results but no executed project data or training.
- **Alternatives rejected:** Treating paper values or archive listings as local results.
- **Evidence:** `AGENTS.md`; `research_sources/CODEX_MASTER_PROMPT.md`.

## D-0002 — 2026-11-08 — Create a new isolated repository

- **Decision:** Initialize `spaq-exif-robustness/` rather than modifying the handoff package.
- **Reason:** The workspace had no existing Git repository, and the controlling prompt explicitly prefers an isolated project folder.
- **Evidence:** environment preflight and Git initialization output.

## D-0003 — 2026-11-08 — Keep the study scope locked

- **Decision:** Core sequence is E0/E1, then E3 for minimum viable research, followed by E2/E4 when gates and measured budget allow. PrISM pseudo-labels are excluded from the core scope.
- **Reason:** Preserves causal contrasts and avoids an unverified label dependency.
- **Evidence:** research reports and acceptance criteria.

## D-0004 — 2026-11-09 — Use a resource-adapted frozen-feature track

- **Decision:** Compare E0-E4 using one frozen ImageNet-V2 ResNet-50 feature cache, identical split/seed/head budget, and validation-only checkpoint selection.
- **Reason:** Measured local extraction plus head training completes the full controlled campaign in minutes; Kaggle OAuth is externally unavailable.
- **Limitation:** This is not a faithful end-to-end retraining of the original SPAQ architecture and cannot establish training-seed uncertainty.
- **Evidence:** `evidence/feature_cache_resnet50.json`; E0-E4 evidence JSON files.

## D-0005 — 2026-11-09 — Stop adding fusion complexity

- **Decision:** End controlled model iteration after E4 and proceed to analysis/reporting.
- **Reason:** E2 does not reliably improve E1 under paired bootstrap, while E3/E4 metadata dropout is neutral or harmful and missingness degradation remains small.
- **Evidence:** `evidence/analysis_summary_s20261109.json`; `evidence/missingness_all_s20261109.json`.

## D-0006 — 2026-11-09 — Preserve Baseline v0.1 and preregister Phase 7

- **Decision:** Tag the verified G0-G6 state as `baseline-v0.1`, develop only on `phase7-hypothesis-extension`, and freeze Phase 7 seeds/contrasts before generating new test results.
- **Reason:** Separates immutable one-seed evidence from new confirmatory work and prevents post-test changes to thresholds or acceptance rules.
- **Evidence:** `evidence/phase7_baseline_v01_audit.json`; `configs/phase7_experiments.json`.

