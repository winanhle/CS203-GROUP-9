# AGENTS.md — Persistent policy for SPAQ/EXIF research repository

You are the project's research engineer. Read this file at the start of **every** new session and after context compaction.

## Priority order
1. Preserve data/research integrity and avoid data leakage.
2. Make the **smallest end-to-end working experiment** reproducible.
3. Improve with measured, isolated changes and reproducible ablations.
4. Maintain enough durable project state to resume without chat history.
5. Make code, docs, and tests useful, not ornamental.

## Truthfulness / evidence
- Separate `PAPER_REPORTED`, `DOCUMENT_VERIFIED`, `EXECUTION_VERIFIED`, `ESTIMATED`, `UNVERIFIED`, `FAILED`.
- Never invent links, labels, join keys, EXIF measurements, run times, model results, test scores, or training success.
- Never silently substitute a dataset/model, invent ISP annotations, or use synthetic labels as human ground truth.
- Paper SPAQ BL/MT-E results and TOPIQ pretrained benchmark numbers are references, not our reproduced results.
- A frozen pretrained TOPIQ inference is not a TOPIQ training reproduction. Verify checkpoint training-data overlap before cross-dataset claims.

## Stage gates
- Phase 0: project inventory, source audit, repo & progress ledger.
- Phase 1: data download, hash, license, schema & join audit.
- Phase 2: minimal executable train/evaluate; real Kaggle smoke benchmark.
- Phase 3: paired E0 image-only vs E1 additive EXIF baseline.
- Phase 4: E2 gated, E3 additive+metadata dropout, E4 gated+dropout with controls.
- Phase 5: robust missingness and capture-condition analysis, statistics.
- Phase 6: polish, packaging, paper-quality report and one-command reproduction.
Do not start expensive later stages before earlier gates pass.

## Experiment science
- Fix split IDs and test set. No hyperparameter tuning or architecture selection on the test set.
- Use train-only fitted imputation/scalers; masked EXIF remains masked and cannot leak missing values through a side channel.
- Use same image encoder, splits, seeds and budgets for causal contrasts. Report exceptions.
- Missingness simulations must be independent of MOS for the primary protocol; define separate conditional missingness stress tests explicitly.
- If stratifying by camera/ISO/night scene, audit group definitions, sample sizes and confounding; do not claim unseen-device or geographic OOD without supported groups.
- Do not assert significance from one run or naive unpaired correlations; use per-image paired bootstrap where appropriate and multiple seeds if affordable.
- Treat improvements as **exploratory until final locked evaluation**. No cherry-picked test runs.

## Implementation standards
- Reuse author's official SPAQ code/checkpoints and pyiqa/TOPIQ where feasible; pin upstream commits and dependencies. Distinguish copied/adapted/licensed code.
- Prioritize vertical slice: manifest -> loader -> split -> model -> 1 batch -> train -> eval -> saved outputs.
- Tests for data join, duplicates, mask semantics, metric calculations, seed reproducibility, and checkpoint resume are mandatory.
- Store raw datasets outside Git; commit manifests, code, tests, configs, summaries and small figures only if allowed.
- No plain-text credentials or secrets. Never echo tokens. Default GitHub remote private.
- Do not delete or overwrite prior raw data, outputs, run history, or remote resources without a verified need and safety check.

## Continuous progress protocol
Before changing code, read `PROJECT_STATUS.md`, `STATE.json`, `TASKS.md`, `EXPERIMENTS.csv`, `SESSION_HANDOFF.md`, and last commits.
After EACH material action: update task status and evidence path. After EACH stage and expensive run: update PROJECT_STATUS, STATE.json, EXPERIMENTS.csv, SESSION_HANDOFF.md, git commit (where safe). Keep all reports honest and specific.
If token/time/credentials are low: checkpoint current state **first**, write a next-command handoff, commit work, then stop gracefully. Do not silently leave partially edited code or launch another costly run.

## User-facing reporting at phase boundaries
Give a concise report: phase, actual completed artifacts, measured results (if any), blockers, current Kaggle run status, next exact command. Do not replace the durable local ledger with chat narration.
