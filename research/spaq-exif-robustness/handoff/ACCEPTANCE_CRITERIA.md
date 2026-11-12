# Acceptance Criteria — research output must be genuinely runnable

## G0 — Source audit and project bootstrap
- [ ] `git status` clean after initial commit; repository private if pushed.
- [ ] `README.md`, `AGENTS.md`, `PROJECT_STATUS.md`, `STATE.json`, `TASKS.md`, `DECISIONS.md`, `SESSION_HANDOFF.md`, `EXPERIMENTS.csv`, `EVIDENCE_LOG.md` exist.
- [ ] Official SPAQ paper/README, TOPIQ paper/pyiqa docs linked with retrieval timestamp and pinned upstream commits.
- [ ] Report has precise statement of what was *read*, what was *run*, and whether official training scripts exist.

## G1 — Dataset and provenance gate
- [ ] Dataset links are real and terms/license evaluated. Original download URL, exact filenames, hash (where feasible), file sizes, access timestamp recorded.
- [ ] Images and annotation files actually downloaded or attached; paths exist in execution environment.
- [ ] Row-level join on verified keys; report coverage, duplicates, unmatched records, MOS range, EXIF columns/types/missingness.
- [ ] No imputed metadata silently represented as actual measured EXIF.
- [ ] Data split manifest frozen with seed(s); near-duplicate and same-scene risk audited.
- [ ] Raw data, Kaggle credential, and restricted checkpoints are ignored by Git.

## G2 — Vertical slice, local/CPU sanity and Kaggle smoke gate
- [ ] One batch forward + backward succeeds; deterministic tiny subset can overfit or expected sanity checks pass.
- [ ] Unit/integration tests pass; metrics checked against known toy arrays; missingness masks tested.
- [ ] Actual Kaggle run recorded, or documented BLOCKED with CLI/auth/error evidence.
- [ ] Kaggle smoke logs: GPU identity/VRAM, installed versions, disk free, 300-sample IO/inference where possible, 20 train steps, max VRAM, projected runtime.
- [ ] Runtime projection includes **every** training variant, seeds, evaluation, I/O and 15–20% buffer within 8.5h. Otherwise document reduced plan.

## G3 — Baseline end-to-end
- [ ] At minimum E0 and E1 train on identical frozen split and report real metrics.
- [ ] Each successful run has `config.yaml`/JSON, source commit SHA, seed, split hash, model weight hash, environment info, checkpoint, per-image predictions, logs.
- [ ] Checkpoint resume tested. Evaluation uses unchanged test set with no tuning on test.
- [ ] Paper-reported numbers clearly separated from observed project results.

## G4 — Controlled extensions
- [ ] E2 compares gated vs additive with same data/encoder/budget; parameter-count difference stated.
- [ ] E3/E4 test metadata dropout; missingness corruption applied **only under correct partition logic** and evaluated in predeclared conditions.
- [ ] Each new iteration has a hypothesis, config diff, measured contrast, results, cost, interpretation.
- [ ] No extrapolated improvement marketed as actual improvement.

## G5 — Analysis
- [ ] SRCC/PLCC/MAE; confidence intervals or clear uncertainty, matched predictions, subgroup sample sizes.
- [ ] Missingness curve at 0, 25, 50, 100% on same test image IDs and fixed masking seeds.
- [ ] Confirmatory primary contrasts fixed before final test; error cases analyzed without cherry-picking.
- [ ] Cross-dataset claims only where label scale, checkpoint overlap, input availability and protocol genuinely permit comparison.

## G6 — Deliverable
- [ ] README with problem, citations, setup, data acquisition, Kaggle steps, reproduction commands, source provenance, caveats, verified experiment table.
- [ ] Results report with figures, raw evidence references, limits and negative findings.
- [ ] Test commands and download instructions run in clean environment, or limitations explicitly documented.
- [ ] `PROJECT_STATUS.md`, `SESSION_HANDOFF.md`, `STATE.json`, `EXPERIMENTS.csv` up to date and consistent with git status/remote Kaggle state.
- [ ] Human-readable next steps for new Codex session even if full goal incomplete.

## MVR (minimum viable research)
E0 image-only + E1 additive EXIF + E3 additive with metadata dropout, on identical fixed split with real per-image predictions, SRCC/PLCC/MAE and missingness robustness analysis. E2/E4 and TOPIQ comparisons are high-priority extensions but **not** prerequisites for first meaningful end-to-end result.
