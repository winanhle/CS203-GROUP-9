# Acceptance audit

**Audited UTC:** 2026-11-09  
**Policy:** `EXECUTION_VERIFIED` requires persisted artifacts or command results; externally blocked work remains blocked.

## G0 — source audit and bootstrap: PASS

- Repository and persistent ledgers exist; primary sources, pinned revisions, and findings are documented.
- Raw data, credentials, checkpoints, caches, predictions, and Kaggle account metadata are ignored.

## G1 — data and provenance: PASS

- Both archives were hash-verified; URLs, revisions, sizes, contents, and license cautions are persisted.
- Authentic workbooks and embedded EXIF were inspected; 11,125/11,125 rows join exactly.
- Observed versus estimated metadata is explicit; no EXIF or MOS is fabricated.
- All 1,069 repeated-content candidate edges stay within partitions.

## G2 — vertical slice and smoke: PASS

- Local real-image forward/backward, checkpoint/reload/resume, tiny overfit, timing, VRAM, and tests pass.
- Kaggle v2 completed on Tesla T4 with verified input sample hash, 300-image inference, 20 train steps, VRAM, checkpoint reload/resume, output retrieval, and artifact hashes.
- Measured extraction plus E0-E4 is 157.5 seconds locally; Kaggle projects 11,125-image inference at 493.65 seconds including 20% contingency.

## G3 — matched baselines: PASS

- E0/E1 completed with identical rows, cache, split, seed, budget, validation selection, and test policy.
- Checkpoints, configs, logs, predictions, hashes, metrics, and paired uncertainty are persisted.

## G4 — controlled extensions: PASS

- E2 changes only fusion and documents 8,704 versus 16 metadata parameters.
- E3/E4 change only whole-record dropout; independent fixed test masks preserve partition logic.
- Iteration documents record hypothesis, config difference, cost, result, and interpretation. Negative findings remain.

## G5 — analysis: PASS

- Matched 2,228-row predictions provide SRCC/PLCC/MAE and 10,000-resample paired intervals.
- Missingness uses identical IDs at 0/25/50/100% with fixed seeds.
- Train-defined subgroup sample sizes and deterministic error selection are reported.
- Seven figures trace to hashes in `evidence/analysis_summary_s20261109.json`.

## G6 — deliverable: PASS

- README and experiment report contain setup, reproduction, results, provenance, figures, citations, caveats, and evidence paths.
- Tests and final audit commands are rerun before handoff.
- GitHub `origin` is configured and `main` was pushed successfully; the upstream branch matches the verified local commit.
- Kaggle summary, log, and checkpoint were downloaded and hash-verified.

## Minimum viable research: PASS

E0/E1/E3 and E2/E4 were executed on one frozen split with real predictions, metrics, missingness stress, paired uncertainty, and explicit limitations.
