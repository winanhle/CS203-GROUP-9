# SPAQ research import manifest

This directory contains the complete tracked research workspace for the SPAQ EXIF-metadata robustness study.

## Source snapshot

- Source repository: `https://github.com/longlth2297/EXIF-Metadata-Help-Smartphone-Image-Quality-Assessment-.git`
- Source branch: `phase7-hypothesis-extension`
- Source commit: `146ff0776d6d3629fa4e8cf0d56fc37546a56a77`
- Imported into the group repository on: 2026-11-12
- Import method: `git archive` of the exact source commit, preserving all 130 tracked files.

The additional `handoff/` directory preserves the 14-file research handoff package from `Codex_SPAQ_Research_Handoff_2026-11-09` (prompts, acceptance criteria, templates, and long-form reports). It is included because several handoff-only files are not tracked in the source repository.

## Start here

1. Read `README.md` for setup and project structure.
2. Read `PROJECT_STATUS.md`, `SESSION_HANDOFF.md`, and `STATE.json` for current status.
3. Read `docs/PHASE7_EXPERIMENT_REPORT.md` and `EVIDENCE_LOG.md` for the latest findings and evidence.
4. Read `LICENSE_AND_DATA_PROVENANCE.md` before downloading or redistributing data.
5. Run `python scripts/validate_release.py` to validate the included release evidence.

Large raw image datasets, model checkpoints, feature tensors, virtual environments, and API credentials are intentionally not committed. Their provenance and reconstruction steps are documented in the repository files.

## Import verification

- All 130 imported Git blobs match the source commit exactly after staging.
- All 14 handoff files match the source handoff package by SHA-256.
- Unit tests in the imported location: `16 passed` on 2026-11-12.
- `scripts/validate_release.py` additionally requires ignored runtime artifacts such as `artifacts/kaggle_download/v2/smoke_checkpoint.pt`. Those large generated files are intentionally not added to the group repository; restore them using the documented provenance before running that validator end to end.
