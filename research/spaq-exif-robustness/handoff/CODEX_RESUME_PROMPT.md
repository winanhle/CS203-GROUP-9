# RESUME PROMPT — for a NEW Codex session after context/token exhaustion

You are CONTINUING an existing end-to-end SPAQ/EXIF NR-IQA research engineering project, **not starting a new one**.

## Mandatory first steps, in order

1. Locate project repository. Read `AGENTS.md` and the original `CODEX_MASTER_PROMPT.md` if available (or handoff copy).
2. Read these durable truth sources in this order:
   - `STATE.json`
   - `PROJECT_STATUS.md`
   - `SESSION_HANDOFF.md`
   - `TASKS.md`
   - `EXPERIMENTS.csv`
   - `EVIDENCE_LOG.md`
   - `DECISIONS.md`
   - `docs/SOURCE_AUDIT.md`, `docs/DATASET_AUDIT.md` as relevant.
3. Inspect actual repository state (`pwd`, `git status`, `git log --oneline -8`) and confirm files/artifacts exist and hashes/paths are consistent.
4. If a Kaggle job is marked RUNNING/QUEUED/UNKNOWN, check **actual** `kaggle kernels status <owner>/<slug>` and logs/output BEFORE submitting a new kernel run. Keep remote run version, owner/slug, artifact paths and checkpoint info. No duplicate expensive run.
5. Identify the most advanced **verified** completed gate (G0–G6), not the most optimistic status text. Reconcile inaccuracies immediately in the state ledger.
6. Execute the **next unfinished task** based on dependencies and gate checks. Preserve E0/E1 baseline artifacts and splits. Do not rewrite from scratch or change the research question without a documented scientific reason.
7. Before this session ends, update all progress/state files, commit safe changes, and write precise next commands in `SESSION_HANDOFF.md`.

## Hard restrictions

- Don't assume training occurred unless logs, predictions, metrics and checkpoints actually exist.
- Never replace original SPAQ metadata with fake labels or fabricate numerical improvements.
- No test-set-driven tuning, no change to frozen splits without an explicit new protocol/version.
- If Kaggle is unauthenticated, mark blocker and continue work possible without it. Don't ask for passwords or tokens in chat.
- Do not push raw datasets/checkpoints/secrets to a public Git repo.
- Don't claim you'll monitor a run beyond the active session.
- Respect 8.5-hour target and use measured timings for remaining runtime estimation.

## Your first reply after inspection (compact)

```text
RECOVERED REPOSITORY: <path>, commit <SHA>
LAST VERIFIED GATE: <G0-G6>
ACTUAL LAST RUN: <run id / kernel version / current status>
ALREADY VERIFIED: <specific outputs and paths>
OPEN BLOCKERS: <actual blockers>
NEXT TASK: <task ID and command>
```

After that, DO the task. **This is an execution request, not a request for a fresh roadmap.**
