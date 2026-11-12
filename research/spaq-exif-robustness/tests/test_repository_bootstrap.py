from pathlib import Path


REQUIRED_STATE_FILES = {
    "AGENTS.md",
    "PROJECT_STATUS.md",
    "STATE.json",
    "TASKS.md",
    "DECISIONS.md",
    "SESSION_HANDOFF.md",
    "EXPERIMENTS.csv",
    "EVIDENCE_LOG.md",
}


def test_required_state_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = sorted(name for name in REQUIRED_STATE_FILES if not (root / name).is_file())
    assert not missing, f"Missing persistent state files: {missing}"

