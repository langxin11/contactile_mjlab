"""train.py / play.py wrappers should expose mjlab entry points with TASK_ID registered."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_script_help_lists_task():
    """train.py must register TASK_ID before delegating to mjlab.scripts.train.

    Invoke with a bogus task name; tyro's "Invalid choice" error enumerates all
    registered task IDs, so a successful side-effect import will surface our
    TASK_ID in the combined output.
    """
    result = subprocess.run(
        ["uv", "run", "python", "scripts/train.py", "__NONEXISTENT_TASK__", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert "Mjlab-TactileGrasp-Robotiq2F85" in combined, combined


def test_old_train_ppo_removed():
    """Legacy hand-rolled trainer must be deleted."""
    assert not (REPO_ROOT / "scripts" / "train_ppo.py").exists()
