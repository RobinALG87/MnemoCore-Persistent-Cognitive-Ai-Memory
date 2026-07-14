"""Contract for the standalone v3 AgentMemory release smoke script."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ops" / "agent_memory_release_smoke.py"


def test_v3_release_smoke_runs_with_source_checkout() -> None:
    """The release workflow's local-only smoke must validate version 3.0.0."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "AgentMemory release smoke passed for 3.0.0" in result.stdout
