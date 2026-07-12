"""Smoke checks for the dependency-free AgentMemory surface."""

from __future__ import annotations

import subprocess
import sys
from os import environ
from pathlib import Path


def test_agent_memory_import_does_not_load_optional_integrations() -> None:
    code = (
        "import sys; import mnemocore.agent_memory; "
        "assert not any(name in sys.modules for name in "
        "('fastapi', 'uvicorn', 'redis', 'qdrant_client', 'numpy'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={
            **environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        },
    )
    assert result.returncode == 0, result.stderr
