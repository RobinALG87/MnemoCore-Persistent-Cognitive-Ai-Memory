#!/usr/bin/env python3
"""
Claude Code PreToolUse hook — MnemoCore context injection
==========================================================
On the FIRST tool call of a session, queries MnemoCore for recent context
and writes it to a temporary file that is referenced from CLAUDE.md.

This gives Claude Code automatic memory of previous sessions WITHOUT
requiring any explicit user commands.

Configure in ~/.claude/settings.json:
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/pre_session_inject.py"
          }
        ]
      }
    ]
  }
}

The hook must exit 0 (allow) or 2 (block with message).
It never blocks — silently degrades if MnemoCore is offline.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BRIDGE      = Path(__file__).resolve().parents[2] / "mnemo_bridge.py"
CONTEXT_DIR = Path(os.getenv("MNEMOCORE_CONTEXT_DIR", Path.home() / ".claude" / "mnemo_context"))
DONE_FILE   = CONTEXT_DIR / ".session_injected"


def main() -> int:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0

    session_id = data.get("session_id", "")

    # Only inject once per session
    done_marker = CONTEXT_DIR / f".injected_{session_id[:16]}"
    if done_marker.exists():
        return 0

    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Query MnemoCore for context
    try:
        result = subprocess.run(
            [sys.executable, str(BRIDGE), "context", "--top-k", "8"],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ},
        )
        context_md = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        context_md = ""

    if context_md:
        context_file = CONTEXT_DIR / "latest_context.md"
        context_file.write_text(context_md, encoding="utf-8")

    # Mark session as injected
    done_marker.touch()

    # Output context as additional system information if available
    if context_md:
        # Claude Code hooks can output JSON to inject content
        output = {
            "type": "system_reminder",
            "content": context_md,
        }
        print(json.dumps(output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
