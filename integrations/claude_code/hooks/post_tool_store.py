#!/usr/bin/env python3
"""
Claude Code PostToolUse hook — MnemoCore auto-store
====================================================
Automatically stores the result of significant file edits into MnemoCore.

Configure in ~/.claude/settings.json:
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/post_tool_store.py"
          }
        ]
      }
    ]
  }
}

The hook receives a JSON blob on stdin and exits 0 to allow the tool call.
It stores a lightweight memory entry in the background (non-blocking via
subprocess) so it never delays Claude Code's response.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

BRIDGE = Path(__file__).resolve().parents[2] / "mnemo_bridge.py"
MIN_CONTENT_LEN = 30  # Ignore trivially short edits


def main() -> int:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0  # Never block Claude Code

    tool_name  = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    session_id = data.get("session_id", "")

    # Only act on file-writing tools
    if tool_name not in {"Edit", "Write", "MultiEdit"}:
        return 0

    file_path = tool_input.get("file_path", "")
    new_string = tool_input.get("new_string") or tool_input.get("content", "")

    if not new_string or len(new_string) < MIN_CONTENT_LEN:
        return 0

    # Build a concise memory entry — just the file + a short excerpt
    excerpt = new_string[:200].replace("\n", " ").strip()
    memory_text = f"Modified {file_path}: {excerpt}"

    tags = "claude-code,edit"
    if file_path.endswith(".py"):
        tags += ",python"
    elif file_path.endswith((".ts", ".js")):
        tags += ",javascript"

    ctx = session_id[:16] if session_id else None

    cmd = [
        sys.executable, str(BRIDGE),
        "store", memory_text,
        "--source", "claude-code-hook",
        "--tags", tags,
    ]
    if ctx:
        cmd += ["--ctx", ctx]

    env = {**os.environ}

    # Fire-and-forget: do not block Claude Code
    subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
