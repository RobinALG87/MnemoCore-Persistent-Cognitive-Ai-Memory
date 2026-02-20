#!/usr/bin/env bash
# aider_wrap.sh — MnemoCore context injector for Aider
# ======================================================
# Usage: ./aider_wrap.sh [any aider args...]
#
# Injects MnemoCore memory context into Aider's system prompt
# using the --system-prompt flag (available in Aider 0.40+).
#
# Environment variables:
#   MNEMOCORE_URL      MnemoCore REST URL (default: http://localhost:8100)
#   HAIM_API_KEY       API key for MnemoCore
#   BRIDGE_PY          Path to mnemo_bridge.py (auto-detected)
#   AIDER_BIN          Path to aider binary (default: aider)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_PY="${BRIDGE_PY:-$(realpath "$SCRIPT_DIR/../mnemo_bridge.py")}"
AIDER_BIN="${AIDER_BIN:-aider}"

# ── Fetch context ──────────────────────────────────────────────────────────
CONTEXT=""
if python3 "$BRIDGE_PY" health &>/dev/null 2>&1; then
    CONTEXT="$(python3 "$BRIDGE_PY" context --top-k 6 2>/dev/null || true)"
fi

# ── Run Aider with or without injected context ─────────────────────────────
if [[ -n "$CONTEXT" ]]; then
    PROMPT_FILE="$(mktemp /tmp/mnemo_aider_XXXXXX.md)"
    trap 'rm -f "$PROMPT_FILE"' EXIT

    cat > "$PROMPT_FILE" <<'HEREDOC'
## MnemoCore: Memory from previous sessions

Use the following context from persistent memory to inform your work.
Do not repeat known decisions. Reference this to avoid re-discovering bugs.

HEREDOC
    echo "$CONTEXT" >> "$PROMPT_FILE"

    exec "$AIDER_BIN" --system-prompt "$PROMPT_FILE" "$@"
else
    exec "$AIDER_BIN" "$@"
fi
