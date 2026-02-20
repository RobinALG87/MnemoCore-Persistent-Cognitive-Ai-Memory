#!/usr/bin/env bash
# gemini_wrap.sh — MnemoCore context injector for Gemini CLI
# =============================================================
# Usage: ./gemini_wrap.sh [any gemini CLI args...]
#
# Fetches recent MnemoCore context and prepends it to the system prompt
# via a temporary file, then delegates to the real `gemini` binary.
#
# Environment variables:
#   MNEMOCORE_URL      MnemoCore REST URL (default: http://localhost:8100)
#   HAIM_API_KEY       API key for MnemoCore
#   BRIDGE_PY          Path to mnemo_bridge.py (auto-detected)
#   GEMINI_BIN         Path to gemini binary (default: gemini)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_PY="${BRIDGE_PY:-$(realpath "$SCRIPT_DIR/../mnemo_bridge.py")}"
GEMINI_BIN="${GEMINI_BIN:-gemini}"
MNEMOCORE_URL="${MNEMOCORE_URL:-http://localhost:8100}"

# ── Fetch context (silently degrade if offline) ────────────────────────────
CONTEXT=""
if python3 "$BRIDGE_PY" health &>/dev/null; then
    CONTEXT="$(python3 "$BRIDGE_PY" context --top-k 6 2>/dev/null || true)"
fi

# ── Build the injected system prompt fragment ──────────────────────────────
if [[ -n "$CONTEXT" ]]; then
    MEMORY_FILE="$(mktemp /tmp/mnemo_context_XXXXXX.md)"
    trap 'rm -f "$MEMORY_FILE"' EXIT

    cat > "$MEMORY_FILE" <<'HEREDOC'
## Persistent Memory Context (from MnemoCore)

The following is relevant context from your memory of previous sessions.
Use it to avoid re-discovering known patterns, bugs, and decisions.

HEREDOC
    echo "$CONTEXT" >> "$MEMORY_FILE"

    # Gemini CLI supports --system-prompt-file or similar flags.
    # Adjust this to match the actual Gemini CLI interface.
    exec "$GEMINI_BIN" --system-prompt-file "$MEMORY_FILE" "$@"
else
    exec "$GEMINI_BIN" "$@"
fi
