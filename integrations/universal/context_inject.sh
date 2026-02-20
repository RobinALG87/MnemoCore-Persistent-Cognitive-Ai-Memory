#!/usr/bin/env bash
# context_inject.sh — Universal MnemoCore context provider
# =========================================================
# Outputs MnemoCore memory context as plain text/markdown.
# Pipe or include the output into any tool that accepts system prompts.
#
# Usage:
#   ./context_inject.sh                    # General context
#   ./context_inject.sh "bug fix async"    # Focused query
#   ./context_inject.sh "" 10             # Top-10 results
#
# Examples:
#   codex --system "$(./context_inject.sh)" ...
#   openai-cli --system-prompt "$(./context_inject.sh 'auth')"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_PY="${BRIDGE_PY:-$(realpath "$SCRIPT_DIR/../mnemo_bridge.py")}"

QUERY="${1:-}"
TOP_K="${2:-6}"

ARGS=(context --top-k "$TOP_K")
if [[ -n "$QUERY" ]]; then
    ARGS+=(--query "$QUERY")
fi

python3 "$BRIDGE_PY" "${ARGS[@]}" 2>/dev/null || true
