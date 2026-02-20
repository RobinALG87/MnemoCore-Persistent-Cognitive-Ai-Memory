#!/usr/bin/env bash
# store_session.sh — Store session outcomes into MnemoCore
# =========================================================
# Call this at the end of an AI coding session to persist key findings.
#
# Usage (interactive):
#   ./store_session.sh
#
# Usage (non-interactive / scripted):
#   ./store_session.sh "Fixed race condition in tier_manager.py" "bugfix,async" "my-project"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_PY="${BRIDGE_PY:-$(realpath "$SCRIPT_DIR/../mnemo_bridge.py")}"

if [[ $# -ge 1 ]]; then
    CONTENT="$1"
    TAGS="${2:-cli}"
    CTX="${3:-}"
else
    echo "Enter memory content (what was done/decided/fixed):"
    read -r CONTENT
    echo "Tags (comma-separated, e.g. bugfix,python,auth):"
    read -r TAGS
    echo "Context/project ID (optional, press Enter to skip):"
    read -r CTX
fi

if [[ -z "$CONTENT" ]]; then
    echo "No content provided, nothing stored." >&2
    exit 0
fi

ARGS=(store "$CONTENT" --source "manual-session" --tags "$TAGS")
if [[ -n "$CTX" ]]; then
    ARGS+=(--ctx "$CTX")
fi

python3 "$BRIDGE_PY" "${ARGS[@]}"
