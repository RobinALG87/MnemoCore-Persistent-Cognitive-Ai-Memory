#!/usr/bin/env bash
# MnemoCore Integration Setup
# ============================
# One-command setup for Claude Code, Gemini CLI, Aider, and universal tools.
#
# Usage:
#   ./setup.sh                      # Interactive, choose integrations
#   ./setup.sh --all                # Enable all integrations
#   ./setup.sh --claude-code        # Claude Code only
#   ./setup.sh --gemini             # Gemini CLI only
#   ./setup.sh --aider              # Aider only
#
# Prerequisites:
#   - Python 3.10+ with 'requests' package
#   - MnemoCore running (uvicorn mnemocore.api.main:app --port 8100)
#   - HAIM_API_KEY environment variable set

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNEMOCORE_DIR="$(realpath "$SCRIPT_DIR/..")"
BRIDGE_PY="$SCRIPT_DIR/mnemo_bridge.py"
HOOKS_DIR="$SCRIPT_DIR/claude_code/hooks"

CLAUDE_SETTINGS="$HOME/.claude/settings.json"
CLAUDE_MCP="$HOME/.claude/mcp.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Helpers ────────────────────────────────────────────────────────────────

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

check_python() {
    if ! python3 -c "import requests" &>/dev/null; then
        warn "Python 'requests' not installed. Installing..."
        python3 -m pip install --quiet requests
        success "requests installed"
    fi
}

check_mnemocore() {
    info "Checking MnemoCore connectivity..."
    if python3 "$BRIDGE_PY" health &>/dev/null; then
        success "MnemoCore is online"
        return 0
    else
        warn "MnemoCore is not running. Start it first with:"
        warn "  cd $MNEMOCORE_DIR && uvicorn mnemocore.api.main:app --port 8100"
        return 1
    fi
}

merge_json() {
    # Merge JSON object $2 into file $1 (creates if not exists)
    local target="$1"
    local fragment="$2"

    if [[ ! -f "$target" ]]; then
        echo '{}' > "$target"
    fi

    python3 - <<PYEOF
import json, sys
with open("$target") as f:
    existing = json.load(f)
with open("$fragment") as f:
    new = json.load(f)
# Deep merge (one level)
for k, v in new.items():
    if k.startswith("_"):
        continue
    if k in existing and isinstance(existing[k], dict) and isinstance(v, dict):
        existing[k].update(v)
    else:
        existing[k] = v
with open("$target", "w") as f:
    json.dump(existing, f, indent=2)
print("Merged successfully")
PYEOF
}

# ── Integration: Claude Code ───────────────────────────────────────────────

setup_claude_code() {
    info "Setting up Claude Code integration..."
    mkdir -p "$HOME/.claude/mnemo_context"

    # 1. MCP Server
    info "  Configuring MCP server..."
    local mcp_tmp
    mcp_tmp="$(mktemp /tmp/mnemo_mcp_XXXXXX.json)"
    sed \
        -e "s|\${MNEMOCORE_DIR}|$MNEMOCORE_DIR|g" \
        -e "s|\${HAIM_API_KEY}|${HAIM_API_KEY:-}|g" \
        "$SCRIPT_DIR/claude_code/mcp_config.json" > "$mcp_tmp"

    if [[ ! -f "$CLAUDE_MCP" ]]; then
        echo '{"mcpServers": {}}' > "$CLAUDE_MCP"
    fi

    python3 - "$CLAUDE_MCP" "$mcp_tmp" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    existing = json.load(f)
with open(sys.argv[2]) as f:
    new = json.load(f)
existing.setdefault("mcpServers", {}).update(new.get("mcpServers", {}))
with open(sys.argv[1], "w") as f:
    json.dump(existing, f, indent=2)
PYEOF
    rm -f "$mcp_tmp"
    success "  MCP server registered in $CLAUDE_MCP"

    # 2. Hooks
    info "  Installing hooks in $CLAUDE_SETTINGS..."
    if [[ ! -f "$CLAUDE_SETTINGS" ]]; then
        echo '{}' > "$CLAUDE_SETTINGS"
    fi

    python3 - "$CLAUDE_SETTINGS" "$HOOKS_DIR" <<PYEOF
import json, sys
settings_path = sys.argv[1]
hooks_dir = sys.argv[2]
with open(settings_path) as f:
    settings = json.load(f)

hooks = settings.setdefault("hooks", {})
pre = hooks.setdefault("PreToolUse", [])
post = hooks.setdefault("PostToolUse", [])

pre_hook = {
    "matcher": ".*",
    "hooks": [{"type": "command", "command": f"python3 {hooks_dir}/pre_session_inject.py"}]
}
post_hook = {
    "matcher": "Edit|Write|MultiEdit",
    "hooks": [{"type": "command", "command": f"python3 {hooks_dir}/post_tool_store.py"}]
}

# Only add if not already present
pre_cmds = [h["hooks"][0]["command"] for h in pre if h.get("hooks")]
post_cmds = [h["hooks"][0]["command"] for h in post if h.get("hooks")]

if pre_hook["hooks"][0]["command"] not in pre_cmds:
    pre.append(pre_hook)
if post_hook["hooks"][0]["command"] not in post_cmds:
    post.append(post_hook)

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)
print("Hooks installed")
PYEOF
    success "  Hooks installed in $CLAUDE_SETTINGS"

    # 3. CLAUDE.md snippet — append if not already present
    local clause_md="$MNEMOCORE_DIR/CLAUDE.md"
    local snippet="$SCRIPT_DIR/claude_code/CLAUDE_memory_snippet.md"
    local marker="# MnemoCore — Persistent Cognitive Memory"
    if [[ -f "$clause_md" ]] && grep -qF "$marker" "$clause_md"; then
        info "  CLAUDE.md already contains MnemoCore memory instructions"
    else
        echo "" >> "$clause_md"
        cat "$snippet" >> "$clause_md"
        success "  Memory instructions appended to $clause_md"
    fi

    success "Claude Code integration complete"
}

# ── Integration: Gemini CLI ────────────────────────────────────────────────

setup_gemini() {
    info "Setting up Gemini CLI integration..."

    # Make wrapper executable
    chmod +x "$SCRIPT_DIR/gemini_cli/gemini_wrap.sh"

    # Append to GEMINI.md if it exists
    local gemini_md="$MNEMOCORE_DIR/GEMINI.md"
    local snippet="$SCRIPT_DIR/gemini_cli/GEMINI_memory_snippet.md"
    local marker="# MnemoCore — Persistent Cognitive Memory"
    if [[ -f "$gemini_md" ]] && grep -qF "$marker" "$gemini_md"; then
        info "  GEMINI.md already contains MnemoCore instructions"
    elif [[ -f "$gemini_md" ]]; then
        echo "" >> "$gemini_md"
        cat "$snippet" >> "$gemini_md"
        success "  Memory instructions appended to $gemini_md"
    else
        cp "$snippet" "$gemini_md"
        success "  Created $gemini_md with memory instructions"
    fi

    success "Gemini CLI integration complete"
    info "  Use: $SCRIPT_DIR/gemini_cli/gemini_wrap.sh [args] instead of 'gemini'"
    info "  Or alias: alias gemini='$SCRIPT_DIR/gemini_cli/gemini_wrap.sh'"
}

# ── Integration: Aider ─────────────────────────────────────────────────────

setup_aider() {
    info "Setting up Aider integration..."
    chmod +x "$SCRIPT_DIR/aider/aider_wrap.sh"

    # Write .env fragment for aider
    local aider_env="$MNEMOCORE_DIR/.aider.env"
    cat > "$aider_env" <<EOF
# MnemoCore environment for Aider
export MNEMOCORE_URL="${MNEMOCORE_URL:-http://localhost:8100}"
export HAIM_API_KEY="${HAIM_API_KEY:-}"
export BRIDGE_PY="$BRIDGE_PY"
EOF
    success "Aider integration complete"
    info "  Use: $SCRIPT_DIR/aider/aider_wrap.sh [args] instead of 'aider'"
    info "  Or alias: alias aider='$SCRIPT_DIR/aider/aider_wrap.sh'"
}

# ── Integration: Universal scripts ─────────────────────────────────────────

setup_universal() {
    chmod +x "$SCRIPT_DIR/universal/context_inject.sh"
    chmod +x "$SCRIPT_DIR/universal/store_session.sh"
    success "Universal scripts ready"
    info "  Context: $SCRIPT_DIR/universal/context_inject.sh [query] [top-k]"
    info "  Store:   $SCRIPT_DIR/universal/store_session.sh [text] [tags] [ctx]"
}

# ── Main ───────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          MnemoCore Integration Setup                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
check_python
check_mnemocore || true  # Non-fatal — offline check is a warning

DO_ALL=false
DO_CLAUDE=false
DO_GEMINI=false
DO_AIDER=false

for arg in "$@"; do
    case "$arg" in
        --all)         DO_ALL=true ;;
        --claude-code) DO_CLAUDE=true ;;
        --gemini)      DO_GEMINI=true ;;
        --aider)       DO_AIDER=true ;;
    esac
done

if ! $DO_ALL && ! $DO_CLAUDE && ! $DO_GEMINI && ! $DO_AIDER; then
    echo "Which integrations do you want to enable?"
    echo "  1) Claude Code (MCP + hooks + CLAUDE.md)"
    echo "  2) Gemini CLI (GEMINI.md + wrapper)"
    echo "  3) Aider (wrapper script)"
    echo "  4) All of the above"
    echo ""
    read -rp "Enter choice(s) [e.g. 1 3 or 4]: " CHOICES

    for c in $CHOICES; do
        case "$c" in
            1) DO_CLAUDE=true ;;
            2) DO_GEMINI=true ;;
            3) DO_AIDER=true ;;
            4) DO_ALL=true ;;
        esac
    done
fi

if $DO_ALL; then
    DO_CLAUDE=true; DO_GEMINI=true; DO_AIDER=true
fi

echo ""
setup_universal

$DO_CLAUDE && setup_claude_code
$DO_GEMINI && setup_gemini
$DO_AIDER  && setup_aider

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Setup complete! Quick start:                      ║"
echo "║                                                      ║"
echo "║   Test bridge:  python3 integrations/mnemo_bridge.py health"
echo "║   Get context:  integrations/universal/context_inject.sh"
echo "║   Store memory: integrations/universal/store_session.sh"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
