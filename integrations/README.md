# MnemoCore Integrations

Connect MnemoCore's persistent cognitive memory to your AI coding tools.

## Supported tools

| Tool | Method | Notes |
|------|--------|-------|
| **Claude Code** | MCP server + hooks + CLAUDE.md | Best integration — native tool access |
| **Gemini CLI** | GEMINI.md + wrapper script | Context injected at session start |
| **Aider** | Wrapper script (`--system-prompt`) | Context injected at session start |
| **Any CLI tool** | Universal shell scripts | Pipe context into any tool |
| **Open-source agents** | REST API (`mnemo_bridge.py`) | Minimal Python dependency |

---

## Quick setup

### Prerequisites

1. MnemoCore running: `uvicorn mnemocore.api.main:app --port 8100`
2. `HAIM_API_KEY` environment variable set
3. Python 3.10+ with `requests` (`pip install requests`)

### Linux / macOS

```bash
cd integrations/
bash setup.sh --all
```

### Windows (PowerShell)

```powershell
cd integrations\
.\setup.ps1 -All
```

### Manual: test the bridge first

```bash
export MNEMOCORE_URL=http://localhost:8100
export HAIM_API_KEY=your-key-here

python integrations/mnemo_bridge.py health
python integrations/mnemo_bridge.py context --top-k 5
python integrations/mnemo_bridge.py store "Fixed import error in engine.py" --tags "bugfix,python"
```

---

## Claude Code (recommended)

The setup script does three things:

### 1. MCP server — native tool access

Registers MnemoCore as an MCP server in `~/.claude/mcp.json`.
Claude Code gets four tools:
- `memory_query` — search memories
- `memory_store` — store a memory
- `memory_get` / `memory_delete` — manage individual memories

Claude will use these automatically when you instruct it to remember things,
or you can configure CLAUDE.md to trigger them on every session (see below).

**Verify:** Run `claude mcp list` — you should see `mnemocore` listed.

### 2. Hooks — automatic background storage

Two hooks are installed in `~/.claude/settings.json`:

- **PreToolUse** (`pre_session_inject.py`): On the first tool call of a session,
  queries MnemoCore and injects recent context into Claude's awareness.

- **PostToolUse** (`post_tool_store.py`): After every `Edit`/`Write` call,
  stores a lightweight memory entry in the background (non-blocking).

Hooks never block Claude Code — they degrade silently if MnemoCore is offline.

### 3. CLAUDE.md — behavioral instructions

The setup appends memory usage instructions to `CLAUDE.md`.
This tells Claude *when* to use memory tools proactively.

---

## Gemini CLI

```bash
# Option A: Use wrapper (injects context automatically)
alias gemini='bash integrations/gemini_cli/gemini_wrap.sh'
gemini "Fix the async bug in engine.py"

# Option B: Manual context injection
CONTEXT=$(integrations/universal/context_inject.sh)
gemini --system-prompt "$CONTEXT" "Fix the async bug in engine.py"
```

Also add instructions to your `GEMINI.md`:
```bash
cat integrations/gemini_cli/GEMINI_memory_snippet.md >> GEMINI.md
```

---

## Aider

```bash
# Option A: Use wrapper
alias aider='bash integrations/aider/aider_wrap.sh'
aider --model claude-3-5-sonnet-20241022 engine.py

# Option B: Manual
CONTEXT=$(integrations/universal/context_inject.sh "async engine")
aider --system-prompt "$CONTEXT" engine.py
```

---

## Universal / Open-source agents

Any tool that accepts a system prompt can use MnemoCore:

```bash
# Get context as markdown
integrations/universal/context_inject.sh "query text" 6

# Use in any command
MY_CONTEXT=$(integrations/universal/context_inject.sh)
some-ai-cli --system "$MY_CONTEXT" "do the task"

# Store a memory after a session
integrations/universal/store_session.sh \
  "Discovered that warm tier mmap files grow unbounded without consolidation" \
  "discovery,warm-tier,storage" \
  "mnemocore-project"
```

### REST API (Python / any language)

```python
import os, requests

BASE = os.getenv("MNEMOCORE_URL", "http://localhost:8100")
KEY  = os.getenv("HAIM_API_KEY", "")
HDR  = {"X-API-Key": KEY}

# Query
r = requests.post(f"{BASE}/query", json={"query": "async bugs", "top_k": 5}, headers=HDR)
for m in r.json()["results"]:
    print(m["score"], m["content"])

# Store
requests.post(f"{BASE}/store", json={
    "content": "Found root cause of memory leak in consolidation worker",
    "metadata": {"source": "my-agent", "tags": ["bugfix", "memory"]}
}, headers=HDR)
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOCORE_URL` | `http://localhost:8100` | MnemoCore API base URL |
| `HAIM_API_KEY` | — | API key (same as MnemoCore's `HAIM_API_KEY`) |
| `MNEMOCORE_TIMEOUT` | `5` | Request timeout in seconds |
| `MNEMOCORE_CONTEXT_DIR` | `~/.claude/mnemo_context` | Where hook writes context files |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  AI Coding Tool                      │
│   (Claude Code / Gemini CLI / Aider / Custom)       │
└──────────────┬──────────────────────┬───────────────┘
               │ MCP tools            │ System prompt
               │ (Claude Code only)   │ (all tools)
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────┐
│  mnemocore MCP server│  │    mnemo_bridge.py CLI     │
│  (stdio transport)   │  │    (lightweight wrapper)   │
└──────────┬───────────┘  └─────────────┬─────────────┘
           │                            │
           └────────────┬───────────────┘
                        │ HTTP REST
                        ▼
           ┌────────────────────────┐
           │   MnemoCore API        │
           │   localhost:8100       │
           │                        │
           │  ┌──────────────────┐  │
           │  │  HAIMEngine      │  │
           │  │  HOT/WARM/COLD   │  │
           │  │  HDV vectors     │  │
           │  └──────────────────┘  │
           └────────────────────────┘
```

---

## Troubleshooting

**MnemoCore offline:**
```bash
python integrations/mnemo_bridge.py health
# → MnemoCore is OFFLINE
# Start it: uvicorn mnemocore.api.main:app --port 8100
```

**API key error (401):**
```bash
export HAIM_API_KEY="your-key-from-.env"
python integrations/mnemo_bridge.py health
```

**Hook not triggering (Claude Code):**
```bash
# Check settings.json
cat ~/.claude/settings.json | python -m json.tool | grep -A5 hooks
```

**MCP server not found (Claude Code):**
```bash
# Verify mcp.json
cat ~/.claude/mcp.json
# Check PYTHONPATH includes src/
cd /path/to/mnemocore && python -m mnemocore.mcp.server --help
```
