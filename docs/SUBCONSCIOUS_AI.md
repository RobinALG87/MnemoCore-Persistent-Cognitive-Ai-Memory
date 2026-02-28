# Subconscious AI Worker — Comprehensive Setup & Reference Guide

> **MnemoCore v5.0.0+ · Phase 4.4**
>
> The Subconscious AI Worker (`SubconsciousAIWorker`) is an optional background service that connects
> MnemoCore to a local or remote Large Language Model. It periodically wakes up ("pulses"), reads a
> sample of the memory store, and asks the LLM to suggest improvements — memory re-ranking, semantic
> clustering, associative dreaming, and micro-self-improvement notes. All writes are gated by a
> `dry_run` flag so you can audit every suggestion before it is applied.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Quick-Start: Ollama + Phi 3.5 in 5 Minutes](#3-quick-start-ollama--phi-35-in-5-minutes)
4. [Supported Providers](#4-supported-providers)
5. [Ollama Model Selection Guide](#5-ollama-model-selection-guide)
6. [Full `config.yaml` Reference](#6-full-configyaml-reference)
7. [Environment Variable Reference](#7-environment-variable-reference)
8. [Docker / Docker-Compose Setup](#8-docker--docker-compose-setup)
9. [Runtime Operations Explained](#9-runtime-operations-explained)
10. [Audit Trail Format](#10-audit-trail-format)
11. [dry_run Mode](#11-dry_run-mode)
12. [Monitoring & Log Patterns](#12-monitoring--log-patterns)
13. [Troubleshooting](#13-troubleshooting)
14. [Security Notes](#14-security-notes)
15. [Extending: Adding a New Provider](#15-extending-adding-a-new-provider)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      MnemoCore Runtime                          │
│                                                                 │
│  ┌──────────────┐   pulse()    ┌──────────────────────────┐    │
│  │ Subconscious │─────────────►│                          │    │
│  │ AI Worker    │              │   Memory Store (FAISS /  │    │
│  │              │◄─────────────│   Qdrant / in-memory)    │    │
│  │  round-robin │  suggestions │                          │    │
│  │  operations  │              └──────────────────────────┘    │
│  └──────┬───────┘                                              │
│         │ HTTP (aiohttp)                                        │
└─────────┼───────────────────────────────────────────────────────┘
          │
    ┌─────▼──────────────────────────────────────────────────┐
    │            Model Client (selected by provider)          │
    │                                                         │
    │  OllamaClient   ──► POST /api/generate                  │
    │  LMStudioClient ──► POST /v1/chat/completions           │
    │  APIClient      ──► POST /v1/chat/completions (OpenAI)  │
    │                 ──► POST /v1/messages       (Anthropic) │
    └─────────────────────────────────────────────────────────┘
```

The worker lives in `src/mnemocore/core/subconscious_ai.py`. It is started automatically when
`mnemocore` boots if `subconscious_ai.enabled` is `true` in `config.yaml` (or the matching
environment variable is set).

A `ResourceGuard` prevents the pulse from running when CPU usage exceeds `max_cpu_percent` and
enforces a maximum number of LLM calls per hour (`rate_limit_per_hour`). This keeps the worker from
interfering with normal request processing.

---

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.12 recommended |
| `aiohttp>=3.9.0` | HTTP client used by all model clients |
| `psutil>=5.9.0` | CPU-usage monitoring in `ResourceGuard` |
| One of the supported providers running | Local (Ollama / LM Studio) or remote API |

Both `aiohttp` and `psutil` are included in `requirements.txt` as of v5.0.0. Install all
dependencies with:

```bash
pip install -r requirements.txt
```

---

## 3. Quick-Start: Ollama + Phi 3.5 in 5 Minutes

This is the **recommended path** for developers who want to try the feature locally.
Phi 3.5 (Microsoft) is a 3.8 B-parameter model that runs comfortably on CPU-only machines.

### Step 1 — Install Ollama

| Platform | Command |
|---|---|
| macOS | `brew install ollama` |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Windows | Download installer from [ollama.com](https://ollama.com) |

Verify the daemon is running:

```bash
ollama --version          # should print a version string
ollama list               # lists downloaded models
```

On Windows, Ollama runs as a system tray application and starts automatically. The API server
listens on `http://localhost:11434` by default.

### Step 2 — Pull the Model

```bash
ollama pull phi3.5
```

This downloads approximately 2.5 GB. Confirm it landed:

```bash
ollama list
# NAME              ID              SIZE      MODIFIED
# phi3.5:latest     ...             2.5 GB    ...
```

### Step 3 — Enable the Worker in `config.yaml`

Open `config.yaml` at the project root. The relevant section:

```yaml
subconscious_ai:
  enabled: true          # <-- change from false if needed
  model_provider: "ollama"
  model_name: "phi3.5:latest"
  model_url: "http://localhost:11434"
  dry_run: false
```

Everything else can stay at the defaults for the first run.

### Step 4 — Start MnemoCore

```bash
uvicorn src.mnemocore.api.app:app --reload
# or
python -m mnemocore
```

Look for this log line to confirm the worker started:

```
INFO  SubconsciousAIWorker  starting — provider=ollama model=phi3.5:latest
```

### Step 5 — Verify the Audit Trail

After 1–2 minutes (`pulse_interval_seconds` default: 120) you should see entries in:

```
./data/subconscious_audit.jsonl
```

Each line is a JSON object describing one AI suggestion. See [Section 10](#10-audit-trail-format).

---

## 4. Supported Providers

| Provider | `model_provider` value | Default URL | Protocol |
|---|---|---|---|
| Ollama | `"ollama"` | `http://localhost:11434` | `/api/generate` |
| LM Studio | `"lm_studio"` | `http://localhost:1234` | OpenAI-compatible `/v1/chat/completions` |
| OpenAI | `"openai"` | `https://api.openai.com` | `/v1/chat/completions` |
| Anthropic | `"anthropic"` | `https://api.anthropic.com` | `/v1/messages` |

### Ollama

Free, local, no API key needed. Install any model with `ollama pull <name>`. The worker sends a
raw `prompt` string and reads back the `response` field.

### LM Studio

Free local inference GUI. Once you load a model and start the local server (default port 1234),
LM Studio exposes an OpenAI-compatible API. Set:

```yaml
subconscious_ai:
  model_provider: "lm_studio"
  model_url: "http://localhost:1234"
  model_name: "your-loaded-model-name"   # must match the model loaded in LM Studio
```

No `api_key` required.

### OpenAI

```yaml
subconscious_ai:
  model_provider: "openai"
  model_url: "https://api.openai.com"
  model_name: "gpt-4o-mini"
```

Set the API key via environment variable — **never hard-code it**:

```bash
export SUBCONSCIOUS_AI_API_KEY="sk-..."
```

Recommended models for this use case: `gpt-4o-mini` (cheap, fast), `gpt-4o` (higher quality).

### Anthropic

```yaml
subconscious_ai:
  model_provider: "anthropic"
  model_url: "https://api.anthropic.com"
  model_name: "claude-3-haiku-20240307"
```

```bash
export SUBCONSCIOUS_AI_API_KEY="sk-ant-..."
```

Recommended: `claude-3-haiku-20240307` (fast, cheap), `claude-3-5-sonnet-20241022` (higher quality).

---

## 5. Ollama Model Selection Guide

All models below are open-weight and run fully locally with no API key or internet connection after
the initial download.

| Model | Pull Command | Download Size | Min RAM (CPU) | Notes |
|---|---|---|---|---|
| **phi3.5** ⭐ recommended | `ollama pull phi3.5` | ~2.5 GB | 6 GB | Microsoft. Excellent reasoning at 3.8 B params. Default choice. |
| llama3.2:3b | `ollama pull llama3.2:3b` | ~2.0 GB | 5 GB | Meta Llama 3.2. Fast, general purpose. |
| mistral:7b | `ollama pull mistral:7b` | ~4.1 GB | 10 GB | Strong at structured output. Needs more RAM. |
| gemma3:4b | `ollama pull gemma3:4b` | ~3.3 GB | 8 GB | Google Gemma 3. Good instruction following. |
| qwen2.5:3b | `ollama pull qwen2.5:3b` | ~2.0 GB | 5 GB | Alibaba Qwen 2.5. Very efficient. |
| llama3.1:8b | `ollama pull llama3.1:8b` | ~4.7 GB | 12 GB | Higher quality at cost of more RAM. |

**GPU acceleration:** If you have an NVIDIA GPU with CUDA or an Apple Silicon Mac, Ollama
automatically uses it. GPU inference is 5–20× faster than CPU. No configuration change needed.

**Switching models at runtime:** Simply change `model_name` in `config.yaml` (or the env var) and
restart MnemoCore. The model must already be pulled with `ollama pull`.

**Checking available models:**

```bash
ollama list
```

**Removing a model to free disk space:**

```bash
ollama rm mistral:7b
```

---

## 6. Full `config.yaml` Reference

```yaml
subconscious_ai:
  # ── Master switch ──────────────────────────────────────────────────────────
  enabled: true
  # Set to false to completely disable the background worker. No LLM calls will
  # be made and no audit records will be written.

  # ── Model provider ─────────────────────────────────────────────────────────
  model_provider: "ollama"
  # Values: "ollama" | "lm_studio" | "openai" | "anthropic"

  model_name: "phi3.5:latest"
  # The model identifier string passed to the provider.
  # Ollama examples  : "phi3.5:latest", "llama3.2:3b", "mistral:7b"
  # LM Studio example: match the model name shown in the LM Studio UI
  # OpenAI examples  : "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
  # Anthropic examples: "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"

  model_url: "http://localhost:11434"
  # Base URL of the model server (no trailing slash).
  # Ollama default  : http://localhost:11434
  # LM Studio default: http://localhost:1234
  # OpenAI           : https://api.openai.com
  # Anthropic        : https://api.anthropic.com
  # Docker note: if Ollama runs on the HOST machine and MnemoCore runs inside a
  # container, change localhost to host.docker.internal, e.g.:
  #   http://host.docker.internal:11434

  api_key: ""
  # Leave empty for local providers (Ollama, LM Studio).
  # For OpenAI / Anthropic: set via environment variable SUBCONSCIOUS_AI_API_KEY
  # instead of hard-coding it here.

  # ── Timing & resource limits ───────────────────────────────────────────────
  pulse_interval_seconds: 120
  # How often the worker wakes up to run one operation.
  # Minimum: 30. Recommended: 120–600 for production.

  rate_limit_per_hour: 30
  # Maximum number of LLM API calls per hour. Prevents runaway costs on paid
  # APIs and reduces background CPU load. Default: 30 calls/hour.

  max_cpu_percent: 30.0
  # If system CPU usage is above this percentage when the pulse fires, the
  # pulse is skipped entirely. Protects request-serving latency.

  # ── Operation flags ────────────────────────────────────────────────────────
  enable_memory_sorting: true
  # Each pulse cycle 1: ask the LLM to suggest a new importance ranking for a
  # sample of recent memories.

  enable_enhanced_dreaming: true
  # Each pulse cycle 2: ask the LLM to find associative / semantic connections
  # between memory fragments and surface new "dream" associations.

  enable_micro_self_improvement: false
  # Each pulse cycle 3: ask the LLM to propose small self-improvement notes for
  # the memory system itself. Disabled by default — enable only if you understand
  # what it does. See docs/SELF_IMPROVEMENT_DEEP_DIVE.md.

  # ── Write control ──────────────────────────────────────────────────────────
  dry_run: false
  # When true: the worker generates suggestions but DOES NOT write them back to
  # the memory store. All suggestions are still recorded in the audit trail.
  # Set to true when evaluating a new model before committing to live writes.

  # ── Audit ──────────────────────────────────────────────────────────────────
  audit_log_path: "data/subconscious_audit.jsonl"
  # Path to the JSONL audit log (relative to project root).
  # Each line records one full LLM interaction: operation, prompt, response,
  # timestamp, whether the suggestion was applied, and the model metadata.
```

---

## 7. Environment Variable Reference

Every setting in `config.yaml` can be overridden by an environment variable. Environment variables
take precedence over `config.yaml` values. This is useful for Docker deployments, CI/CD pipelines,
and per-environment secrets.

| Environment Variable | Overrides | Type | Example |
|---|---|---|---|
| `SUBCONSCIOUS_AI_ENABLED` | `enabled` | bool (`true`/`false`) | `true` |
| `SUBCONSCIOUS_AI_MODEL_PROVIDER` | `model_provider` | string | `ollama` |
| `SUBCONSCIOUS_AI_MODEL_NAME` | `model_name` | string | `phi3.5:latest` |
| `SUBCONSCIOUS_AI_MODEL_URL` | `model_url` | string | `http://host.docker.internal:11434` |
| `SUBCONSCIOUS_AI_API_KEY` | `api_key` | string | `sk-...` |
| `SUBCONSCIOUS_AI_PULSE_INTERVAL_SECONDS` | `pulse_interval_seconds` | int | `300` |
| `SUBCONSCIOUS_AI_RATE_LIMIT_PER_HOUR` | `rate_limit_per_hour` | int | `20` |
| `SUBCONSCIOUS_AI_MAX_CPU_PERCENT` | `max_cpu_percent` | float | `40.0` |
| `SUBCONSCIOUS_AI_DRY_RUN` | `dry_run` | bool | `false` |
| `SUBCONSCIOUS_AI_AUDIT_LOG_PATH` | `audit_log_path` | path | `data/subconscious_audit.jsonl` |

### Setting on Linux/macOS

```bash
export SUBCONSCIOUS_AI_ENABLED=true
export SUBCONSCIOUS_AI_MODEL_NAME="llama3.2:3b"
```

### Setting on Windows PowerShell

```powershell
$env:SUBCONSCIOUS_AI_ENABLED = "true"
$env:SUBCONSCIOUS_AI_MODEL_NAME = "llama3.2:3b"
```

### Setting in a `.env` file (for Docker Compose or python-dotenv)

```dotenv
SUBCONSCIOUS_AI_ENABLED=true
SUBCONSCIOUS_AI_MODEL_PROVIDER=ollama
SUBCONSCIOUS_AI_MODEL_NAME=phi3.5:latest
SUBCONSCIOUS_AI_MODEL_URL=http://host.docker.internal:11434
# SUBCONSCIOUS_AI_API_KEY=sk-...   # only for OpenAI / Anthropic
```

---

## 8. Docker / Docker-Compose Setup

### The `localhost` Problem

When MnemoCore runs inside a Docker container and Ollama runs **on the host machine** (outside
Docker), the container cannot reach `http://localhost:11434` because `localhost` inside a container
refers to the container itself, not the host.

**Solution:** Use the special Docker hostname `host.docker.internal` instead:

```
http://host.docker.internal:11434
```

`host.docker.internal` is automatically available on:
- Docker Desktop for Windows
- Docker Desktop for macOS
- Docker Engine >= 20.10 on Linux with `--add-host=host.docker.internal:host-gateway`

### docker-compose.yml

The `docker-compose.yml` in this repository already contains a commented-out block in the
`mnemocore` service `environment:` section:

```yaml
services:
  mnemocore:
    environment:
      - HOST=0.0.0.0
      - PORT=8100
      # ── Subconscious AI ────────────────────────────────────────────────────
      # Uncomment and set values to override what is in config.yaml.
      # - SUBCONSCIOUS_AI_ENABLED=${SUBCONSCIOUS_AI_ENABLED:-false}
      # - SUBCONSCIOUS_AI_MODEL_PROVIDER=${SUBCONSCIOUS_AI_MODEL_PROVIDER:-ollama}
      # - SUBCONSCIOUS_AI_MODEL_NAME=${SUBCONSCIOUS_AI_MODEL_NAME:-phi3.5:latest}
      # - SUBCONSCIOUS_AI_MODEL_URL=${SUBCONSCIOUS_AI_MODEL_URL:-http://host.docker.internal:11434}
      # - SUBCONSCIOUS_AI_API_KEY=${SUBCONSCIOUS_AI_API_KEY:-}
      # - SUBCONSCIOUS_AI_PULSE_INTERVAL_SECONDS=${SUBCONSCIOUS_AI_PULSE_INTERVAL_SECONDS:-120}
      # - SUBCONSCIOUS_AI_MAX_CPU_PERCENT=${SUBCONSCIOUS_AI_MAX_CPU_PERCENT:-30.0}
```

To enable for Docker:

1. Uncomment the lines above (remove the leading `# `).
2. Create a `.env` file next to `docker-compose.yml`:
   ```dotenv
   SUBCONSCIOUS_AI_ENABLED=true
   SUBCONSCIOUS_AI_MODEL_URL=http://host.docker.internal:11434
   ```
3. Bring the stack up:
   ```bash
   docker compose up -d
   ```

### Linux: Adding `host.docker.internal`

On Linux the hostname is not added automatically. Add it to the service in `docker-compose.yml`:

```yaml
services:
  mnemocore:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### Running Ollama Inside Docker

If you prefer to also containerise Ollama itself, add an `ollama` service and use the internal
Docker network name:

```yaml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  mnemocore:
    environment:
      - SUBCONSCIOUS_AI_MODEL_URL=http://ollama:11434   # internal network name

volumes:
  ollama_data:
```

Then pull the model inside the container:

```bash
docker compose exec ollama ollama pull phi3.5
```

---

## 9. Runtime Operations Explained

The worker rotates through up to three operations in a round-robin cycle. Each pulse executes
exactly one operation (the next one in the cycle that is enabled).

### Operation 1 — Memory Sorting (`memory_sorting`)

**Prompt intent:** Given a sample of recent memories with their current importance scores, ask the
LLM to propose a re-ranked ordering and a brief rationale for each change.

**What the worker does with the response:**
- Parses the suggested new ranking.
- If `dry_run: false`, updates the importance score of each affected memory in the store.
- Writes the full interaction to the audit log.

**Config flag:** `enable_memory_sorting: true`

### Operation 2 — Enhanced Dreaming (`enhanced_dreaming`)

**Prompt intent:** Given a sample of memory fragments, ask the LLM to identify latent semantic
associations — connections between memories that are not explicitly linked — and create new
associative edges.

**What the worker does with the response:**
- Parses the suggested associations (pairs of memory IDs + relationship description).
- If `dry_run: false`, creates new association edges in the memory graph.
- Writes the full interaction to the audit log.

**Config flag:** `enable_enhanced_dreaming: true`

### Operation 3 — Micro Self-Improvement (`micro_self_improvement`)

**Prompt intent:** Ask the LLM to observe patterns in recent interactions and propose small
operational notes for the memory system (e.g., "user frequently queries time-based context — consider
boosting recency weighting").

**What the worker does with the response:**
- Records the suggestion in the improvement notes store (read-only by default in Phase 0).
- **Does not modify any scores, weights, or code.**
- Writes the full interaction to the audit log.

**Config flag:** `enable_micro_self_improvement: false` (disabled by default)

> **Note:** This operation is still in Phase 0 "dry-observe" mode. The suggestions are logged but
> the system never automatically applies architectural changes. See
> [docs/SELF_IMPROVEMENT_DEEP_DIVE.md](SELF_IMPROVEMENT_DEEP_DIVE.md) for the full roadmap.

---

## 10. Audit Trail Format

Every LLM interaction (whether or not writes are applied) is appended to:

```
./data/subconscious_audit.jsonl
```

Each line is a self-contained JSON object with the following schema:

```jsonc
{
  "timestamp": "2025-01-15T14:32:10.123456Z",   // ISO-8601 UTC
  "operation": "memory_sorting",                  // one of the three operation names
  "model_provider": "ollama",
  "model_name": "phi3.5:latest",
  "prompt": "...",                                // the exact prompt sent to the LLM
  "response": "...",                              // the raw response from the LLM
  "suggestions_count": 3,                         // number of parsed suggestions
  "applied": true,                                // false when dry_run=true or no valid suggestions
  "dry_run": false,
  "cpu_percent_at_time": 12.4,                    // CPU load when the pulse fired
  "duration_ms": 843                              // round-trip time to the LLM in milliseconds
}
```

**Inspecting the audit trail:**

```bash
# Print all entries (pretty-printed)
cat data/subconscious_audit.jsonl | python -m json.tool

# Filter for applied suggestions only
grep '"applied": true' data/subconscious_audit.jsonl

# Count entries per operation
grep -o '"operation": "[^"]*"' data/subconscious_audit.jsonl | sort | uniq -c

# Show last 10 entries
tail -10 data/subconscious_audit.jsonl
```

---

## 11. dry_run Mode

`dry_run: true` is the safe way to evaluate a new model or new configuration before committing to
live memory modifications.

**With `dry_run: true`:**
- The worker still wakes up on schedule.
- The LLM is still called and a response is received.
- All suggestions are still parsed and recorded in the audit trail with `"applied": false`.
- **No changes are made to the memory store.**

**With `dry_run: false` (default):**
- Valid suggestions are written back to the memory store.
- The audit trail records `"applied": true`.

**Recommended workflow when switching models:**

1. Set `dry_run: true` and start MnemoCore.
2. Wait for several pulses (e.g., 10 minutes).
3. Review `data/subconscious_audit.jsonl` and validate the suggestion quality.
4. If happy, set `dry_run: false` and restart.

---

## 12. Monitoring & Log Patterns

The worker logs at `INFO` level throughout its lifecycle. Key log messages to watch:

| Log Pattern | Meaning |
|---|---|
| `SubconsciousAIWorker starting — provider=ollama model=phi3.5:latest` | Worker started successfully |
| `SubconsciousAIWorker pulse — op=memory_sorting` | A pulse is being executed |
| `SubconsciousAIWorker pulse skipped — cpu=67.2% > max=30.0%` | Pulse skipped due to high CPU load |
| `SubconsciousAIWorker pulse skipped — rate limit reached` | Hourly rate limit hit |
| `SubconsciousAIWorker applied 3 suggestions` | Suggestions written to memory store |
| `SubconsciousAIWorker dry_run — 3 suggestions NOT applied` | dry_run mode, suggestions discarded |
| `SubconsciousAIWorker error — Connection refused` | LLM server not reachable |
| `SubconsciousAIWorker stopping` | Worker shut down cleanly |

**Recommended log level for production:** `INFO`

To see debug-level detail (full prompts and responses):

```bash
LOG_LEVEL=DEBUG uvicorn src.mnemocore.api.app:app
```

---

## 13. Troubleshooting

### Ollama is unreachable (`Connection refused` or `Cannot connect to host`)

**Symptoms:** Log shows `SubconsciousAIWorker error — Connection refused to http://localhost:11434`

**Checks:**
1. Is Ollama running? Run `ollama list` — if it hangs or errors, start Ollama first.
2. On Windows: look for the Ollama system tray icon. If absent, launch Ollama from Start menu.
3. On Linux/macOS: run `ps aux | grep ollama` and check for the process.
4. Inside Docker: did you change `localhost` to `host.docker.internal`? See [Section 8](#8-docker--docker-compose-setup).
5. Check the port: `curl http://localhost:11434/api/tags` should return a JSON list of models.

### Model not found (`model not found` error from Ollama)

**Symptoms:** Ollama responds with `{"error":"model 'phi3.5:latest' not found"}`

**Fix:** Pull the model:
```bash
ollama pull phi3.5
```
Then verify: `ollama list` should show the model.

### Pulses keep being skipped (CPU too high)

The `ResourceGuard` skips pulses when CPU usage exceeds `max_cpu_percent` (default 30 %).

**Options:**
- Raise the threshold: `max_cpu_percent: 60.0`
- Increase `pulse_interval_seconds` so pulses are less frequent
- Accept it — the guard is doing its job; pulses will resume when load drops

### Rate limit reached too quickly

Lower `rate_limit_per_hour` default (30) may be hit if `pulse_interval_seconds` is very low.

**Fix:** Increase `pulse_interval_seconds` or raise `rate_limit_per_hour`.

### LM Studio — model name mismatch

LM Studio requires that `model_name` exactly matches the identifier shown in its UI.

**Fix:** Copy the model identifier string from the LM Studio "Developer" tab and paste it into
`config.yaml` or the environment variable.

### OpenAI / Anthropic — authentication error (401 / 403)

**Fix:**
1. Never put the API key in `config.yaml` — use the environment variable:
   ```bash
   export SUBCONSCIOUS_AI_API_KEY="sk-..."
   ```
2. Verify the key is not expired and has sufficient credits.

### No entries appearing in the audit trail

1. Confirm `enabled: true` in `config.yaml`.
2. Confirm the file/directory path is writable (`data/` must exist or the worker will create it).
3. Check logs for `pulse skipped` or `error` messages.
4. Wait at least `pulse_interval_seconds` (default 120 s) after startup for the first pulse.

---

## 14. Security Notes

- **Never hard-code API keys** in `config.yaml` or in source code. Use environment variables:
  `SUBCONSCIOUS_AI_API_KEY`.
- The `api_key` field in `config.yaml` is intentionally left blank in the repository. It is marked
  in `.gitignore` patterns via the config loader — but the safest approach is always env vars.
- The audit log (`data/subconscious_audit.jsonl`) contains **full prompt and response text**,
  which may include memory content. Treat it as sensitive data; do not commit it to version
  control or expose it publicly. It is already listed in `.gitignore`.
- For local Ollama deployments, no data leaves your machine. The model runs entirely locally.
- For OpenAI / Anthropic, memory fragments are sent to their APIs. Review their data-retention
  and privacy policies before enabling in production environments that handle sensitive data.
- `dry_run: true` is recommended for initial evaluation with any commercial API to prevent
  unexpected writes and to audit the content being sent before committing to production use.

---

## 15. Extending: Adding a New Provider

All model clients implement the same two-method interface:

```python
class MyProviderClient:
    def __init__(self, config: SubconsciousAIConfig) -> None:
        ...

    async def generate(self, prompt: str) -> str:
        """Send prompt, return response text."""
        ...
```

To add a new provider:

1. Implement the class in `src/mnemocore/core/subconscious_ai.py`.
2. Add the provider string (e.g., `"my_provider"`) to `_init_model_client()` factory method.
3. Add the value to the `model_provider` config docs and to this file.
4. Add at least one test in `tests/test_subconscious_ai.py` covering happy path and error path.

---

*Last updated: MnemoCore v5.0.0 — Phase 4.4 (Subconscious AI Worker)*
