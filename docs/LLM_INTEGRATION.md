# MnemoCore LLM Integration

> **Version**: 5.1.0 &nbsp;|&nbsp; **Source**: `src/mnemocore/llm/`

MnemoCore integrates with multiple LLM providers for cognitive operations including dream synthesis, reconstructive recall, recursive queries, and self-improvement.

---

## Table of Contents

- [Architecture](#architecture)
- [Supported Providers](#supported-providers)
- [Configuration](#configuration)
- [LLM Client Factory](#llm-client-factory)
- [HAIM LLM Integrator](#haim-llm-integrator)
- [Recursive Language Model (RLM)](#recursive-language-model-rlm)
- [Multi-Agent LLM](#multi-agent-llm)
- [Context-Aware LLM](#context-aware-llm)
- [Usage Examples](#usage-examples)

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    LLM Subsystem                           │
│                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  LLMConfig   │───>│ LLMClient    │───>│    Provider  │ │
│  │  (provider,  │    │   Factory    │    │    API       │ │
│  │   model,     │    │              │    │  (OpenAI,    │ │
│  │   api_key)   │    │              │    │   Ollama,    │ │
│  └──────────────┘    └──────────────┘    │   etc.)      │ │
│                                          └──────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              HAIM LLM Integrator                      │ │
│  │  reconstructive_recall() | multi_hypothesis_query()   │ │
│  │  consolidate_memory()                                 │ │
│  └──────────────────────┬───────────────────────────────┘ │
│                         │                                  │
│  ┌──────────────────────▼───────────────────────────────┐ │
│  │              RLM Integrator                           │ │
│  │  rlm_query() - recursive decompose & synthesize      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────┐    ┌──────────────┐                     │
│  │ Multi-Agent  │    │ Context-     │                     │
│  │    LLM       │    │   Aware LLM  │                     │
│  └──────────────┘    └──────────────┘                     │
└────────────────────────────────────────────────────────────┘
```

---

## Supported Providers

| Provider | Identifier | Local/Cloud | Notes |
|----------|-----------|-------------|-------|
| **Ollama** | `ollama` | Local | Recommended for local development. Default URL: `http://localhost:11434` |
| **LM Studio** | `lm_studio` | Local | OpenAI-compatible API. Default URL: `http://localhost:1234/v1` |
| **OpenAI** | `openai` | Cloud | GPT-4, GPT-4o, etc. Requires API key |
| **OpenRouter** | `openrouter` | Cloud | Access multiple models via single API. Uses OpenAI-compatible protocol |
| **Anthropic** | `anthropic` | Cloud | Claude models. Requires API key |
| **Google Gemini** | `google_gemini` | Cloud | Gemini models. Requires API key |
| **Custom** | `custom` | Either | Any OpenAI-compatible endpoint |
| **Mock** | `mock` | — | Returns deterministic responses. For testing |

---

## Configuration

### Python Configuration

```python
from mnemocore.llm.config import LLMConfig, LLMProvider

# Using factory methods (recommended)
config = LLMConfig.ollama(model="llama3.1")
config = LLMConfig.openai(model="gpt-4o", api_key="sk-...")
config = LLMConfig.anthropic(model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")
config = LLMConfig.openrouter(model="anthropic/claude-3.5-sonnet", api_key="sk-or-...")
config = LLMConfig.google_gemini(model="gemini-1.5-pro", api_key="...")
config = LLMConfig.lm_studio(model="local-model")
config = LLMConfig.custom(model="my-model", base_url="http://my-server:8000/v1", api_key="...")
config = LLMConfig.mock()

# Manual configuration
config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model="phi3.5:3.8b",
    base_url="http://localhost:11434",
    max_tokens=1024,
    temperature=0.7,
    extra_headers={"X-Custom": "value"},
    extra_params={"top_p": 0.9}
)
```

### LLMConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `LLMProvider` | `MOCK` | LLM provider |
| `model` | `str` | `"gpt-4"` | Model name |
| `api_key` | `str?` | `None` | API key |
| `base_url` | `str?` | `None` | Custom base URL |
| `max_tokens` | `int` | `1024` | Max response tokens |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `extra_headers` | `dict` | `{}` | Additional HTTP headers |
| `extra_params` | `dict` | `{}` | Additional model parameters |

### YAML Configuration (Subconscious AI)

The subconscious worker uses a separate config path in `config.yaml`:

```yaml
subconscious_ai:
  enabled: true
  model_provider: ollama       # Provider identifier
  model_name: "phi3.5:3.8b"   # Model name
  model_url: "http://localhost:11434"
  api_key: null                # Set via SUBCONSCIOUS_API_KEY env var
  rate_limit_per_hour: 50
  cycle_timeout_seconds: 30
```

---

## LLM Client Factory

The `LLMClientFactory` creates provider-specific clients from configuration:

```python
from mnemocore.llm.factory import LLMClientFactory
from mnemocore.llm.config import LLMConfig

config = LLMConfig.ollama(model="llama3.1")
client = LLMClientFactory.create_client(config)
```

The factory handles provider-specific initialization:

| Provider | Client Created |
|----------|---------------|
| `MOCK` | `None` (uses internal mock) |
| `OPENAI` | `openai.OpenAI(api_key=...)` |
| `OPENROUTER` | `openai.OpenAI(base_url="https://openrouter.ai/api/v1")` |
| `ANTHROPIC` | `anthropic.Anthropic(api_key=...)` |
| `GOOGLE_GEMINI` | `google.generativeai.GenerativeModel(...)` |
| `OLLAMA` | `openai.OpenAI(base_url=...)` or `OllamaClient` fallback |
| `LM_STUDIO` | `openai.OpenAI(base_url=..., api_key="lm-studio")` |
| `CUSTOM` | `openai.OpenAI(base_url=..., api_key=...)` |

---

## HAIM LLM Integrator

The `HAIMLLMIntegrator` bridges the HAIM memory engine with LLM reasoning capabilities.

### Initialization

```python
from mnemocore.llm.integrator import HAIMLLMIntegrator
from mnemocore.llm.config import LLMConfig

# From existing engine + config
integrator = HAIMLLMIntegrator.from_config(engine, LLMConfig.ollama())

# Or with a pre-created client
integrator = HAIMLLMIntegrator(haim_engine=engine, llm_client=client, llm_config=config)
```

### Reconstructive Recall

Synthesizes an answer by combining top memory matches with LLM reasoning:

```python
result = integrator.reconstructive_recall(
    cue="How does migration help species survive?",
    top_memories=5,
    enable_reasoning=True
)
# Returns: {
#   "cue": "...",
#   "memories": [...],
#   "synthesis": "Migration helps species survive by...",
#   "reasoning_enabled": True
# }
```

### Multi-Hypothesis Query

Tests multiple hypotheses against memory evidence:

```python
result = integrator.multi_hypothesis_query(
    query="Why do birds migrate?",
    hypotheses=[
        "Birds migrate to find food",
        "Birds migrate to avoid predators",
        "Birds migrate due to temperature changes"
    ]
)
# Returns ranked hypotheses with evidence scores
```

### Memory Consolidation with LLM

Uses LLM to generate consolidation insights:

```python
integrator.consolidate_memory(
    node_id="mem_abc123",
    new_context="Additional observation about bird behavior",
    success=True
)
```

---

## Recursive Language Model (RLM)

The `RLMIntegrator` (Phase 4.5) provides recursive query decomposition — breaking complex questions into sub-queries, retrieving relevant memories for each, and synthesizing a unified answer.

### How It Works

```
Complex Query
    ├── Sub-query 1 → Memory Search → Results
    ├── Sub-query 2 → Memory Search → Results
    └── Sub-query 3 → Memory Search → Results
         └── LLM Synthesis → Final Answer + Ripple Context
```

### Usage

```python
from mnemocore.llm.rlm import RLMIntegrator

# Create from config
rlm = RLMIntegrator.from_config(engine, llm_config)

# Execute recursive query
result = await rlm.rlm_query(
    query="How do cognitive architectures relate to human memory models?",
    context_text="Optional additional context...",
    project_id="research-project-1"
)

# Result structure:
# {
#   "query": "...",
#   "sub_queries": ["What are cognitive architectures?", "What are human memory models?", ...],
#   "results": [...],
#   "synthesis": "Cognitive architectures and human memory models share...",
#   "max_depth_hit": 2,
#   "elapsed_ms": 450.3,
#   "ripple_snippets": ["Related: ACT-R uses a hybrid approach..."],
#   "stats": {"total_queries": 7, "cache_hits": 2}
# }
```

### RLM via REST API

```bash
curl -X POST http://localhost:8100/rlm/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "query": "How do cognitive architectures relate to human memory models?",
    "max_depth": 2,
    "max_sub_queries": 3,
    "top_k": 5
  }'
```

---

## Multi-Agent LLM

The `multi_agent` module supports multi-agent LLM conversations and coordination.

```python
from mnemocore.llm.multi_agent import MultiAgentLLM
```

---

## Context-Aware LLM

The `context_aware` module provides LLM calls that are automatically enriched with relevant memory context.

```python
from mnemocore.llm.context_aware import ContextAwareLLM
```

---

## Usage Examples

### Setting Up Ollama (Recommended for Local)

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull phi3.5:3.8b      # Small, fast
ollama pull llama3.1          # Medium, versatile
ollama pull gemma3:1b         # Small, for dreams

# Configure MnemoCore
# config.yaml:
subconscious_ai:
  enabled: true
  model_provider: ollama
  model_name: "phi3.5:3.8b"
  model_url: "http://localhost:11434"
  dry_run: true  # Start in dry-run mode
```

### Using OpenAI

```python
from mnemocore.llm.config import LLMConfig
from mnemocore.llm.integrator import HAIMLLMIntegrator

config = LLMConfig.openai(
    model="gpt-4o",
    api_key="sk-..."  # Or set OPENAI_API_KEY env var
)
integrator = HAIMLLMIntegrator.from_config(engine, config)

result = integrator.reconstructive_recall("What patterns exist in the data?")
```

### Using Anthropic Claude

```python
config = LLMConfig.anthropic(
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-..."
)
integrator = HAIMLLMIntegrator.from_config(engine, config)
```

### Testing with Mock Provider

```python
config = LLMConfig.mock()
integrator = HAIMLLMIntegrator.from_config(engine, config)
# Returns deterministic mock responses — useful for testing
```

---

*See [SUBCONSCIOUS_AI.md](SUBCONSCIOUS_AI.md) for detailed subconscious worker setup. See [CONFIGURATION.md](CONFIGURATION.md) for all config options.*
