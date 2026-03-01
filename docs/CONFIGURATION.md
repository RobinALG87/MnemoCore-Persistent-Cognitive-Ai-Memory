# MnemoCore Configuration Reference

> **Version**: 5.1.0 &nbsp;|&nbsp; **Source**: `src/mnemocore/core/config.py`

This document covers every configuration option in MnemoCore. Configuration is loaded from `config.yaml` and can be overridden by environment variables. Sensitive values (API keys, passwords) should **always** be set via environment variables.

---

## Table of Contents

- [Configuration Loading](#configuration-loading)
- [Section 1 — Infrastructure](#section-1--infrastructure)
- [Section 2 — API & Security](#section-2--api--security)
- [Section 3 — Encoding & Core](#section-3--encoding--core)
- [Section 4 — Subconscious](#section-4--subconscious)
- [Section 5 — Performance](#section-5--performance)
- [Section 6 — Cognitive (Phase 5)](#section-6--cognitive-phase-5)
- [Section 7 — Extensions](#section-7--extensions)
- [Section 8 — Root Config](#section-8--root-config)
- [Environment Variable Overrides](#environment-variable-overrides)
- [Example config.yaml](#example-configyaml)

---

## Configuration Loading

Configuration is loaded by `load_config()` with the following priority:

```
Environment Variables  >  config.yaml  >  Dataclass Defaults
```

All configuration is stored in **frozen dataclasses** — once loaded, config is immutable for thread safety.

```python
from mnemocore.core.config import load_config, get_config

# Load from file
config = load_config(Path("config.yaml"))

# Get singleton (thread-safe)
config = get_config()
```

---

## Section 1 — Infrastructure

### `TierConfig`

Controls the three-tier memory storage system.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_memories` | `int` | varies | Max memories per tier |
| `ltp_threshold_min` | `float` | varies | Min LTP strength to remain in tier |
| `eviction_policy` | `str` | `"lru"` | Eviction policy: `lru`, `lfu`, `fifo`, `random` |
| `consolidation_interval_hours` | `int?` | `null` | Hours between auto-consolidation |
| `storage_backend` | `str` | `"memory"` | Backend: `memory`, `mmap`, `filesystem`, `qdrant`, `redis` |
| `compression` | `str` | `"gzip"` | Compression for filesystem storage |
| `archive_threshold_days` | `int` | `30` | Days before archiving |

```yaml
tiers:
  hot:
    max_memories: 2000
    ltp_threshold_min: 0.3
    eviction_policy: lru
  warm:
    max_memories: 100000
    storage_backend: mmap
  cold:
    max_memories: 0  # unlimited
    storage_backend: filesystem
    compression: gzip
```

### `LTPConfig`

Long-Term Potentiation (memory strength) parameters.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `initial_importance` | `float` | `0.5` | Starting importance for new memories |
| `decay_lambda` | `float` | `0.01` | Exponential decay rate |
| `permanence_threshold` | `float` | `0.95` | LTP above which memory is "permanent" |
| `half_life_days` | `float` | `30.0` | Half-life for decay calculation |

### `HysteresisConfig`

Prevents tier thrashing at boundaries.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `promote_delta` | `float` | `0.15` | Extra LTP needed to promote |
| `demote_delta` | `float` | `0.10` | LTP deficit needed to demote |

### `RedisConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `url` | `str` | `"redis://localhost:6379/0"` | Redis connection URL |
| `stream_key` | `str` | `"haim:subconscious"` | Redis stream key |
| `max_connections` | `int` | `10` | Connection pool size |
| `socket_timeout` | `int` | `5` | Socket timeout (seconds) |
| `password` | `str?` | `null` | Redis password (prefer env var) |

### `QdrantConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `url` | `str` | `"http://localhost:6333"` | Qdrant server URL |
| `collection_hot` | `str` | `"haim_hot"` | Hot tier collection name |
| `collection_warm` | `str` | `"haim_warm"` | Warm tier collection name |
| `binary_quantization` | `bool` | `true` | Enable binary quantization |
| `always_ram` | `bool` | `true` | Keep vectors in RAM |
| `hnsw_m` | `int` | `16` | HNSW graph degree |
| `hnsw_ef_construct` | `int` | `100` | HNSW construction ef |
| `api_key` | `str?` | `null` | Qdrant API key (prefer env var) |

### `GPUConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable GPU acceleration |
| `device` | `str` | `"cuda:0"` | GPU device identifier |
| `batch_size` | `int` | `1000` | Batch size for GPU ops |
| `fallback_to_cpu` | `bool` | `true` | Fall back to CPU if GPU unavailable |

### `PathsConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data_dir` | `str` | `"./data"` | Root data directory |
| `memory_file` | `str` | `"./data/memory.jsonl"` | Memory persistence file |
| `codebook_file` | `str` | `"./data/codebook.json"` | Codebook file |
| `concepts_file` | `str` | `"./data/concepts.json"` | Concepts file |
| `synapses_file` | `str` | `"./data/synapses.json"` | Synapse graph file |
| `warm_mmap_dir` | `str` | `"./data/warm_tier"` | Warm mmap directory |
| `cold_archive_dir` | `str` | `"./data/cold_archive"` | Cold archive directory |

### `SearchConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | `str` | `"hybrid"` | Search mode: `hybrid`, `dense`, `sparse` |
| `hybrid_alpha` | `float` | `0.7` | Weight for dense vs sparse (1.0 = all dense) |
| `rrf_k` | `int` | `60` | Reciprocal Rank Fusion parameter |
| `sparse_model` | `str` | `"bm25"` | Sparse retrieval model |
| `enable_query_expansion` | `bool` | `true` | Enable automatic query expansion |

---

## Section 2 — API & Security

### `SecurityConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_key` | `str?` | `null` | API key for authentication (use `HAIM_API_KEY` env var) |
| `cors_origins` | `list[str]` | `["http://localhost:3000", "http://localhost:8100"]` | Allowed CORS origins |
| `rate_limit_enabled` | `bool` | `true` | Enable rate limiting |
| `rate_limit_requests` | `int` | `100` | Requests per window |
| `rate_limit_window` | `int` | `60` | Rate limit window (seconds) |
| `trusted_proxies` | `list[str]` | `[]` | Trusted reverse proxy IPs |
| `hsts_enabled` | `bool` | `true` | Enable HSTS header |

### `ObservabilityConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `metrics_port` | `int` | `9090` | Prometheus metrics port |
| `log_level` | `str` | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `structured_logging` | `bool` | `true` | Enable structured JSON logging |

### `MCPConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable MCP server |
| `transport` | `str` | `"stdio"` | Transport: `stdio` or `tcp` |
| `host` | `str` | `"127.0.0.1"` | TCP host (if transport = tcp) |
| `port` | `int` | `8110` | TCP port |
| `api_base_url` | `str` | `"http://localhost:8100"` | API base URL for MCP adapter |
| `api_key` | `str?` | `null` | API key for MCP-to-API auth |
| `timeout_seconds` | `int` | `15` | Request timeout |
| `allow_tools` | `list[str]` | all tools | Enabled MCP tools |

---

## Section 3 — Encoding & Core

### `EncodingConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | `str` | `"binary"` | Encoding mode: `binary` or `dense` |
| `token_method` | `str` | `"bundle"` | Token composition method |

### `AttentionMaskingConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable XOR attention masking |

### `ConsolidationConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable background consolidation |
| `interval_seconds` | `int` | `3600` | Consolidation interval |
| `similarity_threshold` | `float` | `0.85` | Min similarity for merge candidates |
| `min_cluster_size` | `int` | `2` | Min cluster size |

### `SynapseConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `similarity_threshold` | `float` | `0.5` | Min similarity for auto-binding |
| `auto_bind_on_store` | `bool` | `true` | Auto-create synapses on store |
| `multi_hop_depth` | `int` | `2` | Max hops for association queries |

### `ContextConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable context tracking |
| `shift_threshold` | `float` | `0.3` | Threshold for context shift detection |
| `rolling_window_size` | `int` | `5` | Rolling window for context tracking |

### `PreferenceConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable preference learning |
| `learning_rate` | `float` | `0.1` | Preference learning rate |
| `history_limit` | `int` | `100` | Max preference history entries |

### `AnticipatoryConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable anticipatory memory |
| `predictive_depth` | `int` | `1` | Prediction depth |

---

## Section 4 — Subconscious

### `DreamLoopConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable dream loop |
| `frequency_seconds` | `int` | `60` | Dream loop frequency |
| `batch_size` | `int` | `10` | Memories per dream batch |
| `max_iterations` | `int` | `0` | Max iterations (0 = unlimited) |
| `subconscious_queue_maxlen` | `int?` | `null` | Max queue length |
| `ollama_url` | `str` | `"http://localhost:11434/api/generate"` | Ollama API URL |
| `model` | `str` | `"gemma3:1b"` | Model for dream synthesis |

### `SubconsciousAIConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable the Subconscious AI Worker |
| `beta_mode` | `bool` | `true` | Enable beta safety nets |
| `model_provider` | `str` | `"ollama"` | Provider: `ollama`, `lm_studio`, `openai`, `anthropic` |
| `model_name` | `str` | `"phi3.5:3.8b"` | Model name |
| `model_url` | `str` | `"http://localhost:11434"` | Model server URL |
| `api_key` | `str?` | `null` | API key for cloud providers |
| `pulse_interval_seconds` | `int` | `120` | Interval between subconscious pulses |
| `max_cpu_percent` | `float` | `30.0` | Max CPU usage limit |
| `cycle_timeout_seconds` | `int` | `30` | Timeout per cycle |
| `rate_limit_per_hour` | `int` | `50` | Max LLM calls per hour |
| `memory_sorting_enabled` | `bool` | `true` | Enable memory sorting optimization |
| `enhanced_dreaming_enabled` | `bool` | `true` | Enable enhanced dream synthesis |
| `micro_self_improvement_enabled` | `bool` | `false` | Enable self-improvement (Phase 0) |
| `dry_run` | `bool` | `true` | Dry run mode (no mutations) |
| `log_all_decisions` | `bool` | `true` | Log all subconscious decisions |
| `audit_trail_path` | `str?` | `"./data/subconscious_audit.jsonl"` | Audit trail file |
| `max_memories_per_cycle` | `int` | `10` | Memories processed per cycle |

---

## Section 5 — Performance

### `PerformanceConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `background_rebuild_enabled` | `bool` | `true` | Enable background index rebuilds |
| `process_priority_low` | `bool` | `true` | Run background tasks at low priority |
| `vector_cache_enabled` | `bool` | `true` | Enable vector cache (SQLite) |
| `vector_cache_path` | `str?` | `"./data/vector_cache.sqlite"` | Vector cache file path |

### `VectorCompressionConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable vector compression |
| `pq_n_subvectors` | `int` | `32` | Product quantization sub-vectors |

### `BackupConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable backup system |
| `auto_snapshot_enabled` | `bool` | `true` | Enable automatic snapshots |
| `snapshot_interval_hours` | `int` | `24` | Hours between snapshots |

---

## Section 6 — Cognitive (Phase 5)

### `WorkingMemoryConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_items_per_agent` | `int` | `20` | Max WM items per agent (inspired by Miller's 7±2) |
| `default_ttl_seconds` | `int` | `3600` | Default TTL for WM items |

### `EpisodicConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_active_episodes_per_agent` | `int` | `5` | Max concurrent episodes |
| `max_history_per_agent` | `int` | `500` | Max episode history length |

### `SemanticConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_similarity_threshold` | `float` | `0.5` | Min similarity for semantic matching |
| `max_local_cache_size` | `int` | `10000` | Local semantic cache size |

### `ProceduralConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_procedures` | `int` | `5000` | Max skill procedures |
| `enable_semantic_matching` | `bool` | `true` | Enable semantic procedure matching |

### `MetaMemoryConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_metrics_history` | `int` | `10000` | Max metrics history entries |
| `enable_llm_proposals` | `bool` | `true` | Enable LLM-generated proposals |

### `SelfImprovementConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable self-improvement loop |
| `dry_run` | `bool` | `true` | Dry run mode (no mutations) |
| `safety_mode` | `str` | `"strict"` | Safety mode: `strict`, `moderate`, `permissive` |

### `PulseConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable cognitive heartbeat |
| `interval_seconds` | `int` | `30` | Pulse tick interval |

### `StrategyBankConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable strategy bank |
| `max_strategies` | `int` | `10000` | Max stored strategies |

### `KnowledgeGraphConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable knowledge graph |
| `max_nodes` | `int` | `50000` | Max graph nodes |

### `MemorySchedulerConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable memory scheduler |
| `max_queue_size` | `int` | `10000` | Max scheduler queue size |

### `MemoryExchangeConfig` (SAMEP)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable multi-agent memory exchange |
| `max_shared_memories` | `int` | `50000` | Max shared memories |

---

## Section 7 — Extensions

### `EmbeddingRegistryConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable embedding version registry |
| `auto_migrate` | `bool` | `false` | Auto-migrate on model switch |

### `DreamingConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable dreaming pipeline |
| `idle_threshold_seconds` | `float` | `300.0` | Idle time before dream trigger (5 min) |

### `EFTConfig` (Episodic Future Thinking)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable future thinking |
| `max_scenarios_per_simulation` | `int` | `5` | Max scenarios per simulation run |

### `WebhookConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable webhook delivery |
| `persistence_path` | `str` | `"./data/webhooks.json"` | Webhook registration file |

### `EventsConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable event bus |
| `max_queue_size` | `int` | `10000` | Event queue capacity |
| `delivery_timeout` | `float` | `30.0` | Event delivery timeout (seconds) |

### `AssociationsConfig`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_save` | `bool` | `true` | Auto-save association graph |
| `decay_enabled` | `bool` | `true` | Enable association strength decay |

---

## Section 8 — Root Config

`HAIMConfig` is the top-level composite that aggregates all section configs:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | `str` | `"4.5"` | Config schema version |
| `dimensionality` | `int` | `16384` | HDV dimensionality (bits) |

Plus all section configs as nested attributes.

---

## Environment Variable Overrides

Sensitive values should be set via environment variables:

| Environment Variable | Config Path | Description |
|---------------------|-------------|-------------|
| `HAIM_API_KEY` | `security.api_key` | API authentication key |
| `REDIS_URL` | `redis.url` | Redis connection URL |
| `REDIS_PASSWORD` | `redis.password` | Redis password |
| `QDRANT_URL` | `qdrant.url` | Qdrant server URL |
| `QDRANT_API_KEY` | `qdrant.api_key` | Qdrant API key |
| `LOG_LEVEL` | `observability.log_level` | Logging level |
| `LOGURU_LEVEL` | — | Loguru-specific log level |
| `HOST` | — | API server host (default `0.0.0.0`) |
| `PORT` | — | API server port (default `8100`) |

---

## Example config.yaml

```yaml
# MnemoCore Configuration
version: "5.1"
dimensionality: 16384

tiers:
  hot:
    max_memories: 2000
    ltp_threshold_min: 0.3
    eviction_policy: lru
  warm:
    max_memories: 100000
    storage_backend: mmap
  cold:
    max_memories: 0
    storage_backend: filesystem
    compression: gzip

ltp:
  initial_importance: 0.5
  decay_lambda: 0.01
  permanence_threshold: 0.95
  half_life_days: 30.0

hysteresis:
  promote_delta: 0.15
  demote_delta: 0.10

security:
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8100"
  rate_limit_enabled: true
  rate_limit_requests: 100
  rate_limit_window: 60

encoding:
  mode: binary
  token_method: bundle

consolidation:
  enabled: true
  interval_seconds: 3600
  similarity_threshold: 0.85

synapse:
  similarity_threshold: 0.5
  auto_bind_on_store: true
  multi_hop_depth: 2

subconscious_ai:
  enabled: false
  beta_mode: true
  model_provider: ollama
  model_name: "phi3.5:3.8b"
  model_url: "http://localhost:11434"
  dry_run: true

working_memory:
  max_items_per_agent: 20
  default_ttl_seconds: 3600

pulse:
  enabled: true
  interval_seconds: 30

mcp:
  enabled: false
  transport: stdio

dreaming:
  enabled: true
  idle_threshold_seconds: 300.0

performance:
  vector_cache_enabled: true
  vector_cache_path: "./data/vector_cache.sqlite"

backup:
  enabled: true
  auto_snapshot_enabled: true
  snapshot_interval_hours: 24
```

---

*See [ARCHITECTURE.md](ARCHITECTURE.md) for system design. See [API.md](API.md) for endpoint reference.*
