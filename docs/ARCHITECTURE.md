# MnemoCore Architecture — v5.0.0

## Overview

MnemoCore is a cognitive memory infrastructure built on Binary Hyperdimensional Computing (HDC/VSA). It provides AI agents with persistent, self-organizing memory that consolidates, decays, and reasons — not just stores and retrieves.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           REST API (FastAPI)                         │
│  /store  /query  /feedback  /insights/gaps  /stats  /health         │
│  Rate Limiting · API Key Auth · Prometheus /metrics                  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                         HAIM Engine                                   │
│  engine.py + engine_core.py + engine_lifecycle.py (3 mixins)         │
│  Central cognitive coordinator — store, query, dream, feedback       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    DI Container (container.py)                  │  │
│  │  Wires all services via frozen config → engine constructor     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────────┐                     │
│  │   Binary HDV Core  │  │   Text Encoder     │                     │
│  │   16384-bit VSA    │  │   Token→HDV via    │                     │
│  │   XOR · Bundle ·   │  │   SHAKE-256 seed + │                     │
│  │   Permute · Hamming│  │   positional bind  │                     │
│  └────────────────────┘  └────────────────────┘                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                   Tier Manager                                │    │
│  │   🔥 HOT (dict, ≤2k, <1ms)                                   │    │
│  │   🌡 WARM (Redis/mmap, ≤100k, <10ms)                         │    │
│  │   ❄️  COLD (Qdrant/disk, ∞, <100ms)                           │    │
│  │   LTP-driven eviction · hysteresis · auto-promotion           │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────── Phase 4 Cognitive Layer ─────────────────────────┐    │
│  │  Bayesian LTP · Semantic Consolidation · Gap Detection       │    │
│  │  Immunology (attractor cleanup) · XOR Attention Masking      │    │
│  │  Episodic Chaining · Synapse Index · HNSW Index              │    │
│  │  Recursive Synthesizer · Ripple Context · Batch Ops          │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────── Phase 5 Cognitive Services ──────────────────────┐    │
│  │  WorkingMemory      — active slot buffer (7±2 items, TTL)    │    │
│  │  EpisodicStore      — temporal chains, LTP, chain repair     │    │
│  │  SemanticStore      — Qdrant concepts, CLS consolidation     │    │
│  │  ProceduralStore    — skill library, word-overlap matching    │    │
│  │  MetaMemory         — anomaly detection, LLM proposals       │    │
│  │  SelfImprovement    — dry-run Phase 0, 5 validation gates    │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────── Phase 6 Research Services ───────────────────────┐    │
│  │  StrategyBank       — 5-phase strategy loop, Bayesian conf.  │    │
│  │  KnowledgeGraph     — spreading activation, community det.   │    │
│  │  MemoryScheduler    — priority queue, interrupts, load shed  │    │
│  │  SAMEP (Exchange)   — multi-agent memory sharing, HMAC       │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────── Pulse Loop (Cognitive Heartbeat) ────────────────┐    │
│  │  Tick 1:  WM Maintenance         Tick  7: Meta Reflection    │    │
│  │  Tick 2:  Episodic Chaining      Tick  8: Strategy Refine    │    │
│  │  Tick 3:  Semantic Refresh       Tick  9: Graph Maintenance  │    │
│  │  Tick 4:  Gap Detection          Tick 10: Scheduler Tick     │    │
│  │  Tick 5:  Insight Generation     Tick 11: Exchange Sync      │    │
│  │  Tick 6:  Procedure Refinement                               │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────── Subconscious Layer ──────────────────────────────┐    │
│  │  SubconsciousAI     — LLM dream synthesis                    │    │
│  │  SubconsciousDaemon — background asyncio orchestrator         │    │
│  │  ConsolidationWorker — nightly merge + prune                  │    │
│  │  SelfImprovementWorker — autonomous memory refinement (P0)   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────── Meta Layer ──────────────────────────────────────┐    │
│  │  GoalTree           — hierarchical task tracking              │    │
│  │  LearningJournal    — persistent learning log                 │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Files

| Component | Path | Purpose |
|-----------|------|---------|
| **Engine** | `src/mnemocore/core/engine.py` | Central coordinator (3 mixins) |
| **Container** | `src/mnemocore/core/container.py` | DI wiring for all services |
| **Config** | `src/mnemocore/core/config.py` | 41 frozen dataclasses, YAML loader |
| **BinaryHDV** | `src/mnemocore/core/binary_hdv.py` | 16384-dim binary vector math |
| **TierManager** | `src/mnemocore/core/tier_manager.py` | HOT/WARM/COLD orchestration |
| **Pulse** | `src/mnemocore/core/pulse.py` | 11-phase cognitive heartbeat |
| **StrategyBank** | `src/mnemocore/core/strategy_bank.py` | Strategy distillation loop |
| **KnowledgeGraph** | `src/mnemocore/core/knowledge_graph.py` | Semantic graph + activation |
| **MemoryScheduler** | `src/mnemocore/core/memory_scheduler.py` | Priority job scheduler |
| **SAMEP** | `src/mnemocore/core/memory_exchange.py` | Multi-agent memory exchange |
| **WorkingMemory** | `src/mnemocore/core/working_memory.py` | Active slot buffer |
| **EpisodicStore** | `src/mnemocore/core/episodic_store.py` | Temporal episode chains |
| **SemanticStore** | `src/mnemocore/core/semantic_store.py` | Qdrant concept persistence |
| **ProceduralStore** | `src/mnemocore/core/procedural_store.py` | Skill library |
| **MetaMemory** | `src/mnemocore/core/meta_memory.py` | Anomaly detection |
| **API** | `src/mnemocore/api/main.py` | FastAPI REST interface |
| **MCP Server** | `src/mnemocore/mcp/server.py` | MCP protocol adapter |

---

## Configuration

All configuration lives in `config.yaml` and is loaded into a hierarchy of frozen dataclasses by `load_config()`. The 41 config classes are organized into 9 sections:

1. **§1 Infrastructure** — Redis, Qdrant, paths, performance
2. **§2 API & Security** — CORS, rate limits, API keys
3. **§3 Encoding & Core** — HDV dimensions, LTP params, tiering
4. **§4 Subconscious** — Dream worker, consolidation, self-improvement
5. **§5 Performance** — Batch sizes, concurrency, HNSW
6. **§6 Cognitive** — WM, episodic, semantic, procedural, meta
7. **§7 Extensions** — StrategyBank, KnowledgeGraph, MemoryScheduler, SAMEP
8. **§8 Root Composite** — `HAIMConfig` aggregating all sections
9. **§9 Loader** — YAML parsing + environment variable overrides

Sensitive values (API keys, passwords) are always read from environment variables, never stored in YAML.

---

## Testing

- **1291+ tests** across unit, integration, and regression suites
- `pytest` with `asyncio_mode=auto` for async test support
- Key test files: `test_cognitive_services.py` (65), `test_self_improvement_worker.py` (60+), `test_pulse_phases.py` (50+), `test_store_integration.py` (25+), `test_phase6_*.py` (85)

---

## Observability

- Prometheus metrics at `/metrics`
- Grafana dashboard: `grafana-dashboard.json`
- Structured logging via `loguru` (services) and stdlib `logging` (pulse/API)
- Per-phase timing in pulse loop stats

---

*Architecture document maintained alongside code. See `CHANGELOG.md` for version history.*

