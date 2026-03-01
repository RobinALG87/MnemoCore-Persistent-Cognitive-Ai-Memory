# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-02-28

### Fixed

#### llm_integration.py (6 fixes)
- **Import paths**: Fixed incorrect import paths from `haim.src.core.engine` to `src.core.engine` and `haim.src.core.node` to `src.core.node`
- **Missing import**: Added `from datetime import datetime` for dynamic timestamps
- **Memory access API**: Changed `self.haim.memory_nodes.get()` to `self.haim.tier_manager.get_memory()` - using the correct API for memory access
- **Superposition query**: Replaced non-existent `superposition_query()` call with combined hypotheses retrieval path
- **Concept binding**: Replaced non-existent `bind_concepts()` with placeholder - engine has `bind_memories()` available
- **OR orchestration**: Integrated `orchestrate_orch_or()` from engine and removed workaround sorting path

#### api/main.py (1 fix)
- **Delete endpoint**: Fixed attribute reference from `engine.memory_nodes` to `engine.tier_manager.hot` - correct attribute for hot memory tier

#### engine.py (1 fix)
- **Synapse persistence**: Implemented `_save_synapses()` method that was previously an empty stub
  - Creates parent directory if it doesn't exist
  - Writes all synapses to disk in JSONL format
  - Includes all synapse attributes: `neuron_a_id`, `neuron_b_id`, `strength`, `fire_count`, `success_count`, `last_fired`
  - Handles errors gracefully with logging

### Security

#### Hardening improvements
- **Thread safety**: Added proper locking in MemoryScheduler `_execute_job()`, SAMEP `discover()`, and StrategyBank `distill_from_episode()`
- **Interrupted jobs list**: Capped at 100 entries to prevent unbounded growth
- **Docker security**: Non-root user execution, read-only root filesystem, dropped capabilities
- **Network security**: Docker compose ports bound to 127.0.0.1 only, network policies in Helm chart

### Changed

#### Phase 4.3 hardening
- **Chrono-weighting**: Uses batched node lookup instead of per-node await chain
- **include_neighbors**: Now preserves `top_k` result contract
- **Private access**: `_dream_sem._value` replaced by public `locked()` API
- **Episodic chaining**: Race reduced with serialized store path (`_store_lock`, `_last_stored_id`)
- **engine_version**: Updated to `4.3.0` in stats
- **HOT-tier time_range**: Filtering enforced in `TierManager.search()`
- **orchestrate_orch_or()**: Made async and lock-guarded

#### Config system improvements
- **Dynamic timestamps**: LLM integration now uses `datetime.now().isoformat()` instead of hardcoded timestamp
- **Config-service alignment**: All cognitive service config fields aligned to match service `getattr()` reads

### Added

#### Subconscious AI Worker (Phase 4.4+)
- **Multi-provider support**: Ollama, LM Studio, OpenAI, Anthropic integration
- **Resource guard**: CPU usage monitoring with configurable limits
- **Pulse integration**: Background cognitive processing with configurable intervals
- **aiohttp dependency**: Added for async HTTP requests to AI providers
- **psutil dependency**: Added for CPU monitoring in ResourceGuard

#### Testing improvements
- **136+ dedicated tests**: Added for Phase 4.3 regressions and new features
- **Comprehensive coverage**: Tests for all cognitive services, edge cases, and integration flows

#### Documentation
- **Ollama integration docs**: Complete setup guide for local AI model integration
- **Docker docs**: Environment variable configuration for subconscious AI

## [2.0.0] — 2026-03-01

### Added

#### Phase 5.0–5.4: Cognitive Memory Architecture
- **SemanticStoreService** — Full Qdrant persistence, CLS-style hippocampus→neocortex consolidation, reliability decay, `find_nearby_concepts_qdrant()` with local fallback
- **EpisodicStoreService** — Temporal chain verification/repair, LTP calculation (`outcome × events × reward × decay`), max active episode enforcement, agent-scoped history eviction
- **ProceduralStoreService** — JSON disk persistence, word-overlap semantic matching, `create_procedure_from_episode()` bridge, outcome recording with boost/penalty, reliability decay
- **SelfImprovementWorker** — Phase 0 dry-run worker with 5 validation gates (semantic drift, fact safety, structure, policy, resource), candidate selection, rule-based proposals, 7 metrics counters, decision audit logging, backpressure detection
- **Pulse Loop** — All 7 cognitive phases implemented: WM maintenance, episodic chaining, semantic refresh, gap detection, insight generation, procedure refinement, meta self-reflection
- **Cognitive Configuration** — 6 frozen config dataclasses (`WorkingMemoryConfig`, `EpisodicConfig`, `SemanticConfig`, `ProceduralConfig`, `MetaMemoryConfig`, `SelfImprovementConfig`) wired into `HAIMConfig`
- **Engine Integration** — `procedural_store` and `meta_memory` wired via DI container; lazy-loaded cognitive accessors (`get_context_prioritizer`, `get_future_simulator`, `get_forgetting_analytics`)
- **Cognitive Health** — `cognitive_services` section added to `health_check()` with stats from all stores

#### Phase 6: Research-Backed Cognitive Services
- **StrategyBank** (Reasoning Bank) — 5-phase closed loop for strategy distillation: Evaluate→Select→Apply→Distill→Store. Bayesian confidence tracking, 60/40 positive/negative exemplar balance, category/tag indexing, disk persistence, capacity enforcement
- **KnowledgeGraph** — Bidirectional semantic graph with spreading activation, edge weight decay, reciprocal linking, community detection (label propagation), subgraph extraction, activation snapshots, max 50k nodes
- **MemoryScheduler** (MemoryOS) — Priority-queue job scheduler for memory operations: consolidation, pruning, linking, decay, health checks. Memory interrupts for critical information, system load awareness with automatic load shedding, stale job expiry, health-based lifecycle scoring (STM→MTM→LTM)
- **SAMEP** (Shared Associative Memory Exchange Protocol) — Multi-agent memory sharing with HMAC integrity verification, access control (NONE/READ/WRITE/ADMIN), tier-based visibility, semantic search via word overlap, access auditing
- **Pulse Loop Extended** — 4 new research phases: strategy refinement (Phase 8), graph maintenance (Phase 9), scheduler tick (Phase 10), exchange sync (Phase 11)

#### Testing
- 200+ new tests across Phase 5 and Phase 6 covering all services, edge cases, integration flows, and regression guards
- Full test suite: **1291 tests passing**

### Changed

- **Config system** — 41 frozen dataclasses organized with §1–§9 section markers. All cognitive service config fields aligned to match service `getattr()` reads (eliminates silent fallback-to-default bugs)
- **Pulse loop** — Expanded from 7 to 11 cognitive phases
- **Container** — `build_container()` now wires all Phase 6 services with config injection
- **Engine** — Constructor accepts 4 new DI params: `strategy_bank`, `knowledge_graph`, `memory_scheduler`, `memory_exchange`

### Fixed

- **Config-service alignment** — 5 critical field-name mismatches fixed where YAML config values were silently ignored (services always fell back to hardcoded defaults)
- **Thread safety** — `_execute_job()` in MemoryScheduler now wraps `_active_job` and `_completed_count` mutations inside lock; `discover()` in SAMEP wraps `access_count` mutations inside lock; `distill_from_episode()` in StrategyBank moves ratio check inside lock
- **Pulse scheduler_tick** — Fixed bug where `process_tick()` dict return was logged as an integer; now extracts `result.get('processed', 0)`
- **Interrupted jobs list** — Capped at 100 entries to prevent unbounded growth
- **Unused imports** — Cleaned across 5 service files (hashlib, asdict, Literal, Callable, Iterable, Set, Tuple, traceback)

### Removed

- Obsolete debug/temp files: `error_log.txt`, `failed_*.txt`, `test_*.txt`, `coverage.json`, `phase 6.md.md`
- Leaked temp directories: `.tmp_phase43_tests/`, `.tmp_pytest/`, `pytest_base_temp/`, `MagicMock/`, `test_data_perf/`
- Debug scripts: `mnemocore_verify.py`, `test_qdrant_scores.py`
- Obsolete docs: `docs/Aggera som en innovatör och en senior arkitekt ino.md`

## [Unreleased]

### Deprecated

#### Float HDV deprecation (src/core/hdv.py)
- **HDV class**: All public methods now emit `DeprecationWarning` when called
- **Migration path**: Use `BinaryHDV` from `src.core.binary_hdv` instead
- **API mappings**:
  - `HDV(dimension=N)` -> `BinaryHDV.random(dimension=N)`
  - `hdv.bind(other)` -> `hdv.xor_bind(other)`
  - `hdv.unbind(other)` -> `hdv.xor_bind(other)` (XOR is self-inverse)
  - `hdv.cosine_similarity(other)` -> `hdv.similarity(other)`
  - `hdv.permute(shift)` -> `hdv.permute(shift)`
  - `hdv.normalize()` -> No-op (binary vectors are already normalized)
- **Removal timeline**: Float HDV will be removed in a future version

#### BinaryHDV compatibility shims added
- **bind()**: Alias for `xor_bind()` - for legacy API compatibility
- **unbind()**: Alias for `xor_bind()` - XOR is self-inverse
- **cosine_similarity()**: Alias for `similarity()` - returns Hamming-based similarity
- **normalize()**: No-op for binary vectors
- **__xor__()**: Enables `v1 ^ v2` syntax for binding

### Fixed

#### llm_integration.py (6 fixes)
- **Import paths**: Fixed incorrect import paths from `haim.src.core.engine` to `src.core.engine` and `haim.src.core.node` to `src.core.node`
- **Missing import**: Added `from datetime import datetime` for dynamic timestamps
- **Memory access API**: Changed `self.haim.memory_nodes.get()` to `self.haim.tier_manager.get_memory()` at lines 34, 114, 182, 244, 272 - using the correct API for memory access
- **Superposition query**: Replaced non-existent `superposition_query()` call with combined hypotheses retrieval path
- **Concept binding**: Replaced non-existent `bind_concepts()` with placeholder - engine has `bind_memories()` available
- **OR orchestration**: Integrated `orchestrate_orch_or()` from engine and removed workaround sorting path

#### api/main.py (1 fix)
- **Delete endpoint**: Fixed attribute reference from `engine.memory_nodes` to `engine.tier_manager.hot` at line 229 - correct attribute for hot memory tier

#### engine.py (1 fix)
- **Synapse persistence**: Implemented `_save_synapses()` method (lines 369-390) that was previously an empty stub
  - Creates parent directory if it doesn't exist
  - Writes all synapses to disk in JSONL format
  - Includes all synapse attributes: `neuron_a_id`, `neuron_b_id`, `strength`, `fire_count`, `success_count`, `last_fired`
  - Handles errors gracefully with logging

### Changed

- **Dynamic timestamps**: LLM integration now uses `datetime.now().isoformat()` instead of hardcoded timestamp `"2026-02-04"` for accurate temporal tracking
- **Phase 4.3 hardening**:
  - Chrono-weighting uses batched node lookup instead of per-node await chain
  - `include_neighbors` now preserves `top_k` result contract
  - `_dream_sem._value` private access replaced by public `locked()` API
  - Episodic chaining race reduced with serialized store path (`_store_lock`, `_last_stored_id`)
  - `engine_version` in stats updated to `4.3.0`
  - HOT-tier `time_range` filtering enforced in `TierManager.search()`
  - `orchestrate_orch_or()` made async and lock-guarded
