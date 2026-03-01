# MnemoCore v5.0.0 → v5.1.0 Production-Readiness Plan

## Multi-Agent Work Distribution & Complete Remediation Guide

> **Generated:** 2026-02-28
> **Scope:** Full codebase audit — code quality, security, tests, infra, documentation
> **Target:** Production-ready release v5.1.0
> **Method:** 20 parallel agents, each with isolated responsibility domains

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Severity Dashboard](#2-severity-dashboard)
3. [Agent Assignment Matrix](#3-agent-assignment-matrix)
4. [Agent 1 — CRITICAL Bug Fixes](#agent-1--critical-bug-fixes)
5. [Agent 2 — Thread Safety & Concurrency](#agent-2--thread-safety--concurrency)
6. [Agent 3 — Security Hardening](#agent-3--security-hardening)
7. [Agent 4 — Engine Refactoring](#agent-4--engine-refactoring)
8. [Agent 5 — Error Handling & Resilience](#agent-5--error-handling--resilience)
9. [Agent 6 — Storage Layer Fixes](#agent-6--storage-layer-fixes)
10. [Agent 7 — Subconscious/Dream Refactoring](#agent-7--subconsciousdream-refactoring)
11. [Agent 8 — API & Middleware Hardening](#agent-8--api--middleware-hardening)
12. [Agent 9 — Config & Dependency Cleanup](#agent-9--config--dependency-cleanup)
13. [Agent 10 — Docker, Helm & CI/CD](#agent-10--docker-helm--cicd)
14. [Agent 11 — Test Suite: Core Engine](#agent-11--test-suite-core-engine)
15. [Agent 12 — Test Suite: Storage & Import/Export](#agent-12--test-suite-storage--importexport)
16. [Agent 13 — Test Suite: AI & Subconscious](#agent-13--test-suite-ai--subconscious)
17. [Agent 14 — Test Suite: Meta, Cognitive, Events](#agent-14--test-suite-meta-cognitive-events)
18. [Agent 15 — Test Suite: API, MCP, CLI](#agent-15--test-suite-api-mcp-cli)
19. [Agent 16 — Test Suite: Concurrency & Integration](#agent-16--test-suite-concurrency--integration)
20. [Agent 17 — Performance Optimization](#agent-17--performance-optimization)
21. [Agent 18 — Documentation Overhaul](#agent-18--documentation-overhaul)
22. [Agent 19 — Metrics & Observability](#agent-19--metrics--observability)
23. [Agent 20 — Final Validation & Release Prep](#agent-20--final-validation--release-prep)
24. [Dependency Graph Between Agents](#dependency-graph-between-agents)
25. [Complete Issue Registry](#5-complete-issue-registry)

---

## 1. Executive Summary

A full audit of the MnemoCore codebase (88 source files, 75 test files, ~55,000 lines of application code) identified **178 issues** across 6 severity levels. The codebase has strong foundations — good test patterns, modular architecture, comprehensive feature set — but has accumulated technical debt across 6 development phases that must be resolved before a production release.

**Critical blockers:** 11 issues that will cause runtime crashes, data loss, or security breaches in production.

**Key themes:**
- **Thread/async safety** — 12 race conditions, 3 deadlock risks, 5 lock-design flaws
- **Security** — 7 pickle RCE vectors, exposed databases, wildcard CORS, API key leaks
- **God files** — 8 files >800 lines needing decomposition
- **Test gaps** — 24 source files with zero test coverage, 16 with partial coverage
- **Infrastructure drift** — version numbers out of sync across 5 config files
- **Performance** — 6 O(N²) algorithms in hot paths, 4 files with sync I/O in async context

---

## 2. Severity Dashboard

| Severity | Count | Category Breakdown |
|----------|-------|--------------------|
| **CRITICAL** | 11 | 4 bugs, 3 security, 2 thread safety, 1 infra, 1 dep |
| **HIGH** | 49 | 14 thread safety, 10 code quality, 8 security, 7 performance, 5 error handling, 5 infra |
| **MEDIUM** | 76 | Mixed across all categories |
| **LOW** | 42 | Cosmetic, minor type hints, docstrings |

---

## 3. Agent Assignment Matrix

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1 — BLOCKING (Agents 1-3)                      │
│  Must complete before any other agent starts writing code               │
│                                                                         │
│  [Agent 1] CRITICAL Bugs ──► Runtime crashes, data loss                 │
│  [Agent 2] Thread Safety ──► Race conditions, deadlocks                 │
│  [Agent 3] Security ─────► RCE vectors, auth bypass, data exposure      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    PHASE 2 — STRUCTURAL (Agents 4-10)                   │
│  Can run in parallel after Phase 1 merges                               │
│                                                                         │
│  [Agent 4]  Engine refactoring     [Agent 7]  Subconscious refactoring  │
│  [Agent 5]  Error handling         [Agent 8]  API hardening             │
│  [Agent 6]  Storage layer          [Agent 9]  Config & deps             │
│                                    [Agent 10] Docker/Helm/CI            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    PHASE 3 — TESTING (Agents 11-16)                     │
│  Can start after Phase 2 stabilizes; runs in parallel                   │
│                                                                         │
│  [Agent 11] Core engine tests    [Agent 14] Meta/cognitive/events tests │
│  [Agent 12] Storage tests        [Agent 15] API/MCP/CLI tests           │
│  [Agent 13] AI/subconscious      [Agent 16] Concurrency/integration     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    PHASE 4 — POLISH (Agents 17-20)                      │
│  Final pass after all tests pass                                        │
│                                                                         │
│  [Agent 17] Performance          [Agent 19] Metrics/observability       │
│  [Agent 18] Documentation        [Agent 20] Final validation/release    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent 1 — CRITICAL Bug Fixes

**Priority:** P0 — BLOCKING
**Scope:** Fix all issues that cause immediate runtime crashes or data corruption
**Files touched:** 7 files
**Estimated tasks:** 11

### Task 1.1 — `exceptions.py`: Rename shadowed `ImportError`
**File:** `src/mnemocore/core/exceptions.py` ~L460
**Problem:** `class ImportError(MnemoCoreError)` shadows Python's builtin `ImportError`. Any file doing `from .exceptions import *` or importing this class will mask the builtin, causing confusing failures in actual `import` statements.
**Fix:** Rename to `MnemoCoreImportError` or `DependencyImportError`. Update all references across the codebase (grep for `exceptions.ImportError`).
**Verify:** `python -c "from mnemocore.core.exceptions import *; import json"` should not raise.

### Task 1.2 — `engine_coordinator.py`: Fix broken `export_memories`
**File:** `src/mnemocore/core/engine_coordinator.py` ~L480-L517
**Problem 1:** `asyncio.run()` called inside an async coroutine context → raises `RuntimeError: cannot be called from a running event loop`.
**Problem 2:** `functools._run_in_thread` does not exist → raises `AttributeError` at runtime.
**Fix:** Replace `asyncio.run()` with `await`, replace `functools._run_in_thread` with `asyncio.get_event_loop().run_in_executor(None, ...)` or `asyncio.to_thread()`.
**Verify:** Write a test that calls `export_memories()` from an async context and confirms it succeeds.

### Task 1.3 — `engine.py`: Fix frozen dataclass mutation
**File:** `src/mnemocore/core/engine.py` ~L331-L336
**Problem:** `enable_reconstructive_memory()` does `setattr` on a `frozen=True` dataclass (`ReconstructionConfig`) → `FrozenInstanceError` at runtime.
**Fix:** Use `dataclasses.replace()` to create a new config instance with the modified fields.
**Verify:** Test `enable_reconstructive_memory()` call completes without error.

### Task 1.4 — `tier_eviction.py`: Fix attribute typo
**File:** `src/mnemocore/core/tier_eviction.py` ~L260
**Problem:** `get_warm_strategy()` references `self.warm_strategy` which doesn't exist — should be `self.warm_policy`. Crashes on any WARM tier eviction.
**Fix:** Change `self.warm_strategy` to `self.warm_policy`.
**Verify:** Test `get_warm_strategy()` returns the configured policy.

### Task 1.5 — `logging_config.py`: Disable `diagnose=True` in production
**File:** `src/mnemocore/core/logging_config.py` ~L82
**Problem:** Loguru's `diagnose=True` includes local variable values in tracebacks — leaks API keys, memory content, user data into log files/centralized logging.
**Fix:** Set `diagnose=False`. Optionally make it `diagnose=(os.getenv("LOG_LEVEL", "").upper() == "DEBUG")`.
**Also:** Set `backtrace=False` by default for the same reason (~L81).
**Verify:** Trigger an exception and confirm traceback does NOT include local variable values.

### Task 1.6 — `meta/goal_tree.py`: Fix ID collision
**File:** `src/mnemocore/meta/goal_tree.py` ~L77
**Problem:** `goal_id = f"goal_{len(self.goals)}"` — deleting a goal and adding a new one reuses the same ID, corrupting references.
**Fix:** Use `uuid.uuid4().hex[:12]` or a monotonic counter stored in the state file.
**Verify:** Create 3 goals → delete goal 2 → create goal 4 → all IDs are unique.

### Task 1.7 — `meta/learning_journal.py`: Fix ID collision
**File:** `src/mnemocore/meta/learning_journal.py` ~L54
**Problem:** Same `entry_id = f"learn_{len(self.entries)}"` bug as goal_tree.
**Fix:** Same as Task 1.6 — use UUID or monotonic counter.
**Verify:** Same deletion+recreation test.

### Task 1.8 — `docker-compose.yml`: Fix Redis/Qdrant exposed without auth
**File:** `docker-compose.yml`
**Problem:** Redis starts without `--requirepass`. Qdrant has no API key. Both are port-forwarded to the host. Any local process can read/write all data.
**Fix:**
- Redis: Add `command: redis-server --requirepass ${REDIS_PASSWORD:-changeme}` and set `REDIS_URL=redis://:${REDIS_PASSWORD:-changeme}@redis:6379/0` in mnemocore environment.
- Qdrant: Add `environment: - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY:-changeme}` and configure the MnemoCore Qdrant client to send the key.
- Remove host port bindings or bind to `127.0.0.1` only.
**Verify:** `redis-cli -p 6379 PING` returns `NOAUTH` error. Qdrant `GET /collections` without key returns 403.

### Task 1.9 — `Dockerfile`: Fix healthcheck path
**File:** `Dockerfile`
**Problem:** Healthcheck references `/app/scripts/healthcheck.py` but the actual file is at `/app/scripts/ops/healthcheck.py`. Container health will always report "unhealthy".
**Fix:** Change to `/app/scripts/ops/healthcheck.py`.
**Verify:** Build image and run `docker inspect --format='{{.State.Health.Status}}' <container>` returns "healthy".

### Task 1.10 — `pyproject.toml`: Add missing `aiohttp` dependency
**File:** `pyproject.toml`
**Problem:** `aiohttp` is imported by `subconscious_ai.py`, `daemon.py`, `webhook_manager.py` but is NOT listed in `[project.dependencies]`. A clean install from the private GitHub repository will fail at runtime.
**Fix:** Add `"aiohttp>=3.9.0"` to the `dependencies` list in `pyproject.toml`. Also add `"psutil>=5.9.0"` which is similarly missing.
**Verify:** `pip install -e .` in a clean venv → `python -c "from mnemocore.core.subconscious_ai import SubconsciousAIWorker"` succeeds.

### Task 1.11 — `dream_scheduler.py`: Fix AttributeError in init log
**File:** `src/mnemocore/subconscious/dream_scheduler.py` ~L508
**Problem:** References `self.cfg.idle_threshold` but the attribute name is `idle_threshold_seconds` → `AttributeError` on scheduler init.
**Fix:** Change to `self.cfg.idle_threshold_seconds`.
**Verify:** Instantiate `DreamScheduler` with default config and confirm no error.

---

## Agent 2 — Thread Safety & Concurrency

**Priority:** P0 — BLOCKING
**Scope:** Fix all race conditions, deadlock risks, and lock-design flaws
**Files touched:** 10 files
**Estimated tasks:** 12

### Task 2.1 — `reliability.py`: Make circuit breaker thread-safe
**File:** `src/mnemocore/core/reliability.py`
**Problem:** `NativeCircuitBreaker._check_state()` reads/mutates `state`, `failures`, timestamps without locking. State transitions are lost under concurrent access to a component whose sole purpose is concurrency safety.
**Fix:** Add `threading.Lock` around state transitions. Switch `time.time()` to `time.monotonic()`.
**Verify:** Test concurrent access from 50 threads to a breaker with a failing service. Assert state transitions are correct.

### Task 2.2 — `tier_manager.py`: Fix TOCTOU race conditions
**File:** `src/mnemocore/core/tier_manager.py` ~L400-L628
**Problem:** `get_memory()` checks `hot_node.tier == "warm"` inside the lock but demotion I/O happens outside. `_promote_to_hot()` deletes from WARM outside the lock then acquires lock for HOT insertion — duplicates possible.
**Fix:** Move all tier-transition I/O (both read and write) inside the same lock acquisition. Use a single `async with self._lock:` block that covers the entire promote/demote operation.
**Verify:** Stress test: 100 concurrent `get_memory` + `store` operations. Assert no duplicates or lost nodes.

### Task 2.3 — `tier_storage.py`: Fix double-lock deadlock risk
**File:** `src/mnemocore/core/tier_storage.py`
**Problem:** `HotTierStorage` has its own `asyncio.Lock` while `TierManager` also holds a separate lock. If acquisition order is ever inverted, deadlock. Also, `get_storage_dict()` returns a mutable reference bypassing all locks.
**Fix:** Remove `HotTierStorage`'s internal lock — let `TierManager`'s lock be the single source of truth. Make `get_storage_dict()` return a copy or a read-only view.
**Verify:** Run tier tests under `-X dev` with `PYTHONASYNCIODEBUG=1`. No deadlock warnings.

### Task 2.4 — `hnsw_index.py`: Fix data races
**File:** `src/mnemocore/core/hnsw_index.py`
**Problem:** `search()` reads `_stale_count` without lock. `_save()` and `_load()` use bare `except Exception`.
**Fix:** Protect `_stale_count` reads with the existing lock. Replace bare excepts with specific exceptions (`IOError`, `faiss.FaissError`).
**Verify:** Concurrent add+search+remove test with assertions on search result validity.

### Task 2.5 — `synapse_index.py`: Add internal locking
**File:** `src/mnemocore/core/synapse_index.py`
**Problem:** Documentation says "use under external asyncio.Lock" but no enforcement. `_get_bayesian_updater()` re-imports on every call.
**Fix:** Add internal `asyncio.Lock`. Cache the bayesian updater after first import.
**Verify:** Concurrent `add_or_fire()` + `get_neighbors()` test.

### Task 2.6 — `holographic.py`: Fix concurrent file corruption
**File:** `src/mnemocore/core/holographic.py`
**Problem:** `save()` writes entire codebook JSON on every `store_concept()`. No file locking. Concurrent writes corrupt the file.
**Fix:** Add `asyncio.Lock` for file operations. Debounce writes (flush on `close()` or every N seconds, not every call). Use atomic write (write to temp file then rename).
**Verify:** Concurrent `store_concept()` test. Verify file remains valid JSON after 100 concurrent writes.

### Task 2.7 — `working_memory.py`: Replace threading lock with async lock
**File:** `src/mnemocore/core/working_memory.py`
**Problem:** Uses `threading.RLock` but called from async coroutines. Blocks the event loop.
**Fix:** Replace with `asyncio.Lock` if purely async, or use `asyncio.Lock` with `run_in_executor` for the blocking operations. Audit all callers.
**Verify:** Working memory operations don't block the event loop (test with asyncio debug mode).

### Task 2.8 — `procedural_store.py`: Replace sync I/O lock with async
**File:** `src/mnemocore/core/procedural_store.py`
**Problem:** `_persist_to_disk()` does sync JSON write under `threading.RLock`, blocking async callers.
**Fix:** Use `asyncio.Lock` + `aiofiles` for async file writes. Debounce writes.
**Verify:** Store 100 procedures rapidly. No event loop blocking. File remains valid.

### Task 2.9 — `strategy_bank.py`: Debounce disk persistence
**File:** `src/mnemocore/core/strategy_bank.py`
**Problem:** `_persist_to_disk()` called after every `judge_retrieval()`, `distill_from_episode()`, `store_strategy()`. Synchronous, serializes all writes.
**Fix:** Implement a dirty flag + periodic flush (every 5 seconds or on `close()`). Use atomic write.
**Verify:** High-throughput judgment test. Verify persistence works and no data loss.

### Task 2.10 — `event_bus.py`: Fix global singleton thread safety
**File:** `src/mnemocore/events/event_bus.py` ~L714
**Problem:** `_EVENT_BUS` global is not thread-safe. Concurrent first access creates multiple instances.
**Fix:** Use `threading.Lock` around singleton creation.
**Verify:** Spawn 100 threads, all call `get_event_bus()`. Assert `id()` is identical.

### Task 2.11 — `config.py`: Fix global config thread safety
**File:** `src/mnemocore/core/config.py` ~L1315-L1326
**Problem:** `_CONFIG` singleton with `get_config()` / `reset_config()` is not thread-safe.
**Fix:** Use `threading.Lock` around config access and reset.
**Verify:** Concurrent config access test.

### Task 2.12 — `future_thinking.py`: Fix reentrant lock deadlock
**File:** `src/mnemocore/cognitive/future_thinking.py`
**Problem:** `stats()` acquires `self._lock`, then calls `await self.list_active()` which also acquires `self._lock` → deadlock (asyncio.Lock is not reentrant).
**Fix:** Factor out the unlocked version of `list_active()` as `_list_active_unlocked()` and call that from `stats()`.
**Verify:** Call `stats()` — should not deadlock.

---

## Agent 3 — Security Hardening

**Priority:** P0 — BLOCKING
**Scope:** Eliminate all RCE vectors, auth bypasses, data exposure
**Files touched:** 11 files
**Estimated tasks:** 13

### Task 3.1 — `vector_compression.py`: Remove pickle deserialization (RCE)
**File:** `src/mnemocore/storage/vector_compression.py` ~L104-L113, L743-L810
**Problem:** `CompressedVector.to_bytes()/.from_bytes()` and `VectorCompressor.save()/.load()` use `pickle.dumps`/`pickle.loads` — arbitrary code execution if data is tampered with.
**Fix:** Replace with `numpy.save`/`numpy.load` for vector data and `json` or `msgpack` for metadata. Define a safe binary format.
**Verify:** Round-trip test: compress → save → load → decompress. Verify corrupt file raises clean error, not code execution.

### Task 3.2 — `binary_vector_compression.py`: Remove pickle deserialization (RCE)
**File:** `src/mnemocore/storage/binary_vector_compression.py` ~L637, L664
**Problem:** `pickle.dumps`/`pickle.loads` for PQ codebook serialization — same RCE risk.
**Fix:** Store codebooks as numpy `.npy` format in SQLite BLOBs (using `numpy.frombuffer`/`numpy.tobytes`). Store metadata separately as JSON.
**Verify:** Same round-trip test. Corrupt data does not execute code.

### Task 3.3 — `config.py`: Change CORS default from `["*"]`
**File:** `src/mnemocore/core/config.py` ~L102
**Problem:** `SecurityConfig.cors_origins` defaults to `["*"]` — allows all origins in production.
**Fix:** Default to `["http://localhost:3000", "http://localhost:8100"]` for dev. Document that production deploys must set explicit origins.
**Verify:** Start API without setting CORS. Verify cross-origin request from `evil.com` is rejected.

### Task 3.4 — `config.py`: Remove YAML-based API key support
**File:** `src/mnemocore/core/config.py` ~L101, L77, L238
**Problem:** `SecurityConfig.api_key`, `QdrantConfig.api_key`, `SubconsciousAIConfig.api_key` can be set in YAML config which may be committed to version control.
**Fix:** Make these fields read from env vars ONLY. If set in YAML, log a warning and ignore. Add `DeprecationWarning`.
**Verify:** Set API key in YAML → warning logged. Set via env var → works. Both set → env var wins with warning.

### Task 3.5 — `daemon.py`: Remove `sys.path.insert` hack
**File:** `src/mnemocore/subconscious/daemon.py` ~L20
**Problem:** `sys.path.insert(0, ...)` is fragile and enables import hijacking.
**Fix:** Remove the `sys.path` manipulation. Ensure proper package installation via `pyproject.toml` and use absolute imports.
**Also fix:** `src/mnemocore/api/main.py` ~L22 — same issue.
**Verify:** Run daemon and API server without `sys.path` hacks. All imports resolve.

### Task 3.6 — `daemon.py`: Fix hardcoded `/tmp` log path
**File:** `src/mnemocore/subconscious/daemon.py` ~L37
**Problem:** `LOG_PATH = "/tmp/subconscious.log"` — world-writable directory, Unix-only, no rotation or size limit. Potential log injection or symlink attack.
**Fix:** Use `config.yaml` or env var for log path. Default to `./data/subconscious.log`. Add log rotation (via loguru or custom).
**Verify:** Log file created in configured path. Old file rotated correctly.

### Task 3.7 — `api_adapter.py`: Fix URL injection + enforce HTTPS
**File:** `src/mnemocore/mcp/adapters/api_adapter.py` ~L93, L109
**Problem:** `query` parameter directly interpolated into URL without encoding. API key sent without enforcing HTTPS.
**Fix:** Use `urllib.parse.urlencode()` for all query parameters. Add `if not base_url.startswith("https")` warning/error in production mode.
**Verify:** Query with special chars (`?`, `&`, `#`, spaces) correctly encoded. HTTPS enforcement test.

### Task 3.8 — `middleware.py`: Fix X-Forwarded-For spoofing
**File:** `src/mnemocore/api/middleware.py` ~L152
**Problem:** Rate limiter trusts `X-Forwarded-For` header which can be spoofed to bypass rate limits.
**Fix:** Add configurable `trusted_proxies` list. Only use `X-Forwarded-For` if `request.client.host` is in the trusted list. Default: trust nothing.
**Verify:** Test: send request with spoofed `X-Forwarded-For` from untrusted IP. Rate limit applies to actual IP.

### Task 3.9 — `mcp/server.py`: Sanitize error messages
**File:** `src/mnemocore/mcp/server.py` ~L56-L61
**Problem:** `with_error_handling` returns `str(exc)` which can contain internal paths, stack traces, SQL queries.
**Fix:** Return generic error message. Log the full exception internally. In debug mode only, include details.
**Verify:** Trigger an error via MCP. Response contains "Internal error" not file paths.

### Task 3.10 — `webhook_manager.py`: Encrypt secrets at rest
**File:** `src/mnemocore/events/webhook_manager.py` ~L1017
**Problem:** Webhook secrets stored in plaintext JSON on disk.
**Fix:** Use `cryptography.fernet` to encrypt secrets before writing. Key derived from `HAIM_API_KEY` or a dedicated `WEBHOOK_ENCRYPTION_KEY` env var.
**Verify:** Read webhook file — secrets are encrypted, not plaintext.

### Task 3.11 — `backup_manager.py`: Fix SQL injection
**File:** `src/mnemocore/storage/backup_manager.py` ~L361
**Problem:** `get_entries_since()` uses f-string for `LIMIT {limit}` — SQL injection vector.
**Fix:** Use parameterized query: `cursor.execute("... LIMIT ?", (limit,))`.
**Verify:** Pass `limit="1; DROP TABLE wal_entries"` — query parameterized, no injection.

### Task 3.12 — `api/main.py`: Fix hardcoded CORS `["*"]` fallback
**File:** `src/mnemocore/api/main.py` ~L253
**Problem:** Falls back to `["*"]` if config doesn't have `cors_origins`.
**Fix:** Fall back to `[]` (deny all) or `["http://localhost:8100"]`. Never `["*"]`.
**Verify:** Start API without CORS config. Cross-origin request rejected.

### Task 3.13 — `api/main.py`: Require API key at startup
**File:** `src/mnemocore/api/main.py` + `Dockerfile`
**Problem:** Empty `HAIM_API_KEY=""` default ENV means API runs without auth.
**Fix:** On startup, if `HAIM_API_KEY` is empty/unset and `config.security.require_api_key` is True (default), log error and exit. The Dockerfile should NOT set a default.
**Verify:** Start without `HAIM_API_KEY` → fails with clear error. Set key → starts normally.

---

## Agent 4 — Engine Refactoring

**Priority:** P1 — STRUCTURAL
**Scope:** Break up god functions, eliminate dead code, fix duplication in engine modules
**Files touched:** 5 files
**Estimated tasks:** 9

### Task 4.1 — `engine_core.py`: Decompose `store()` (100+ lines)
**File:** `src/mnemocore/core/engine_core.py` ~L420-L525
**Problem:** Single function handles validation, encoding, persistence, WM push, episodic logging, semantic consolidation, event emission, association update, meta-memory. SRP violation.
**Fix:** Extract into pipeline stages:
- `_validate_content(content) → str`
- `_encode_content(content) → BinaryHDV`
- `_persist_memory(node) → str` (returns ID)
- `_post_store_hooks(node)` (WM, episodic, semantic, events, associations, meta)
**Verify:** All existing store tests pass. No behavior change.

### Task 4.2 — `engine_core.py`: Decompose `query()` (170+ lines)
**File:** `src/mnemocore/core/engine_core.py` ~L540-L780
**Problem:** Monolithic query function with 12 stages.
**Fix:** Extract into a query pipeline:
- `_encode_query(query_text) → BinaryHDV`
- `_search_tiers(vector, top_k) → List[MemoryNode]`
- `_apply_temporal_weighting(results) → List`
- `_apply_preference_bias(results) → List`
- `_apply_wm_boost(results) → List`
- `_apply_attention_rerank(results) → List`
- `_trigger_post_query(results)` (async hooks)
**Verify:** All existing query tests pass. Same results for same inputs.

### Task 4.3 — `engine_core.py`: Add exception callbacks to `ensure_future`
**File:** `src/mnemocore/core/engine_core.py` ~L766-L780
**Problem:** `asyncio.ensure_future()` fire-and-forget — unhandled exceptions silently lost.
**Fix:** Add `task.add_done_callback(_log_task_exception)` helper. Or use `asyncio.TaskGroup` (Python 3.11+).
**Verify:** Force an exception in a post-query hook. Confirm it appears in logs.

### Task 4.4 — `engine_core.py`: Remove duplicate `_run_in_thread` and `_get_token_vector`
**File:** `src/mnemocore/core/engine_core.py` L170, `engine_coordinator.py` L695, `engine_lifecycle.py`
**Problem:** `_run_in_thread` defined in 3 files. `_get_token_vector` defined in 2 files.
**Fix:** Move to a shared utility (`core/_utils.py`) and import.
**Verify:** Grep confirms no duplicates. All callers use the shared version.

### Task 4.5 — `engine.py`: Clean up unused imports
**File:** `src/mnemocore/core/engine.py` ~L13-L20
**Problem:** `uuid`, `Deque`, `deque`, `asyncio`, `np`, `replace`, `datetime`, `timezone`, `Path` imported but unused.
**Fix:** Remove all unused imports.
**Verify:** `flake8 --select F401 src/mnemocore/core/engine.py` returns clean.

### Task 4.6 — `engine.py`: Extract inline imports from `__init__`
**File:** `src/mnemocore/core/engine.py` ~L158-L195
**Problem:** `TopicTracker`, `PreferenceStore`, `AnticipatoryEngine`, `AssociationsNetwork`, `ConceptualMemory` imported inside `__init__`. Hides dependencies.
**Fix:** Move to top-level imports with `TYPE_CHECKING` guard if needed for circular deps.
**Verify:** All engine tests pass. No circular import errors.

### Task 4.7 — `engine.py`: Replace `getattr` config access
**File:** `src/mnemocore/core/engine.py` ~L145, L185-L186
**Problem:** `getattr(self.config, 'attention_masking', None)` suggests config schema drift.
**Fix:** Add all used config fields to `HAIMConfig` dataclass with proper defaults. Remove `getattr` fallbacks.
**Verify:** Config tests cover the newly-added fields.

### Task 4.8 — `config.py`: Refactor `load_config()` god function
**File:** `src/mnemocore/core/config.py` ~L745-L1320
**Problem:** 330-line function manually building ~30 config objects with repetitive `_env_override` calls.
**Fix:** Extract a generic `_load_section(yaml_dict, section_key, dataclass_cls, env_prefix) → T` helper that:
1. Reads YAML section
2. Applies env overrides for each field
3. Constructs the dataclass
Then `load_config()` becomes a simple list of `_load_section()` calls.
**Verify:** All config tests pass. All env overrides still work.

### Task 4.9 — `config.py`: Add config validation
**File:** `src/mnemocore/core/config.py`
**Problem:** No validation — negative `max_memories`, negative intervals, invalid policy strings are silently accepted.
**Fix:** Add `__post_init__` validation to each config dataclass. Raise `ConfigurationError` with descriptive messages.
**Verify:** Test: negative `max_memories` → `ConfigurationError`. Test: invalid `eviction_policy` → `ConfigurationError`.

---

## Agent 5 — Error Handling & Resilience

**Priority:** P1 — STRUCTURAL
**Scope:** Fix swallowed exceptions, bare excepts, misleading health checks
**Files touched:** 12 files
**Estimated tasks:** 10

### Task 5.1 — `engine_core.py`: Replace silent exception swallowing
**File:** `src/mnemocore/core/engine_core.py` ~L513-L514, L497, L506, L832, L940
**Problem:** `except Exception: pass` and `except Exception as e: logger.debug(...)` — silent failures in production.
**Fix:** Change `pass` to `logger.warning(...)`. Change `debug` to `warning` for operational failures. Add metrics counter `mnemocore_operation_errors_total` by operation type.
**Verify:** Force a meta-memory error. Confirm warning appears in logs and metric increments.

### Task 5.2 — `engine_lifecycle.py`: Harden `close()` shutdown
**File:** `src/mnemocore/core/engine_lifecycle.py` ~L162-L171
**Problem:** If one worker's `stop()` raises, subsequent workers are never stopped.
**Fix:** Wrap each `stop()` in individual try/except. Log errors but continue shutting down all workers.
**Verify:** Mock one worker to raise on `stop()`. Confirm all other workers still stopped.

### Task 5.3 — `engine_lifecycle.py`: Fix lying health check
**File:** `src/mnemocore/core/engine_lifecycle.py` ~L325-L345
**Problem:** Multiple `except Exception` blocks report `{"status": "active"}` even on failure.
**Fix:** On failure, report `{"status": "degraded", "error": str(e)}` for each component. Overall status should be "degraded" if any component fails, "error" if a critical component fails.
**Verify:** Mock Qdrant failure. Health check returns `{"status": "degraded", "components": {"qdrant": {"status": "error"}}}`.

### Task 5.4 — `engine_lifecycle.py`: Roll back `_initialized` on failure
**File:** `src/mnemocore/core/engine_lifecycle.py` ~L95-L115
**Problem:** If a background worker fails to start, `_initialized` is still `True`.
**Fix:** Wrap the entire init in try/except. On failure, set `_initialized = False` and stop all already-started workers.
**Verify:** Test init with a failing worker. Confirm `_initialized` is `False` and cleanup was called.

### Task 5.5 — `engine_lifecycle.py`: Fix swallowed I/O errors in `_load_legacy`
**File:** `src/mnemocore/core/engine_lifecycle.py` ~L120-L123
**Problem:** Inner `_load()` has `except Exception: return []` — file I/O errors completely swallowed.
**Fix:** Catch `(IOError, json.JSONDecodeError)` specifically. Log at warning level with file path.
**Verify:** Provide a corrupt JSON file. Confirm warning logged with file path and error.

### Task 5.6 — `daemon.py`: Replace bare `except:` blocks
**File:** `src/mnemocore/subconscious/daemon.py` ~L229, L260, L345
**Problem:** Three bare `except:` blocks silently swallow JSON parse errors from LLM output.
**Fix:** Change to `except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:` with `logger.warning(...)`.
**Verify:** Feed daemon malformed LLM output. Confirm warning logged, daemon doesn't crash.

### Task 5.7 — `llm_integration.py`: Return structured errors instead of error strings
**File:** `src/mnemocore/llm_integration.py` ~L300
**Problem:** `_call_llm()` catches `Exception` and returns `"[LLM Error: {str(e)}]"` — callers treat this as valid content.
**Fix:** Raise a custom `LLMError` exception. Callers handle it explicitly. Add `LLMError` to `exceptions.py`.
**Verify:** Test: LLM call fails → caller receives `LLMError`, not a string.

### Task 5.8 — `meta/goal_tree.py` + `learning_journal.py`: Add JSON error handling
**File:** Both files, `_load()` method
**Problem:** No try/except around `json.load()`. Malformed JSON crashes the app at init.
**Fix:** `except json.JSONDecodeError as e: logger.warning("..."); self.goals = {}` (or `self.entries = {}`).
**Verify:** Test with corrupt JSON file. Confirm graceful degradation, not crash.

### Task 5.9 — `backup_manager.py`: Replace broad exception swallowing
**File:** `src/mnemocore/storage/backup_manager.py` ~L288, L800
**Problem:** `_log_operation` returns `False` on any error. `list_snapshots` returns `[]` on any error. No error detail.
**Fix:** Log errors at warning level. Use `with sqlite3.connect()` context manager for connection safety.
**Verify:** Simulate DB write failure. Confirm warning logged with error details.

### Task 5.10 — `prediction_store.py`: Improve error logging
**File:** `src/mnemocore/core/prediction_store.py`
**Problem:** `_generate_lesson()`, `_strengthen_related()`, `_weaken_related()` catch `except Exception` with debug logging only.
**Fix:** Log at `warning` level. Include the prediction ID and operation name in the log message.
**Verify:** Force a lesson generation error. Confirm warning appears.

---

## Agent 6 — Storage Layer Fixes

**Priority:** P1 — STRUCTURAL
**Scope:** Fix async/sync mismatches, blocking I/O, data integrity issues in storage
**Files touched:** 7 files
**Estimated tasks:** 9

### Task 6.1 — `backup_manager.py`: Replace blocking SQLite with aiosqlite
**File:** `src/mnemocore/storage/backup_manager.py` ~L192, L800+
**Problem:** `sqlite3.Connection` with `check_same_thread=False` shared across async methods. All SQLite operations block the event loop.
**Fix:** Replace all `sqlite3` usage with `aiosqlite`. Convert synchronous `cursor.execute()` to `await cursor.execute()`. Add `aiosqlite>=0.19.0` to requirements.
**Verify:** WAL write test under async. No event loop blocking (test with asyncio debug mode).

### Task 6.2 — `binary_vector_compression.py`: Replace blocking SQLite
**File:** `src/mnemocore/storage/binary_vector_compression.py` ~L414
**Problem:** Same blocking SQLite pattern as backup_manager.
**Fix:** Same solution — `aiosqlite`.
**Verify:** Compression/decompression test under async.

### Task 6.3 — `binary_vector_compression.py`: Fix `cleanup_old_codebooks` bug
**File:** `src/mnemocore/storage/binary_vector_compression.py` ~L862
**Problem:** `fetchone()` used but then iterated with `for (codebook_id,) in rows:` — `fetchone` returns a single row, not a list. Likely should be `fetchall()`.
**Fix:** Change `fetchone()` to `fetchall()`. Add test.
**Verify:** Test with multiple old codebooks. All are cleaned up, not just one.

### Task 6.4 — `memory_importer.py`: Fix dangling import + streaming reads
**File:** `src/mnemocore/storage/memory_importer.py` ~L877, L910
**Problem:** `import os` at bottom of file (dead/misplaced). `_generate_import_id` uses `if 'os' in globals()` — fragile. `_read_json` reads entire file into memory.
**Fix:** Move `import os` to top. Remove `if 'os' in globals()` guard. Make `_read_json` stream using `ijson` library or read in chunks.
**Verify:** Import a 100MB JSON file without OOM. `os` import is at top.

### Task 6.5 — `memory_exporter.py`: Stream large exports
**File:** `src/mnemocore/storage/memory_exporter.py` ~L256
**Problem:** `_export_json` accumulates all records in memory before writing.
**Fix:** Write records incrementally using JSONL format or streaming JSON writer.
**Verify:** Export 100K memories. Peak memory usage < 200MB.

### Task 6.6 — `qdrant_store.py`: Prevent destructive collection recreation
**File:** `src/mnemocore/core/qdrant_store.py`
**Problem:** `_ensure_collections()` silently drops and recreates collections when distance metric doesn't match config → destroys all existing vectors.
**Fix:** If metric mismatch detected, raise `ConfigurationError` with instructions to either change config or explicitly migrate. Never auto-drop.
**Verify:** Test: create collection with Cosine, change config to Dot Product. Confirm error, not data loss.

### Task 6.7 — `memory_importer.py`: Replace SQLite with aiosqlite
**File:** `src/mnemocore/storage/memory_importer.py`
**Problem:** `ImportLog` uses synchronous `sqlite3.connect` from async context.
**Fix:** Same `aiosqlite` pattern as backup_manager.
**Verify:** Import test under async.

### Task 6.8 — `webhook_manager.py`: Fix aiohttp timeout misuse
**File:** `src/mnemocore/events/webhook_manager.py` ~L797
**Problem:** `aiohttp.ClientSession(total=...)` — `total` is not a `ClientSession` parameter. Should be `aiohttp.ClientTimeout(total=...)`.
**Fix:** Change to `timeout = aiohttp.ClientTimeout(total=self.delivery_timeout)` then `async with aiohttp.ClientSession(timeout=timeout)`.
**Verify:** Webhook delivery test with short timeout. Verify timeout is enforced.

### Task 6.9 — `webhook_manager.py`: Queue retries instead of inline sleep
**File:** `src/mnemocore/events/webhook_manager.py` ~L767
**Problem:** Retry with `asyncio.sleep(delay)` blocks the coroutine for up to 300s.
**Fix:** Push failed deliveries to a retry queue processed by a background task with exponential backoff.
**Verify:** Webhook delivery failure → retry happens asynchronously without blocking caller.

---

## Agent 7 — Subconscious/Dream Refactoring

**Priority:** P1 — STRUCTURAL
**Scope:** Break up god files, fix cron parser, clean up daemon
**Files touched:** 5 files
**Estimated tasks:** 8

### Task 7.1 — `dream_pipeline.py`: Split into per-stage modules
**File:** `src/mnemocore/subconscious/dream_pipeline.py` (1336 lines, 8 classes)
**Problem:** God file.
**Fix:** Create `subconscious/dream/` package:
- `subconscious/dream/__init__.py` — public exports
- `subconscious/dream/clusterer.py` — `EpisodicClusterer`
- `subconscious/dream/patterns.py` — `PatternExtractor`
- `subconscious/dream/synthesizer.py` — `DreamSynthesizer`
- `subconscious/dream/contradictions.py` — `ContradictionResolver`
- `subconscious/dream/promoter.py` — `SemanticPromoter`
- `subconscious/dream/report.py` — `DreamReportGenerator`
- `subconscious/dream/pipeline.py` — `DreamPipeline` (orchestrator)
**Verify:** All existing dream tests pass. No import changes needed for external consumers (re-export from `__init__`).

### Task 7.2 — `dream_scheduler.py`: Replace buggy cron parser
**File:** `src/mnemocore/subconscious/dream_scheduler.py` ~L580-L640
**Problem:** Custom cron parser doesn't handle ranges, `*/N` syntax, and produces incorrect next-run times.
**Fix:** Replace with `croniter` library (`pip install croniter`). Add to `requirements.txt` and `pyproject.toml`.
**Verify:** Test: `*/5 * * * *` schedules every 5 minutes. `0 2 * * *` schedules daily at 2 AM.

### Task 7.3 — `daemon.py`: Remove hardcoded paths
**File:** `src/mnemocore/subconscious/daemon.py` ~L34-L38
**Problem:** `HAIM_DATA_PATH`, `DEFAULT_CYCLE_INTERVAL`, `EVOLUTION_STATE_PATH`, `LOG_PATH` all hardcoded.
**Fix:** Read from config or env vars. Fall back to defaults from config.
**Verify:** Set `SUBCONSCIOUS_DATA_PATH` env var. Daemon uses it.

### Task 7.4 — `daemon.py`: Reuse aiohttp session
**File:** `src/mnemocore/subconscious/daemon.py` ~L207
**Problem:** New `aiohttp.ClientSession` created on every `query_ollama` call.
**Fix:** Create session once in `__init__` or `start()`. Reuse across calls. Close in `stop()`.
**Verify:** 100 consecutive queries. Connection pool reused (verify via aiohttp debug logs).

### Task 7.5 — `daemon.py`: Fix signal handler for Windows compatibility
**File:** `src/mnemocore/subconscious/daemon.py` ~L683
**Problem:** `signal.signal(SIGTERM)` is not safe to call from async context on Windows.
**Fix:** Use `loop.add_signal_handler()` on Unix. On Windows, use alternative (e.g., `signal.SIGBREAK` or skip signal handling with a note).
**Verify:** Daemon starts on Windows without signal handler error.

### Task 7.6 — `llm_integration.py`: Split into per-provider modules
**File:** `src/mnemocore/llm_integration.py` (1102 lines, 6 classes)
**Problem:** God file mixing OllamaClient, HAIMLLMIntegrator, MultiAgentHAIM, ContextAwareLLMIntegrator, RLMIntegrator.
**Fix:** Create `src/mnemocore/llm/` package:
- `llm/__init__.py` — public exports
- `llm/ollama.py` — `OllamaClient`
- `llm/integrator.py` — `HAIMLLMIntegrator` (base)
- `llm/multi_agent.py` — `MultiAgentHAIM`
- `llm/context_aware.py` — `ContextAwareLLMIntegrator`
- `llm/rlm.py` — `RLMIntegrator`
**Backward compat:** Re-export everything from `llm_integration.py` (import from new package).
**Verify:** All imports that currently reference `llm_integration` still work.

### Task 7.7 — `forgetting_curve.py`: Split god file
**File:** `src/mnemocore/core/forgetting_curve.py` (1501 lines)
**Problem:** 6+ classes in one file.
**Fix:** Split into:
- `forgetting_curve.py` — `ForgettingCurveManager`, `SM2State`, `ReviewEntry`, `LearningProfile`
- `forgetting_analytics.py` — `ForgettingAnalytics`, `RetentionCurve`, `AgentAnalytics`
Note: `cognitive/forgetting_analytics.py` already exists — merge or redirect.
**Verify:** All tests pass. No broken imports.

### Task 7.8 — `consolidation.py`: Remove dead code
**File:** `src/mnemocore/core/consolidation.py`
**Problem:** `batch_size` variable assigned but never used.
**Fix:** Remove dead variable.
**Verify:** `flake8 --select F841` clean.

---

## Agent 8 — API & Middleware Hardening

**Priority:** P1 — STRUCTURAL
**Scope:** Decompose API god file, fix inline models, harden middleware
**Files touched:** 6+ files
**Estimated tasks:** 8

### Task 8.1 — `api/main.py`: Split into route modules
**File:** `src/mnemocore/api/main.py` (1577 lines)
**Problem:** All routes in one file.
**Fix:** Create route modules:
- `api/routes/__init__.py`
- `api/routes/memories.py` — `/store`, `/query`, `/memories/{id}`, `/memories/{id}/delete`
- `api/routes/episodes.py` — `/episodes/*`
- `api/routes/observations.py` — `/observe`
- `api/routes/dreams.py` — `/dream`
- `api/routes/health.py` — `/health`, `/stats`
- `api/routes/export.py` — `/export`
- `api/routes/procedures.py` — `/procedures/*`
- `api/routes/predictions.py` — `/predictions/*`
Use `APIRouter` and `include_router()` in main.py.
**Verify:** All API functional tests pass. All endpoints respond at same paths.

### Task 8.2 — `api/main.py`: Move inline models to `models.py`
**File:** `src/mnemocore/api/main.py`
**Problem:** `ObserveRequest`, `EpisodeStartRequest`, `ProposalStatusUpdate`, `DreamRequest`, `RLMQueryRequest` defined inline.
**Fix:** Move all request/response models to `api/models.py`. Import in route modules.
**Verify:** All API tests pass.

### Task 8.3 — `api/main.py`: Fix version string inconsistency
**File:** `src/mnemocore/api/main.py` ~L167, L297
**Problem:** `FastAPI(version="3.5.2")` and `"engine_version": "3.5.1"` — both wrong (should be 5.0.0).
**Fix:** Read version from `importlib.metadata.version("mnemocore")` or a central `__version__` variable.
**Verify:** `/health` returns correct version matching `pyproject.toml`.

### Task 8.4 — `api/main.py`: Fix prediction store race condition
**File:** `src/mnemocore/api/main.py` ~L1040
**Problem:** `_prediction_store_instance` global is mutated without lock.
**Fix:** Use `functools.lru_cache()` for singleton pattern or `asyncio.Lock` for init.
**Verify:** 100 concurrent requests hitting prediction endpoint. Only one instance created.

### Task 8.5 — `api/main.py`: Add rate limiting to `/dream`
**File:** `src/mnemocore/api/main.py`
**Problem:** `/dream` triggers expensive LLM calls with no rate limit.
**Fix:** Add rate limit decorator (e.g., 5 calls per minute per user via existing `RateLimiter`).
**Verify:** 6th `/dream` call within 1 minute returns 429.

### Task 8.6 — `middleware.py`: Enable HSTS header
**File:** `src/mnemocore/api/middleware.py` ~L76
**Problem:** HSTS header is commented out.
**Fix:** Make it configurable: `security.hsts_enabled` (default: `True` in production, `False` in dev). If enabled, set `Strict-Transport-Security: max-age=31536000; includeSubDomains`.
**Verify:** Start with `HSTS_ENABLED=true`. Response includes HSTS header.

### Task 8.7 — `agent_interface.py`: Fix sync/async mismatch
**File:** `src/mnemocore/agent_interface.py`
**Problem:** `observe()`, `get_working_context()`, `start_episode()` are sync. `recall()` is async. Callers must handle both paradigms.
**Fix:** Make ALL methods async. Add sync wrappers (`recall_sync()`) for convenience.
**Verify:** All agent_interface tests pass.

### Task 8.8 — `cli/main.py`: Extract CLI boilerplate into decorator
**File:** `src/mnemocore/cli/main.py` (855 lines)
**Problem:** Every command repeats: load config → create engine → initialize → try → finally close.
**Fix:** Create a `@with_engine` decorator/context manager that handles the lifecycle.
**Verify:** All CLI tests pass. Commands are shorter and cleaner.

---

## Agent 9 — Config & Dependency Cleanup

**Priority:** P1 — STRUCTURAL
**Scope:** Fix dependency issues, version conflicts, split requirements
**Files touched:** 6 files
**Estimated tasks:** 9

### Task 9.1 — Split `requirements.txt` into runtime and test
**File:** `requirements.txt`, `requirements-dev.txt`
**Problem:** Test deps (`pytest`, `hypothesis`, `pytest-asyncio`) and optional deps (`plotly`, `pandas`) are in the runtime requirements file.
**Fix:**
- `requirements.txt` — ONLY runtime deps
- `requirements-dev.txt` — test + dev deps (add anything moved from requirements.txt)
- `requirements-optional.txt` — visualization deps (`plotly`, `pandas`, `tabulate`)
**Verify:** `pip install -r requirements.txt` in clean venv. No test packages installed. App starts.

### Task 9.2 — Sync `pyproject.toml` with `requirements.txt`
**File:** `pyproject.toml`
**Problem:** `aiohttp`, `psutil`, `click`, `tabulate` present in one but not the other.
**Fix:** Make `pyproject.toml` the canonical source. `requirements.txt` auto-generated via `pip-compile` or `uv pip compile`.
**Verify:** `pip install -e .` and `pip install -r requirements.txt` install the same packages.

### Task 9.3 — Create lockfile for reproducible builds
**File:** `requirements.lock` (new file)
**Problem:** All deps use `>=` only. Builds are non-reproducible.
**Fix:** `pip-compile --output-file=requirements.lock requirements.txt`. Add instructions in CONTRIBUTING.md.
**Verify:** Two consecutive `pip install -r requirements.lock` install identical versions.

### Task 9.4 — Fix `pytest.ini` / `pyproject.toml` conflict
**File:** `pytest.ini`, `pyproject.toml`
**Problem:** `pytest.ini` sets `testpaths = tests` (singular). `pyproject.toml` sets `testpaths = ["tests", "benchmarks"]`. `pytest.ini` takes precedence — benchmark tests silently excluded.
**Fix:** Remove `pytest.ini` entirely. Use `pyproject.toml` `[tool.pytest.ini_options]` as single source of truth.
**Verify:** `pytest --collect-only | grep -c test_` matches expected count including benchmarks.

### Task 9.5 — Fix `setup.cfg` mypy config
**File:** `setup.cfg`
**Problem:** `ignore_missing_imports = true` globally masks real import failures.
**Fix:** Change to `ignore_missing_imports = false`. Add `[mypy-<pkg>]` overrides for specific third-party packages that lack stubs.
**Verify:** `mypy src/mnemocore` runs without false positives from stubs.

### Task 9.6 — Create `.pre-commit-config.yaml`
**File:** `.pre-commit-config.yaml` (new file)
**Problem:** `pre-commit` is in dev dependencies but no config file exists.
**Fix:** Create config with hooks for:
- `ruff` (linting + formatting)
- `mypy` (type checking)
- `bandit` (security)
- `check-yaml`, `check-json`, `check-toml`
- `trailing-whitespace`, `end-of-file-fixer`
**Verify:** `pre-commit run --all-files` completes (may show fixable issues).

### Task 9.7 — Sync version numbers across all config files
**Problem:** Version numbers disagree:
- `pyproject.toml`: `5.0.0`
- `Dockerfile` LABEL: `4.5.0`
- `engine_lifecycle.py` get_stats: `4.5.0`
- Helm `Chart.yaml` appVersion: `3.5.0`
- `api/main.py`: `3.5.2` / `3.5.1`
- `RELEASE_CHECKLIST.md`: `v0.5.0-beta`
**Fix:** Set ALL to `5.1.0` (or whatever the next target is). Use a single source of truth:
- Read version from `importlib.metadata.version("mnemocore")` at runtime
- Helm: set `appVersion` from pyproject.toml during CI
**Verify:** `grep -rn "version" Dockerfile helm/ src/mnemocore/api/main.py src/mnemocore/core/engine_lifecycle.py` — all match.

### Task 9.8 — Add `aiosqlite` to dependencies
**File:** `pyproject.toml`, `requirements.txt`
**Problem:** Agents 2, 6 are replacing `sqlite3` with `aiosqlite` — must be in dependencies.
**Fix:** Add `"aiosqlite>=0.19.0"` to both files.
**Verify:** `pip install -e .` → `python -c "import aiosqlite"` succeeds.

### Task 9.9 — Add `croniter` to dependencies
**File:** `pyproject.toml`, `requirements.txt`
**Problem:** Agent 7 is replacing the custom cron parser with `croniter`.
**Fix:** Add `"croniter>=1.3.0"` to both files.
**Verify:** `python -c "import croniter"` succeeds.

---

## Agent 10 — Docker, Helm & CI/CD

**Priority:** P1 — STRUCTURAL
**Scope:** Fix Docker image, Helm chart, CI pipeline
**Files touched:** 8+ files
**Estimated tasks:** 11

### Task 10.1 — `Dockerfile`: Pin base image version
**File:** `Dockerfile`
**Problem:** `python:3.11-slim` without patch version pin.
**Fix:** Pin to `python:3.11.8-slim-bookworm` (or latest 3.11 patch).
**Verify:** `docker build` uses exact version.

### Task 10.2 — `Dockerfile`: Remove test deps from production image
**File:** `Dockerfile`
**Problem:** `requirements.txt` includes test packages (see Agent 9 Task 9.1).
**Fix:** After Agent 9 splits requirements, change Dockerfile to `COPY requirements.txt ./` (runtime only). Add multi-stage test image for CI.
**Verify:** `docker run mnemocore:latest pip list` does NOT contain pytest, hypothesis.

### Task 10.3 — `Dockerfile`: Update version label
**File:** `Dockerfile`
**Problem:** `LABEL version="4.5.0"` — wrong.
**Fix:** Use build arg: `ARG VERSION=5.1.0` and `LABEL version="${VERSION}"`. Set in CI.
**Verify:** `docker inspect mnemocore:latest | jq '.[0].Config.Labels.version'` returns `5.1.0`.

### Task 10.4 — `Dockerfile`: Add entrypoint validation
**File:** `Dockerfile`
**Problem:** No validation that required env vars are set before starting.
**Fix:** Create `scripts/entrypoint.sh`:
```bash
#!/bin/bash
if [ -z "$HAIM_API_KEY" ]; then
  echo "ERROR: HAIM_API_KEY must be set" >&2
  exit 1
fi
exec "$@"
```
**Verify:** Start without `HAIM_API_KEY` → exits with error. Start with key → runs normally.

### Task 10.5 — `docker-compose.yml`: Pin Qdrant image version
**File:** `docker-compose.yml`
**Problem:** `qdrant/qdrant:latest` — non-reproducible.
**Fix:** Pin to `qdrant/qdrant:v1.12.1` (or latest stable).
**Verify:** `docker compose up` uses pinned version.

### Task 10.6 — `docker-compose.yml`: Restrict port bindings
**File:** `docker-compose.yml`
**Problem:** Redis (6379) and Qdrant (6333, 6334) exposed to all interfaces.
**Fix:** Bind to localhost only: `127.0.0.1:6379:6379`. Or remove port bindings entirely (internal network only).
**Verify:** `curl localhost:6379` from another machine fails.

### Task 10.7 — Helm: Update versions and pin images
**File:** `helm/mnemocore/Chart.yaml`, `helm/mnemocore/values.yaml`
**Problem:** `appVersion: "3.5.0"`, `:latest` tags, wildcard dependency versions.
**Fix:** Set `appVersion: "5.1.0"`. Pin all image tags. Pin dependency chart versions.
**Verify:** `helm template mnemocore helm/mnemocore/` — all image tags are specific versions.

### Task 10.8 — Helm: Enable network policies by default
**File:** `helm/mnemocore/values.yaml`
**Problem:** `networkPolicy.enabled: false` — pods accept traffic from any source.
**Fix:** Default to `true`. Define ingress rules: allow mnemocore → Redis, mnemocore → Qdrant, ingress → mnemocore.
**Verify:** `helm template` generates NetworkPolicy resources.

### Task 10.9 — CI: Make security scans blocking
**File:** `.github/workflows/ci.yml`
**Problem:** `pip-audit` and `bandit` run with `continue-on-error: true`.
**Fix:** Remove `continue-on-error: true` from security jobs.
**Verify:** Introduce a known-vulnerable dep. CI fails.

### Task 10.10 — CI: Add container image scanning
**File:** `.github/workflows/docker-publish.yml`
**Problem:** No vulnerability scanning of built container images.
**Fix:** Add `aquasecurity/trivy-action@master` step after build, before push.
**Verify:** Build image with known CVE in base. CI fails.

### Task 10.11 — CI: Fix attestation step
**File:** `.github/workflows/docker-publish.yml`
**Problem:** `steps.push.outputs.digest` references a non-existent step ID.
**Fix:** Add `id: push` to the "Build and push Docker image" step.
**Verify:** Attestation step succeeds in CI.

---

## Agent 11 — Test Suite: Core Engine

**Priority:** P2 — TESTING
**Scope:** Write missing tests for core engine modules
**Files to create:** 8 new test files
**Estimated tasks:** 8

### Task 11.1 — `test_subconscious_ai.py` (NEW — CRITICAL PRIORITY)
**For:** `src/mnemocore/core/subconscious_ai.py`
**Tests needed:**
- `OllamaClient.generate()` with mocked `aiohttp.ClientSession` — happy path, timeout, connection error, HTTP 500
- `LMStudioClient.generate()` — same patterns
- `APIClient.generate()` — OpenAI format, Anthropic format, missing API key
- `ResourceGuard.can_pulse()` — CPU below/above threshold, rate limit not reached/exceeded
- `SubconsciousAIWorker` lifecycle — `start()`, `stop()`, pulse fires at interval
- `_init_model_client()` factory — each provider string returns correct client class
- dry_run mode — suggestions generated but NOT applied

### Task 11.2 — `test_gap_filler.py` (NEW)
**For:** `src/mnemocore/core/gap_filler.py`
**Tests needed:**
- `fill_now()` with mocked GapDetector + LLM
- Rate limiting enforcement
- Poll loop start/stop lifecycle
- Error handling: LLM returns garbage

### Task 11.3 — `test_holographic.py` (NEW)
**For:** `src/mnemocore/core/holographic.py`
**Tests needed:**
- `store_concept()` and `recall_concept()`
- Codebook persistence (save/load roundtrip)
- Concurrent writes don't corrupt file
- Load with mismatched dimension (graceful handling)

### Task 11.4 — `test_semantic_consolidation.py` (NEW)
**For:** `src/mnemocore/core/semantic_consolidation.py`
**Tests needed:**
- `run_once()` with mocked node store
- Hamming distance computation correctness
- Cluster merging logic
- Node existence check

### Task 11.5 — `test_cross_domain.py` (NEW)
**For:** `src/mnemocore/core/cross_domain.py`
**Tests needed:**
- Domain inference from content keywords
- Cross-domain synapse creation when overlap detected
- Buffer management and trim logic

### Task 11.6 — `test_hybrid_search_core.py` (NEW)
**For:** `src/mnemocore/core/hybrid_search.py`
**Tests needed:**
- `SparseEncoder.index_documents()` and `search()`
- Combined vector+keyword search result merging
- `expand_query()` (even if stub — test it returns unchanged)

### Task 11.7 — `test_topic_tracker.py` (NEW)
**For:** `src/mnemocore/core/topic_tracker.py`
**Tests needed:**
- Topic tracking across sequential queries
- Topic switching detection
- Context window behavior

### Task 11.8 — `test_agent_profile.py` (NEW)
**For:** `src/mnemocore/core/agent_profile.py`
**Tests needed:**
- CRUD operations on agent profiles
- Defaults for missing fields
- In-memory persistence behavior

---

## Agent 12 — Test Suite: Storage & Import/Export

**Priority:** P2 — TESTING
**Scope:** Write missing tests for storage layer
**Files to create:** 4 new test files
**Estimated tasks:** 4

### Task 12.1 — `test_backup_manager.py` (NEW — CRITICAL PRIORITY)
**For:** `src/mnemocore/storage/backup_manager.py`
**Tests needed:**
- `WriteAheadLog`: insert/update/delete/clear entries
- WAL rotation and replay
- `BackupManager.create_snapshot()` — full and incremental
- `BackupManager.restore_snapshot()` — happy path and corrupt file
- `list_snapshots()`, `get_snapshot_info()`, `delete_snapshot()`
- Concurrent WAL writes (after Agent 6 async conversion)

### Task 12.2 — `test_memory_exporter.py` (NEW)
**For:** `src/mnemocore/storage/memory_exporter.py`
**Tests needed:**
- Export to JSON, JSONL, Parquet (if deps available)
- Large export (1000+ memories) — stream vs batch
- Empty collection export
- Vector compression in export output
- Round-trip export → import integrity check

### Task 12.3 — `test_memory_importer.py` (NEW)
**For:** `src/mnemocore/storage/memory_importer.py`
**Tests needed:**
- Import from JSON, JSONL
- Deduplication strategies: skip, overwrite, merge
- Validation levels: lenient, strict
- Malformed input handling (corrupt JSON, invalid vectors)
- Large file handling
- Import log persistence

### Task 12.4 — `test_hybrid_search_storage.py` (NEW)
**For:** `src/mnemocore/storage/hybrid_search.py`
**Tests needed:**
- `search_qdrant_points()` with mocked Qdrant client
- Score extraction from various point types

---

## Agent 13 — Test Suite: AI & Subconscious

**Priority:** P2 — TESTING
**Scope:** Write missing tests for AI integration and dream subsystem
**Files to create:** 4 new test files
**Estimated tasks:** 4

### Task 13.1 — `test_llm_integration.py` (NEW — HIGH PRIORITY)
**For:** `src/mnemocore/llm_integration.py` (or `src/mnemocore/llm/` after Agent 7)
**Tests needed:**
- `OllamaClient.generate()` with mocked urllib — happy path, timeout, URLError
- `HAIMLLMIntegrator._call_llm()` — each provider (openai, anthropic, ollama, gemini mock)
- `MultiAgentHAIM` — agent registration, collaborative query with mocked LLM
- `ContextAwareLLMIntegrator` — context injection into prompts
- `RLMIntegrator.rlm_query()` — async with mocked engine
- Error handling: missing API key, unreachable server, malformed response

### Task 13.2 — `test_dream_pipeline.py` (NEW)
**For:** `src/mnemocore/subconscious/dream_pipeline.py` (or `dream/` package after Agent 7)
**Tests needed:**
- `EpisodicClusterer.cluster()` with mocked engine
- `PatternExtractor` — keyword, temporal, metadata patterns
- `DreamSynthesizer` — dream synthesis from patterns
- `ContradictionResolver` — contradiction detection and resolution
- `SemanticPromoter` — promotion scoring
- Full `DreamPipeline.run()` with mocked components
- Pipeline with disabled stages

### Task 13.3 — `test_dream_scheduler.py` (NEW)
**For:** `src/mnemocore/subconscious/dream_scheduler.py`
**Tests needed:**
- `IdleDetector` — idle state transitions, CPU threshold
- `DreamSession.execute()` — normal completion, cancellation, timeout
- `DreamScheduler` — schedule-based triggering, cooldown enforcement
- Next-run calculation (after cron parser fix by Agent 7)

### Task 13.4 — `test_forgetting_analytics.py` (NEW)
**For:** `src/mnemocore/cognitive/forgetting_analytics.py`
**Tests needed:**
- Dashboard data generation
- Chart data for learning progress, SM-2 performance
- CSV/JSON export
- Empty data handling

---

## Agent 14 — Test Suite: Meta, Cognitive, Events

**Priority:** P2 — TESTING
**Scope:** Write missing tests for meta-cognition, cognitive services, and event schemas
**Files to create:** 6 new test files
**Estimated tasks:** 6

### Task 14.1 — `test_goal_tree.py` (NEW)
**For:** `src/mnemocore/meta/goal_tree.py`
**Tests needed:**
- CRUD: add, complete, block, decompose
- ID uniqueness (after Agent 1 fix)
- Persistence: save to file, load from file
- Corrupt file handling (after Agent 5 fix)
- `stats()` returns correct counts
- Decompose cascading

### Task 14.2 — `test_learning_journal.py` (NEW)
**For:** `src/mnemocore/meta/learning_journal.py`
**Tests needed:**
- Entry creation and retrieval
- Prediction registration and evaluation
- Surprise calculation
- Tag-based querying
- Persistence roundtrip
- Empty journal handling

### Task 14.3 — `test_future_thinking.py` (NEW)
**For:** `src/mnemocore/cognitive/future_thinking.py`
**Tests needed:**
- `ScenarioStore` lifecycle: create, get, verify, archive
- Confidence decay over time
- Cleanup of expired scenarios
- Stats (after deadlock fix by Agent 2)

### Task 14.4 — `test_context_optimizer_full.py` (NEW or extend existing)
**For:** `src/mnemocore/cognitive/context_optimizer.py`
**Tests needed:**
- Token counting accuracy (tiktoken vs heuristic fallback)
- Context window optimization for different model limits
- Paragraph splitting with overlap
- Diversity filtering

### Task 14.5 — `test_event_schemas.py` (NEW)
**For:** `src/mnemocore/events/schemas.py`
**Tests needed:**
- Schema validation for each defined schema
- Type checking for all supported types
- Invalid schema rejection
- Edge cases: empty objects, nested objects

### Task 14.6 — `test_mcp_schemas.py` (NEW)
**For:** `src/mnemocore/mcp/schemas.py`
**Tests needed:**
- Pydantic model validation for each input schema
- Field constraints (min/max values, string patterns)
- Required vs optional fields

---

## Agent 15 — Test Suite: API, MCP, CLI

**Priority:** P2 — TESTING
**Scope:** Expand and fix existing API/MCP/CLI tests; add missing coverage
**Files to modify:** 5 existing test files, 3 new
**Estimated tasks:** 7

### Task 15.1 — `test_cli_formatters.py` (NEW)
**For:** `src/mnemocore/cli/formatters.py`
**Tests needed:**
- Table formatting with various column widths
- Truncation of long content
- Timestamp formatting (valid, invalid, None)
- ANSI color handling (strip on non-TTY)

### Task 15.2 — `test_json_compat.py` (NEW)
**For:** `src/mnemocore/utils/json_compat.py`
**Tests needed:**
- `FallbackEncoder` with datetime, numpy arrays, custom objects
- orjson path vs stdlib fallback
- `dumps()` with various types
- Edge case: NaN, Inf, None

### Task 15.3 — `test_process.py` (NEW)
**For:** `src/mnemocore/utils/process.py`
**Tests needed:**
- `lower_process_priority()` — verify nice value change (Unix) or Windows priority class
- Error handling when psutil unavailable

### Task 15.4 — Expand `test_api_functional.py`
**For:** Existing test file
**Tests to add:**
- `/dream` endpoint with rate limiting
- `/export` with upper bound limit parameter
- Concurrent `/store` requests
- Invalid content types

### Task 15.5 — Expand `test_mcp_adapter.py`
**For:** Existing test file
**Tests to add:**
- URL encoding of query parameters (after Agent 3 fix)
- HTTPS enforcement
- Retry on transient failure
- Connection timeout handling

### Task 15.6 — Fix `test_consolidation_worker.py` stale patches
**For:** Existing test file
**Problem:** Patches reference `src.core.consolidation_worker.*` instead of `mnemocore.core.consolidation_worker.*`.
**Fix:** Update all patch paths.
**Verify:** Tests actually mock the correct objects and pass.

### Task 15.7 — Fix `test_concurrency.py`
**For:** Existing test file
**Problem:** Not a real test — it's a standalone `asyncio.run(main())` script with no assertions.
**Fix:** Convert to pytest-asyncio tests with proper assertions. Move to Agent 16.
**Verify:** `pytest tests/test_concurrency.py` discovers and runs tests.

---

## Agent 16 — Test Suite: Concurrency & Integration

**Priority:** P2 — TESTING
**Scope:** Create proper concurrency tests and integration tests
**Files to create:** 3 new test files
**Estimated tasks:** 5

### Task 16.1 — `test_concurrency_store_query.py` (NEW)
**Tests needed:**
- 100 concurrent `store()` calls → all succeed, no duplicates
- 50 concurrent `store()` + 50 concurrent `query()` → no crashes, valid results
- Concurrent tier promotion/demotion → no data loss
- Concurrent working memory push/pop → no corruption

### Task 16.2 — `test_concurrency_events.py` (NEW)
**Tests needed:**
- 100 concurrent `emit()` calls → all events delivered
- Subscriber add/remove during emission → no errors
- Event bus singleton under concurrent access

### Task 16.3 — `test_concurrency_file_safety.py` (NEW)
**Tests needed:**
- Concurrent `holographic.save()` → file not corrupted
- Concurrent `strategy_bank._persist_to_disk()` → file valid
- Concurrent `procedural_store._persist_to_disk()` → file valid

### Task 16.4 — `test_integration_store_query_cycle.py` (NEW)
**Tests needed:**
- Full lifecycle: store → query → update → query → delete
- Store with associations → query with association spreading → verify links
- Store → dream → verify dream results in store
- Store → export → import → verify round-trip

### Task 16.5 — Create `conftest.py` integration fixtures
**For:** `tests/conftest.py`
**Tests to add:**
- Mark integration tests with `@pytest.mark.integration`
- Add Docker-based Redis/Qdrant fixtures (or skip if unavailable)
- Session-scoped engine fixture that doesn't require mocking

---

## Agent 17 — Performance Optimization

**Priority:** P3 — POLISH
**Scope:** Fix O(N²) algorithms, vectorize numpy loops, eliminate blocking I/O
**Files touched:** 8 files
**Estimated tasks:** 8

### Task 17.1 — `consolidation.py`: Fix O(N²) clustering
**File:** `src/mnemocore/core/consolidation.py`
**Problem:** `find_clusters()` does pairwise similarity — O(N²).
**Fix:** Use FAISS or `sklearn.cluster.AgglomerativeClustering` for approximate nearest neighbors. Or batch the pairwise comparison using numpy matrix operations.
**Verify:** Benchmark: 500 nodes. Current: >2s. Target: <200ms.

### Task 17.2 — `semantic_consolidation.py`: Vectorize Hamming distance
**File:** `src/mnemocore/core/semantic_consolidation.py`
**Problem:** Python `for` loop over packed uint8 arrays for N×N distance matrix.
**Fix:** Use numpy broadcasting: `np.unpackbits(a)` XOR, or `scipy.spatial.distance.cdist` with Hamming metric.
**Verify:** Benchmark: 500 vectors. Current: >5s. Target: <100ms.

### Task 17.3 — `binary_vector_compression.py`: Vectorize encode/decode
**File:** `src/mnemocore/storage/binary_vector_compression.py` ~L334, L349
**Problem:** Python `for` loop over `dimension // 4` iterations in encode/decode.
**Fix:** Use `np.packbits()` / `np.unpackbits()` for encoding. Vectorized lookup for decode.
**Verify:** Benchmark: 16384-dim vector. Current: >10ms. Target: <1ms.

### Task 17.4 — `vector_compression.py`: Vectorize binary quantizer
**File:** `src/mnemocore/storage/vector_compression.py` ~L560, L575, L593
**Problem:** Python `for` loop in `compress()`, `decompress()`, `hamming_distance()`.
**Fix:** Use `np.packbits()`, `np.unpackbits()`, vectorized popcount.
**Verify:** Same benchmark targets as Task 17.3.

### Task 17.5 — `event_bus.py`: Replace `list.pop(0)` with deque
**File:** `src/mnemocore/events/event_bus.py` ~L640
**Problem:** `_history.pop(0)` is O(n).
**Fix:** Use `collections.deque(maxlen=max_history_size)`.
**Verify:** No behavior change. Benchmark: 10K events in history. No visible delay on append.

### Task 17.6 — `immunology.py`: Reduce `sweep()` memory pressure
**File:** `src/mnemocore/core/immunology.py`
**Problem:** `np.stack()` on all HOT tier vectors every sweep interval.
**Fix:** Process vectors in batches (e.g., 100 at a time). Or subsample when HOT tier is large.
**Verify:** Benchmark: 5000 HOT tier nodes. Peak memory < 500MB.

### Task 17.7 — `llm_integration.py`: Fix O(N) linear scan
**File:** `src/mnemocore/llm_integration.py` ~L710
**Problem:** `_concept_to_memory_id` does O(N) similarity comparisons across entire HOT tier.
**Fix:** Use the engine's existing `query()` (vector search) instead of manual scan.
**Verify:** Benchmark: 5000 HOT tier nodes. Current: >1s. Target: <50ms.

### Task 17.8 — `tier_scoring.py`: Fix stale recency timestamp
**File:** `src/mnemocore/core/tier_scoring.py` ~L200
**Problem:** `RecencyScorer.__init__()` captures `self.now = datetime.now()` at construction time. Becomes stale immediately.
**Fix:** Use `datetime.now()` at scoring time — `score()` should compute recency dynamically.
**Verify:** Score same memory at T=0 and T=1h. Scores differ correctly.

---

## Agent 18 — Documentation Overhaul

**Priority:** P3 — POLISH
**Scope:** Add missing docstrings, update architecture docs, create contributor guide
**Files touched:** 20+ files
**Estimated tasks:** 10

### Task 18.1 — Add module docstrings to all `__init__.py` files
**Files:** All `__init__.py` across `core/`, `storage/`, `cognitive/`, `events/`, `subconscious/`, `meta/`, `mcp/`, `api/`, `cli/`, `utils/`
**Fix:** Each should have a brief module docstring explaining the package purpose and main exports.

### Task 18.2 — Add missing function/class docstrings in core
**Files:** Files identified in audit with missing docstrings
**Convention:** Google-style docstrings with Args, Returns, Raises sections.

### Task 18.3 — Update `docs/ARCHITECTURE.md`
**Problem:** Architecture doc may be outdated after Phase 5/6 additions.
**Fix:** Update with current module structure, data flow diagrams, tier system, dream pipeline, MCP integration.

### Task 18.4 — Create `CONTRIBUTING.md`
**Content:**
- Development setup (venv, deps, pre-commit)
- Code style (ruff config, naming conventions)
- Test requirements (coverage threshold, fixture patterns)
- PR process
- Commit message format

### Task 18.5 — Update `RELEASE_CHECKLIST.md`
**Problem:** Severely outdated (references v0.5.0-beta).
**Fix:** Rewrite for v5.1.0+ with all current steps: version bumps, Helm chart, Docker tag, CHANGELOG, CI validation, image scanning.

### Task 18.6 — Update `SECURITY.md`
**Fix:** Add supported versions table, response SLA, CVE tracking process, encrypted communication option.

### Task 18.7 — Create `docs/DEPLOYMENT.md`
**Content:** Production deployment guide: Docker, Kubernetes/Helm, bare-metal. Including:
- Required env vars
- Redis/Qdrant auth setup
- Monitoring/alerting setup
- Backup/restore procedures
- Scaling guidance

### Task 18.8 — Create `docs/CONFIGURATION.md`
**Content:** Complete config reference: every YAML key, every env var, defaults, validation rules, examples.

### Task 18.9 — Update `README.md`
**Fix:** Ensure it references correct version, links to new docs, has accurate quick-start instructions.

### Task 18.10 — Add inline code comments for complex algorithms
**Files:** `consolidation.py`, `bayesian_ltp.py`, `forgetting_curve.py`, `immunology.py`, `binary_hdv.py`
**Fix:** Add comments explaining the math/algorithm at each non-obvious step.

---

## Agent 19 — Metrics & Observability

**Priority:** P3 — POLISH
**Scope:** Fix duplicate metrics, add missing metrics, update Grafana dashboard
**Files touched:** 4 files
**Estimated tasks:** 6

### Task 19.1 — `metrics.py`: Deduplicate metric names
**File:** `src/mnemocore/core/metrics.py`
**Problem:** `ENGINE_MEMORY_COUNT` vs `MEMORY_COUNT_TOTAL`, `ENGINE_STORE_LATENCY` vs `STORE_DURATION_SECONDS`, etc.
**Fix:** Standardize on `mnemocore_*` prefix. Remove old `haim_*` metrics. Update all call sites.
**Verify:** `grep -rn "haim_" src/` returns zero results.

### Task 19.2 — `metrics.py`: Remove duplicate timer/latency decorators
**File:** `src/mnemocore/core/metrics.py`
**Problem:** `timer`, `track_latency`, `track_async_latency` — three ways to do the same thing.
**Fix:** Keep `timer` (most full-featured). Remove or deprecate the others. Update all call sites.
**Verify:** No references to removed decorators.

### Task 19.3 — `metrics.py`: Add missing metrics
**What's missing:**
- Dream pipeline latency/error rate
- Subconscious AI pulse latency/skip rate
- Webhook delivery success/failure rate
- Backup/restore operation latency
- Working memory size
**Verify:** Each new metric appears in Prometheus exposition format.

### Task 19.4 — `grafana-dashboard.json`: Add alerting rules
**File:** `grafana-dashboard.json`
**Problem:** No alert rules defined.
**Fix:** Add alerts for:
- Error rate > 1% (5 min window)
- P95 latency > 500ms
- Memory count sudden drop > 10%
- Queue length > 1000
**Verify:** Dashboard JSON is valid. Alerts visible in Grafana preview.

### Task 19.5 — `grafana-dashboard.json`: Update metric names
**Problem:** Mixed `haim_*` and `mnemocore_*` prefixes.
**Fix:** After Task 19.1, update all panel queries to use `mnemocore_*`.
**Verify:** All panels render without "no data".

### Task 19.6 — `grafana-dashboard.json`: Add template variables
**Problem:** No variables for filtering by namespace/pod/instance.
**Fix:** Add `namespace`, `pod`, `instance` template variables with label-based queries.
**Verify:** Dashboard works in multi-replica setup.

---

## Agent 20 — Final Validation & Release Prep

**Priority:** P4 — FINAL
**Scope:** Run full test suite, validate all fixes, prepare release
**Files touched:** 3 files
**Estimated tasks:** 8

### Task 20.1 — Run full test suite
```bash
pytest tests/ benchmarks/ -v --tb=short --cov=src/mnemocore --cov-report=html --cov-fail-under=80
```
**Target:** All tests pass. Coverage ≥80%. Zero failures.

### Task 20.2 — Run static analysis
```bash
ruff check src/mnemocore/
mypy src/mnemocore/ --strict
bandit -r src/mnemocore/ -ll
```
**Target:** No errors, no high-severity security findings.

### Task 20.3 — Run security audit
```bash
pip-audit -r requirements.txt
bandit -r src/mnemocore/ -ll
```
**Target:** No known vulnerable dependencies. No code-level security issues.

### Task 20.4 — Docker build and healthcheck
```bash
docker compose build
docker compose up -d
# Wait 30 seconds
docker compose ps  # All services healthy
curl http://localhost:8100/health  # Returns 200
docker compose down
```

### Task 20.5 — Helm chart validation
```bash
helm lint helm/mnemocore/
helm template mnemocore helm/mnemocore/ --debug
```

### Task 20.6 — Update `CHANGELOG.md`
Add v5.1.0 section with all fixes categorized under:
- **Fixed:** (all bug fixes)
- **Security:** (all security hardening)
- **Changed:** (refactoring, breaking changes)
- **Added:** (new tests, docs)

### Task 20.7 — Tag release
```bash
git tag -a v5.1.0 -m "v5.1.0: Production-ready release"
git push origin v5.1.0
```

### Task 20.8 — Verify CI pipeline passes for the tag
- Docker image published
- All tests green
- Security scans blocking and passing
- Container image scan passing

---

## Dependency Graph Between Agents

```
Agent 1 (Critical Bugs) ──────────────┐
Agent 2 (Thread Safety) ──────────────┼── PHASE 1 (must complete first)
Agent 3 (Security) ────────────────────┘
         │
         ▼
Agent 4 (Engine Refactor) ─────────────┐
Agent 5 (Error Handling) ──────────────┤
Agent 6 (Storage Layer) ──────────────┤
Agent 7 (Subconscious Refactor) ──────┼── PHASE 2 (parallel after Phase 1)
Agent 8 (API Hardening) ──────────────┤
Agent 9 (Config & Deps) ──────────────┤   Agent 9 must finish before
Agent 10 (Docker/Helm/CI) ────────────┘   Agent 10 can build images
         │
         ▼
Agent 11 (Test: Core Engine) ──────────┐
Agent 12 (Test: Storage) ─────────────┤
Agent 13 (Test: AI/Subconscious) ─────┼── PHASE 3 (parallel after Phase 2)
Agent 14 (Test: Meta/Cognitive) ──────┤
Agent 15 (Test: API/MCP/CLI) ─────────┤
Agent 16 (Test: Concurrency) ─────────┘
         │
         ▼
Agent 17 (Performance) ───────────────┐
Agent 18 (Documentation) ─────────────┼── PHASE 4 (parallel after Phase 3)
Agent 19 (Metrics) ────────────────────┘
         │
         ▼
Agent 20 (Final Validation) ──────────── PHASE 5 (solo, after everything)
```

### Cross-Agent Dependencies (within same phase)

| Agent | Depends On | Reason |
|-------|-----------|--------|
| Agent 10 | Agent 9 | Needs split requirements.txt before building Docker image |
| Agent 11 | Agents 1, 2 | Tests should verify bug fixes and thread safety fixes |
| Agent 12 | Agent 6 | Tests should verify storage layer fixes (aiosqlite) |
| Agent 13 | Agents 7, 5 | Tests should verify refactored dream pipeline and error handling |
| Agent 14 | Agents 1, 5 | Tests should verify goal_tree ID fix and JSON error handling |
| Agent 15 | Agent 8 | Tests should verify API route decomposition |
| Agent 16 | Agent 2 | Concurrency tests must run after lock fixes |
| Agent 17 | Agents 4, 7 | Performance work should happen on refactored code |
| Agent 19 | Agent 17 | Metrics for new code paths added by performance agent |

---

## 5. Complete Issue Registry

### All CRITICAL Issues (11)

| ID | File | Issue | Agent |
|----|------|-------|-------|
| C-01 | `exceptions.py` | `class ImportError` shadows builtin | Agent 1 |
| C-02 | `engine_coordinator.py` | `export_memories` broken: `asyncio.run()` in async + nonexistent `functools._run_in_thread` | Agent 1 |
| C-03 | `engine.py` | `setattr` on frozen dataclass in `enable_reconstructive_memory` | Agent 1 |
| C-04 | `logging_config.py` | `diagnose=True` leaks secrets into logs | Agent 1 |
| C-05 | `tier_eviction.py` | `self.warm_strategy` typo (should be `warm_policy`) → AttributeError | Agent 1 |
| C-06 | `goal_tree.py` | ID collision on delete+create (`goal_{len}`) | Agent 1 |
| C-07 | `learning_journal.py` | Same ID collision bug | Agent 1 |
| C-08 | `docker-compose.yml` | Redis/Qdrant exposed without authentication | Agent 1 |
| C-09 | `Dockerfile` | Healthcheck path wrong (`/scripts/healthcheck.py` → `/scripts/ops/healthcheck.py`) | Agent 1 |
| C-10 | `pyproject.toml` | `aiohttp` missing from package dependencies | Agent 1 |
| C-11 | `dream_scheduler.py` | `self.cfg.idle_threshold` AttributeError (should be `idle_threshold_seconds`) | Agent 1 |

### All HIGH Issues (49)

| ID | File | Issue | Agent |
|----|------|-------|-------|
| H-01 | `reliability.py` | Non-thread-safe circuit breaker | Agent 2 |
| H-02 | `tier_manager.py` | TOCTOU race conditions in promote/demote | Agent 2 |
| H-03 | `tier_storage.py` | Double-lock deadlock risk + mutable reference leak | Agent 2 |
| H-04 | `hnsw_index.py` | Stale count data race + bare except on save/load | Agent 2 |
| H-05 | `synapse_index.py` | No internal thread safety, doc says external lock needed | Agent 2 |
| H-06 | `holographic.py` | Sync I/O on every write + no file locking | Agent 2 |
| H-07 | `event_bus.py` | Global singleton not thread-safe | Agent 2 |
| H-08 | `vector_compression.py` | pickle.dumps/loads RCE (×2 locations) | Agent 3 |
| H-09 | `binary_vector_compression.py` | pickle.dumps/loads RCE | Agent 3 |
| H-10 | `config.py` | CORS defaults to `["*"]` | Agent 3 |
| H-11 | `config.py` | API keys storable in YAML config files | Agent 3 |
| H-12 | `daemon.py` | `sys.path.insert` import hijacking risk | Agent 3 |
| H-13 | `daemon.py` | Hardcoded `/tmp/subconscious.log` | Agent 3 |
| H-14 | `api_adapter.py` | API key sent without HTTPS enforcement | Agent 3 |
| H-15 | `api_adapter.py` | URL parameter injection (unencoded query) | Agent 3 |
| H-16 | `middleware.py` | X-Forwarded-For spoofable for rate-limit bypass | Agent 3 |
| H-17 | `mcp/server.py` | Error details leaked via `str(exc)` | Agent 3 |
| H-18 | `webhook_manager.py` | Secrets stored plaintext on disk | Agent 3 |
| H-19 | `backup_manager.py` | SQL injection in LIMIT clause | Agent 3 |
| H-20 | `api/main.py` | CORS `["*"]` fallback | Agent 3 |
| H-21 | `engine_core.py` | `store()` is 100-line god function | Agent 4 |
| H-22 | `engine_core.py` | `query()` is 170-line god function | Agent 4 |
| H-23 | `engine_core.py` | `ensure_future` fire-and-forget (exceptions lost) | Agent 4 |
| H-24 | `config.py` | `load_config()` is 330-line god function | Agent 4 |
| H-25 | `engine_lifecycle.py` | `_load()` inner swallows all exceptions | Agent 5 |
| H-26 | `engine_lifecycle.py` | `close()` no error isolation between workers | Agent 5 |
| H-27 | `engine_lifecycle.py` | Health check lies about component health | Agent 5 |
| H-28 | `engine_lifecycle.py` | No rollback of `_initialized` on failure | Agent 5 |
| H-29 | `daemon.py` | 3 bare `except:` blocks | Agent 5 |
| H-30 | `llm_integration.py` | `_call_llm` returns error strings as valid content | Agent 5 |
| H-31 | `goal_tree.py` | No JSON error handling in `_load()` | Agent 5 |
| H-32 | `learning_journal.py` | No JSON error handling in `_load()` | Agent 5 |
| H-33 | `backup_manager.py` | Blocking SQLite in async context | Agent 6 |
| H-34 | `binary_vector_compression.py` | Blocking SQLite in async context | Agent 6 |
| H-35 | `memory_exporter.py` | Accumulates all records in memory | Agent 6 |
| H-36 | `memory_importer.py` | Reads entire file into memory | Agent 6 |
| H-37 | `webhook_manager.py` | `aiohttp.ClientSession(total=...)` misuse | Agent 6 |
| H-38 | `dream_pipeline.py` | 1336-line god file (8 classes) | Agent 7 |
| H-39 | `dream_scheduler.py` | Buggy custom cron parser | Agent 7 |
| H-40 | `llm_integration.py` | 1102-line god file (6 classes) | Agent 7 |
| H-41 | `forgetting_curve.py` | 1501-line god file (6+ classes) | Agent 7 |
| H-42 | `api/main.py` | 1577-line god file — needs router decomposition | Agent 8 |
| H-43 | `api/main.py` | Prediction store global race condition | Agent 8 |
| H-44 | `requirements.txt` | Test deps in production requirements | Agent 9 |
| H-45 | `requirements.txt` | No lockfile for reproducible builds | Agent 9 |
| H-46 | `pytest.ini` | Conflicts with pyproject.toml (benchmark tests excluded) | Agent 9 |
| H-47 | `config.yaml` | `subconscious_ai.enabled: true` + `dry_run: false` as default | Agent 9 |
| H-48 | CI | Security scans non-blocking (`continue-on-error: true`) | Agent 10 |
| H-49 | CI | No container image scanning (Trivy/Grype) | Agent 10 |
| H-50 | `consolidation.py` | O(N²) pairwise clustering | Agent 17 |
| H-51 | `semantic_consolidation.py` | O(N²) Hamming matrix in Python loop | Agent 17 |

### Test Coverage — Source Files With ZERO Tests (24)

| # | Source File | Test Priority | Agent |
|----|-----------|--------------|-------|
| 1 | `core/subconscious_ai.py` | CRITICAL | Agent 11 |
| 2 | `storage/backup_manager.py` | CRITICAL | Agent 12 |
| 3 | `storage/memory_exporter.py` | HIGH | Agent 12 |
| 4 | `storage/memory_importer.py` | HIGH | Agent 12 |
| 5 | `core/gap_filler.py` | HIGH | Agent 11 |
| 6 | `subconscious/dream_pipeline.py` | HIGH | Agent 13 |
| 7 | `subconscious/dream_scheduler.py` | HIGH | Agent 13 |
| 8 | `llm_integration.py` | HIGH | Agent 13 |
| 9 | `meta/goal_tree.py` | MEDIUM | Agent 14 |
| 10 | `meta/learning_journal.py` | MEDIUM | Agent 14 |
| 11 | `cognitive/future_thinking.py` | MEDIUM | Agent 14 |
| 12 | `cognitive/forgetting_analytics.py` | MEDIUM | Agent 14 |
| 13 | `core/cross_domain.py` | MEDIUM | Agent 11 |
| 14 | `core/semantic_consolidation.py` | MEDIUM | Agent 11 |
| 15 | `core/holographic.py` | MEDIUM | Agent 11 |
| 16 | `core/hybrid_search.py` | MEDIUM | Agent 11 |
| 17 | `storage/hybrid_search.py` | MEDIUM | Agent 12 |
| 18 | `core/topic_tracker.py` | LOW | Agent 11 |
| 19 | `core/agent_profile.py` | LOW | Agent 11 |
| 20 | `core/logging_config.py` | LOW | Agent 11 |
| 21 | `cli/formatters.py` | LOW | Agent 15 |
| 22 | `utils/json_compat.py` | LOW | Agent 15 |
| 23 | `utils/process.py` | LOW | Agent 15 |
| 24 | `mcp/schemas.py` | LOW | Agent 14 |

---

## Quick Reference: Files Per Agent

| Agent | Files Modified | Files Created | Total Tasks |
|-------|---------------|---------------|-------------|
| Agent 1 | 10 | 0 | 11 |
| Agent 2 | 10 | 0 | 12 |
| Agent 3 | 11 | 0 | 13 |
| Agent 4 | 5 (+1 new utils) | 1 | 9 |
| Agent 5 | 12 | 0 | 10 |
| Agent 6 | 7 | 0 | 9 |
| Agent 7 | 5 (+8 new modules) | 8 | 8 |
| Agent 8 | 6 (+8 new routes) | 8 | 8 |
| Agent 9 | 6 (+2 new) | 2 | 9 |
| Agent 10 | 8 | 1 | 11 |
| Agent 11 | 0 | 8 | 8 |
| Agent 12 | 0 | 4 | 4 |
| Agent 13 | 0 | 4 | 4 |
| Agent 14 | 0 | 6 | 6 |
| Agent 15 | 3 | 3 | 7 |
| Agent 16 | 1 | 3 | 5 |
| Agent 17 | 8 | 0 | 8 |
| Agent 18 | 20+ | 3 | 10 |
| Agent 19 | 4 | 0 | 6 |
| Agent 20 | 3 | 0 | 8 |
| **Total** | **~100** | **~50** | **166** |

---

*Document generated by automated codebase audit. All findings have been verified by reading the source files directly.*
