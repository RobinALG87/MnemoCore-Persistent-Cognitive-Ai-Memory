# MnemoCore v5.0.0 → v5.1.0 Production Remediation Progress

> **Started:** 2026-02-28
> **Last Audit:** 2026-02-28 (automated codebase verification)
> **Total Tasks:** 166 across 20 agents
> **Method:** Parallel agent execution in 5 phases

> **Current source of truth:** [Platform baseline — 2026-07-12](docs/status/2026-07-12-platform-baseline.md)
>
> This file is a historical remediation inventory from 2026-02-28. Its counts,
> task totals, and test-failure summary are not current release evidence. Use
> the linked baseline for present PASS, FAIL, QUARANTINED, and NOT RUN status.

---

## Test Suite Summary (Last Run)

```
133 failed, 2239 passed, 24 skipped, 34 warnings, 12 errors in 442.55s
```

---

## Phase Progress Overview

| Phase | Agents | Tasks | Verified Done | Partial | Not Done | Status |
|-------|--------|-------|---------------|---------|----------|--------|
| Phase 1 | 1-3 | 36 | 28 | 3 | 5 | ⚠️ MOSTLY DONE |
| Phase 2 | 4-10 | 62 | — | — | 62 | ❌ NOT STARTED |
| Phase 3 | 11-16 | 34 | — | — | 34 | ❌ NOT STARTED |
| Phase 4 | 17-20 | 32 | — | — | 32 | ❌ NOT STARTED |

---

## Agent Status

### PHASE 1 - BLOCKING (Must complete first)

#### Agent 1 — CRITICAL Bug Fixes (11 tasks)
**Status:** ✅ DONE (11/11 verified)
**Started:** 2026-02-28

| Task | Description | Status | Evidence |
|------|-------------|--------|----------|
| 1.1 | Rename shadowed `ImportError` in exceptions.py | ✅ DONE | No `class ImportError` found in exceptions.py. All exception classes use domain-specific names (`MnemoCoreError`, `StorageError`, etc.) |
| 1.2 | Fix broken `export_memories` in engine_coordinator.py | ✅ DONE | `export_memories()` at L424 rewritten — uses `async with self.tier_manager.lock`, no `asyncio.run()` or `functools._run_in_thread` |
| 1.3 | Fix frozen dataclass mutation in engine.py | ✅ DONE | Uses `from dataclasses import replace` (L14) and `cfg = replace(cfg, **valid_updates)` (L339) instead of `setattr` on frozen dataclass |
| 1.4 | Fix attribute typo in tier_eviction.py | ✅ DONE | Uses `self.warm_policy` (L258) consistently, not `warm_strategy`. `get_warm_strategy()` method properly references `self.warm_policy` (L276) |
| 1.5 | Disable `diagnose=True` in logging_config.py | ✅ DONE | `diagnose=False` at L99 with comment "Disabled to prevent leaking sensitive data in tracebacks". Also `backtrace=False` (L98) |
| 1.6 | Fix ID collision in meta/goal_tree.py | ✅ DONE | Uses `uuid.uuid4().hex[:12]` (L86) for goal IDs instead of length-based incrementing |
| 1.7 | Fix ID collision in learning_journal.py | ✅ DONE | Uses `uuid.uuid4().hex[:12]` (L69) for entry IDs |
| 1.8 | Fix Redis/Qdrant exposed without auth | ✅ DONE | `docker-compose.yml`: Redis uses `--requirepass ${REDIS_PASSWORD:-changeme}` (L83), Qdrant uses `QDRANT_API_KEY=${QDRANT_API_KEY:-changeme}` (L27) |
| 1.9 | Fix healthcheck path in Dockerfile | ✅ DONE | `HEALTHCHECK` at L83 uses `CMD python /app/scripts/ops/healthcheck.py` (correct path) |
| 1.10 | Add missing `aiohttp` and `psutil` deps | ✅ DONE | `requirements.txt` contains `aiohttp>=3.9.0` (L36) and `psutil>=5.9.0` (L38) |
| 1.11 | Fix AttributeError in dream_scheduler.py | ✅ DONE | Uses `getattr()` with safe fallbacks (L864-870): `idle_threshold_seconds`, `min_idle_duration`, `max_cpu_percent` |

#### Agent 2 — Thread Safety & Concurrency (12 tasks)
**Status:** ✅ DONE (12/12 verified)
**Started:** 2026-02-28

| Task | Description | Status | Evidence |
|------|-------------|--------|----------|
| 2.1 | Make circuit breaker thread-safe | ✅ DONE | `NativeCircuitBreaker` class (L37) with `self._lock = threading.Lock()` (L51). All state transitions protected: `with self._lock:` at L61, L92, L99. Uses `time.monotonic()` |
| 2.2 | Fix TOCTOU race conditions in tier_manager.py | ✅ DONE | All tier transitions happen inside `async with self.lock:` blocks (L179, L223, L251, L311, L381). Demotion I/O inside lock (L231). `_promote_to_hot_locked()` (L545) for locked-context calls |
| 2.3 | Fix double-lock deadlock risk in tier_storage.py | ✅ DONE | `tier_storage.py` L77-78: "This class does NOT have its own lock. All locking is delegated to TierManager's lock to prevent double-lock deadlock." Returns copies at L141, L150 |
| 2.4 | Fix data races in hnsw_index.py | ✅ DONE | `_write_lock = Lock()` (L92) + `_rebuild_lock = threading.Lock()` (L100). Stale count read under lock (L302). Index swap under lock (L186). Data snapshot under lock (L168) |
| 2.5 | Add internal locking to synapse_index.py | ✅ DONE | `self._lock = asyncio.Lock()` (L81). All mutating operations use `async with self._lock:` (L96, L110, L131, L140, L154, L186, L196, L216, L239, L261) |
| 2.6 | Fix concurrent file corruption in holographic.py | ✅ DONE | `self._file_lock = asyncio.Lock()` (L46). Atomic writes via `_atomic_write_json()` (L207) using temp file + rename. Save under lock (L111) |
| 2.7 | Replace threading lock with async lock in working_memory.py | ✅ DONE | `self._lock = asyncio.Lock()` (L29). All operations use `async with self._lock:` (L32, L41, L60, L65, L73, L110) |
| 2.8 | Replace sync I/O lock in procedural_store.py | ✅ DONE | `self._lock = asyncio.Lock()` (L42). Async file I/O with `asyncio.get_running_loop()` (L295). All ops under `async with self._lock:` |
| 2.9 | Debounce disk persistence in strategy_bank.py | ✅ DONE | Uses `threading.RLock()` (L436). "Debounced persistence" comment at L450, L604. Dirty flag + periodic flush pattern |
| 2.10 | Fix global singleton thread safety in event_bus.py | ✅ DONE | `_EVENT_BUS_LOCK = threading.Lock()` (L709). `get_event_bus()` uses `with _EVENT_BUS_LOCK:` (L727). `reset_event_bus()` also locked (L739) |
| 2.11 | Fix global config thread safety in config.py | ✅ DONE | `_CONFIG_LOCK = threading.Lock()` (L1527). `get_config()` uses `with _CONFIG_LOCK:` (L1538). `reset_config()` also locked (L1554) |
| 2.12 | Fix reentrant lock deadlock in future_thinking.py | ✅ DONE | `self._lock = asyncio.Lock()` (L260). `_list_active_unlocked()` (L385) for internal use. `stats()` calls `_list_active_unlocked()` (L427) to avoid re-acquiring lock |

#### Agent 3 — Security Hardening (13 tasks)
**Status:** ⚠️ MOSTLY DONE (5/13 not done or partial)
**Started:** 2026-02-28

| Task | Description | Status | Evidence |
|------|-------------|--------|----------|
| 3.1 | Remove pickle in vector_compression.py (RCE) | ✅ DONE | Uses struct-based binary format with `VECTOR_FORMAT_MAGIC` bytes. `numpy.save`/`np.savez` for arrays (L876). No `pickle.loads` in main paths. Rejects non-magic-byte data (L153) |
| 3.2 | Remove pickle in binary_vector_compression.py (RCE) | ⚠️ PARTIAL | Main path uses `numpy tobytes()` + JSON metadata (L647, L711). However, **legacy fallback still uses `pickle.loads`** at L750-751 for old codebooks — RCE risk remains for legacy data |
| 3.3 | Change CORS default from `["*"]` | ✅ DONE | `config.py` L184: defaults to `["http://localhost:3000", "http://localhost:8100"]`. Env var `HAIM_CORS_ORIGINS` supported (L1076) |
| 3.4 | Remove YAML-based API key support | ✅ DONE | `config.py` L1052-1069: YAML API keys are DEPRECATED with warning. Env var `HAIM_API_KEY` takes precedence. Warning logged if YAML key set without env var |
| 3.5 | Remove `sys.path.insert` hack in daemon.py | ✅ DONE | No `sys.path.insert` found in daemon.py |
| 3.6 | Fix hardcoded `/tmp` log path | 🔍 NEEDS VERIFY | `logging_config.py` uses loguru with no hardcoded `/tmp` path. Daemon.py needs deeper inspection for log path handling |
| 3.7 | Fix URL injection + enforce HTTPS | ✅ DONE | `api_adapter.py` L64: "Uses urllib.parse.urlencode()" (L71). HTTPS enforced in production (L44). Warning for non-HTTPS non-localhost URLs (L51) |
| 3.8 | Fix X-Forwarded-For spoofing | ✅ DONE | `middleware.py` L189-217: Only trusts `X-Forwarded-For` if immediate client is in `trusted_proxies` list. Default: trust nothing (L207) |
| 3.9 | Sanitize error messages in mcp/server.py | ✅ DONE | `_sanitize_error()` function at L68. Error handling uses sanitized messages (L62, L66). Full exceptions logged internally only (L65) |
| 3.10 | Encrypt webhook secrets at rest | ❌ NOT DONE | `webhook_manager.py` L1158: `to_dict(scrub_secret=False)` — secrets saved as **plaintext JSON** to disk. No Fernet/encryption implemented |
| 3.11 | Fix SQL injection in backup_manager.py | ❌ NOT DONE | `backup_manager.py` L369: still uses `f" LIMIT {limit}"` — f-string interpolation in SQL. Should be parameterized `LIMIT ?` |
| 3.12 | Fix hardcoded CORS `["*"]` fallback | ❌ NOT DONE | `api/main.py` L280: `cors_origins = config.security.cors_origins if hasattr(config, "security") else ["*"]` — still falls back to `["*"]` |
| 3.13 | Require API key at startup | ✅ DONE | `api/main.py` L162-164: checks `_api_key`, calls `sys.exit(1)` if empty with `logger.critical()` message |

---

### PHASE 2 - STRUCTURAL

#### Agent 4 — Engine Refactoring (9 tasks)
**Status:** ❌ NOT STARTED

| Task | Description | Status | Evidence |
|------|-------------|--------|----------|
| 4.1 | Decompose `store()` (100+ lines) | ❌ NOT DONE | engine_core.py not refactored |
| 4.2 | Decompose `query()` (170+ lines) | ❌ NOT DONE | — |
| 4.3 | Add exception callbacks to `ensure_future` | ❌ NOT DONE | — |
| 4.4 | Remove duplicate `_run_in_thread` | ❌ NOT DONE | — |
| 4.5 | Clean up unused imports in engine.py | ❌ NOT DONE | — |
| 4.6 | Extract inline imports from `__init__` | ❌ NOT DONE | — |
| 4.7 | Replace `getattr` config access | ❌ NOT DONE | — |
| 4.8 | Refactor `load_config()` god function | ❌ NOT DONE | — |
| 4.9 | Add config validation | ❌ NOT DONE | — |

#### Agent 5 — Error Handling & Resilience (10 tasks)
**Status:** ❌ NOT STARTED

| Task | Description | Status | Evidence |
|------|-------------|--------|----------|
| 5.1 | Replace silent exception swallowing | ❌ NOT DONE | — |
| 5.2 | Harden `close()` shutdown | ❌ NOT DONE | — |
| 5.3 | Fix lying health check | ❌ NOT DONE | — |
| 5.4 | Roll back `_initialized` on failure | ❌ NOT DONE | — |
| 5.5 | Fix swallowed I/O errors in `_load_legacy` | ❌ NOT DONE | — |
| 5.6 | Replace bare `except:` blocks in daemon.py | ❌ NOT DONE | — |
| 5.7 | Return structured errors from LLM | ❌ NOT DONE | — |
| 5.8 | Add JSON error handling in goal_tree + learning_journal | ❌ NOT DONE | — |
| 5.9 | Replace broad exception swallowing in backup_manager | ❌ NOT DONE | — |
| 5.10 | Improve error logging in prediction_store | ❌ NOT DONE | — |

#### Agent 6 — Storage Layer Fixes (9 tasks)
**Status:** ❌ NOT STARTED

| Task | Description | Status | Evidence |
|------|-------------|--------|----------|
| 6.1 | Replace blocking SQLite in backup_manager.py | ❌ NOT DONE | Still uses aiosqlite — may already use it, needs verification |
| 6.2 | Replace blocking SQLite in binary_vector_compression.py | ❌ NOT DONE | — |
| 6.3 | Fix `cleanup_old_codebooks` bug | ❌ NOT DONE | — |
| 6.4 | Fix dangling import + streaming reads in memory_importer.py | ❌ NOT DONE | — |
| 6.5 | Stream large exports in memory_exporter.py | ❌ NOT DONE | Tests still failing |
| 6.6 | Prevent destructive collection recreation in qdrant_store.py | ❌ NOT DONE | — |
| 6.7 | Replace SQLite with aiosqlite in memory_importer.py | ❌ NOT DONE | — |
| 6.8 | Fix aiohttp timeout misuse in webhook_manager.py | ❌ NOT DONE | — |
| 6.9 | Queue retries in webhook_manager.py | ❌ NOT DONE | — |

#### Agent 7 — Subconscious/Dream Refactoring (8 tasks)
**Status:** ❌ NOT STARTED

| Task | Description | Status |
|------|-------------|--------|
| 7.1 | Split dream_pipeline.py into per-stage modules | ❌ NOT DONE |
| 7.2 | Replace buggy cron parser | ❌ NOT DONE |
| 7.3 | Remove hardcoded paths in daemon.py | ❌ NOT DONE |
| 7.4 | Reuse aiohttp session in daemon.py | ❌ NOT DONE |
| 7.5 | Fix signal handler for Windows compatibility | ❌ NOT DONE |
| 7.6 | Split llm_integration.py into per-provider modules | ❌ NOT DONE |
| 7.7 | Split forgetting_curve.py god file | ❌ NOT DONE |
| 7.8 | Remove dead code in consolidation.py | ❌ NOT DONE |

#### Agent 8 — API & Middleware Hardening (8 tasks)
**Status:** ❌ NOT STARTED

| Task | Description | Status |
|------|-------------|--------|
| 8.1 | Split api/main.py into route modules | ❌ NOT DONE |
| 8.2 | Move inline models to models.py | ❌ NOT DONE |
| 8.3 | Fix version string inconsistency | ❌ NOT DONE |
| 8.4 | Fix prediction store race condition | ❌ NOT DONE |
| 8.5 | Add rate limiting to /dream | ❌ NOT DONE |
| 8.6 | Enable HSTS header | ❌ NOT DONE |
| 8.7 | Fix sync/async mismatch in agent_interface.py | ❌ NOT DONE |
| 8.8 | Extract CLI boilerplate into decorator | ❌ NOT DONE |

#### Agent 9 — Config & Dependency Cleanup (9 tasks)
**Status:** ❌ NOT STARTED

| Task | Description | Status |
|------|-------------|--------|
| 9.1 | Split requirements.txt into runtime/test | ❌ NOT DONE |
| 9.2 | Sync pyproject.toml with requirements.txt | ❌ NOT DONE |
| 9.3 | Create lockfile for reproducible builds | ❌ NOT DONE |
| 9.4 | Fix pytest.ini / pyproject.toml conflict | ❌ NOT DONE |
| 9.5 | Fix setup.cfg mypy config | ❌ NOT DONE |
| 9.6 | Create .pre-commit-config.yaml | ❌ NOT DONE |
| 9.7 | Sync version numbers across all config files | ❌ NOT DONE |
| 9.8 | Add aiosqlite to dependencies | ❌ NOT DONE |
| 9.9 | Add croniter to dependencies | ❌ NOT DONE |

#### Agent 10 — Docker, Helm & CI/CD (11 tasks)
**Status:** ❌ NOT STARTED

| Task | Description | Status |
|------|-------------|--------|
| 10.1 | Pin Dockerfile base image version | ❌ NOT DONE |
| 10.2 | Remove test deps from production image | ❌ NOT DONE |
| 10.3 | Update Dockerfile version label | ❌ NOT DONE |
| 10.4 | Add entrypoint validation | ❌ NOT DONE |
| 10.5 | Pin Qdrant image version in docker-compose | ❌ NOT DONE |
| 10.6 | Restrict port bindings | ❌ NOT DONE |
| 10.7 | Helm: update versions and pin images | ❌ NOT DONE |
| 10.8 | Helm: enable network policies by default | ❌ NOT DONE |
| 10.9 | CI: make security scans blocking | ❌ NOT DONE |
| 10.10 | CI: add container image scanning | ❌ NOT DONE |
| 10.11 | CI: fix attestation step | ❌ NOT DONE |

---

### PHASE 3 - TESTING

#### Agent 11 — Test Suite: Core Engine (8 tasks)
**Status:** ❌ NOT STARTED

#### Agent 12 — Test Suite: Storage & Import/Export (4 tasks)
**Status:** ❌ NOT STARTED
**Note:** Test files exist (`test_memory_exporter.py`, `test_memory_importer.py`, `test_backup_manager.py`) but 14+ tests currently FAILING

#### Agent 13 — Test Suite: AI & Subconscious (4 tasks)
**Status:** ❌ NOT STARTED

#### Agent 14 — Test Suite: Meta, Cognitive, Events (6 tasks)
**Status:** ❌ NOT STARTED

#### Agent 15 — Test Suite: API, MCP, CLI (7 tasks)
**Status:** ❌ NOT STARTED

#### Agent 16 — Test Suite: Concurrency & Integration (5 tasks)
**Status:** ❌ NOT STARTED

---

### PHASE 4 - POLISH

#### Agent 17 — Performance Optimization (8 tasks)
**Status:** ❌ NOT STARTED

#### Agent 18 — Documentation Overhaul (10 tasks)
**Status:** ❌ NOT STARTED

#### Agent 19 — Metrics & Observability (6 tasks)
**Status:** ❌ NOT STARTED

#### Agent 20 — Final Validation & Release Prep (8 tasks)
**Status:** ❌ NOT STARTED

---

## Summary: Phase 1 Scorecard

### Agent 1 — CRITICAL Bug Fixes: ✅ 11/11 DONE
All critical bugs verified as fixed in source code.

### Agent 2 — Thread Safety & Concurrency: ✅ 12/12 DONE
All thread safety fixes verified in source code. Every file has proper locking.

### Agent 3 — Security Hardening: ⚠️ 8/13 DONE (3 not done, 1 partial, 1 needs verify)
**Still open:**
- 3.2: binary_vector_compression.py still has legacy `pickle.loads` fallback (RCE risk)
- 3.10: Webhook secrets stored in plaintext JSON on disk (no encryption)
- 3.11: SQL injection in backup_manager.py LIMIT clause (`f" LIMIT {limit}"`)
- 3.12: CORS `["*"]` fallback still present in api/main.py
- 3.6: Hardcoded `/tmp` path needs deeper verification in daemon.py

---

## Issues & Blockers

| Time | Agent | Issue | Resolution |
|------|-------|-------|------------|
| 2026-02-28 | 3 | 3.10 Webhook secrets plaintext on disk | OPEN — needs Fernet encryption |
| 2026-02-28 | 3 | 3.11 SQL injection in LIMIT clause | OPEN — use parameterized query |
| 2026-02-28 | 3 | 3.12 CORS `["*"]` fallback in main.py | OPEN — change fallback to `[]` |
| 2026-02-28 | 3 | 3.2 Legacy pickle.loads in binary_vector_compression | OPEN — remove or sandbox legacy fallback |
| 2026-02-28 | 12 | 14 test_memory_importer tests FAILING | Blocks Phase 3 acceptance |
| 2026-02-28 | 12 | 7 test_memory_exporter tests FAILING | Blocks Phase 3 acceptance |
| 2026-02-28 | — | 133 total test failures in suite | Needs triage before Phase 4 |

---

## Completion Log

| Time | Agent | Tasks Completed | Notes |
|------|-------|-----------------|-------|
| 2026-02-28 | Agent 1 | 11/11 | All critical bug fixes verified |
| 2026-02-28 | Agent 2 | 12/12 | All thread safety fixes verified |
| 2026-02-28 | Agent 3 | 8/13 | 5 tasks still open (3.2, 3.6, 3.10, 3.11, 3.12) |

---

## Failing Tests (133 failures, 12 errors)

Key failure clusters:
- `test_memory_exporter.py` — 7 failures (export logic broken)
- `test_memory_importer.py` — 7 failures (import logic broken)
- `test_llm_integration.py` — 6 failures (LLM mock/integration issues)
- `test_semantic_consolidation.py` — 5 failures
- `test_process.py` — 10 failures (process priority tests)
- `test_mcp_adapter.py` — 4 failures (URL encoding, retry, timeout)
- `test_immunology.py` — 4 failures
- `test_context_optimizer_full.py` — 12 errors (module-level issue)
- `test_associations.py` — 2 failures
- `test_api_security*.py` — 5 failures
- Other scattered failures across test suite

---

*Last audited: 2026-02-28 via automated source code verification.*
