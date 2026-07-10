# Agent Memory Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a persistent, scope-safe, typed local agent-memory vertical slice with SQLite, FTS5, history, and async/sync Python APIs.

**Architecture:** Add a focused `mnemocore.agent_memory` package beside the legacy engine. SQLite is the canonical local ledger and projection store; every write is transactional, retrieval is exact-scope and lexical in this first slice, and later hybrid providers build on the same typed contracts.

**Tech Stack:** Python 3.10+, standard-library `sqlite3`, `dataclasses`, `asyncio`, `json`, `uuid`, `pytest`, `pytest-asyncio`.

## Global Constraints

- Do not require Redis, Qdrant, FAISS, a graph database, an LLM, or a new dependency.
- Preserve the legacy API and unrelated user files.
- Require `user_id` and `agent_id` for agent-facing scopes; default `tenant_id` to `local`.
- Enforce exact scope in this slice; never use HDV/XOR masking as authorization.
- Store UTC timestamps as ISO-8601 strings ending in `Z`.
- Keep raw events immutable; `forget` is a reversible logical tombstone, not privacy erasure.
- Return typed errors; do not silently downgrade persistence or initialization failures.
- Use TDD and commit each independently working task.

---

## File Map

- `src/mnemocore/agent_memory/models.py`: immutable public models, enums, timestamp helpers, and validation.
- `src/mnemocore/agent_memory/errors.py`: stable exception hierarchy.
- `src/mnemocore/agent_memory/store.py`: async storage protocol.
- `src/mnemocore/agent_memory/sqlite_store.py`: schema, transactions, FTS5, scoped CRUD, history, and lifecycle events.
- `src/mnemocore/agent_memory/client.py`: async primary facade and explicit sync wrapper.
- `src/mnemocore/agent_memory/__init__.py`: supported public exports.
- `tests/agent_memory/test_models.py`: model and scope invariants.
- `tests/agent_memory/test_sqlite_store.py`: persistence, idempotency, scope, temporal, history, and forget behavior.
- `tests/agent_memory/test_client.py`: public async/sync API and import behavior.
- `docs/AGENT_MEMORY_QUICKSTART.md`: runnable foundation usage and limitations.

### Task 1: Public Models and Errors

**Files:**
- Create: `src/mnemocore/agent_memory/__init__.py`
- Create: `src/mnemocore/agent_memory/errors.py`
- Create: `src/mnemocore/agent_memory/models.py`
- Create: `tests/agent_memory/test_models.py`

**Interfaces:**
- Produces: `MemoryScope`, `MemoryKind`, `MemoryStatus`, `MemoryEventType`, `MemoryRecord`, `MemoryEvent`, `MemoryHistoryEntry`, `RecallResult`, `utc_now()`, and the exception hierarchy.
- Consumes: standard library only.

- [ ] **Step 1: Write failing scope and model tests**

```python
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from mnemocore.agent_memory import MemoryKind, MemoryScope
from mnemocore.agent_memory.errors import ScopeError


def test_scope_requires_user_and_agent():
    with pytest.raises(ScopeError):
        MemoryScope(user_id="", agent_id="codex")
    with pytest.raises(ScopeError):
        MemoryScope(user_id="robin", agent_id="")


def test_scope_key_is_stable_and_unambiguous():
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore")
    assert scope.scope_key == '["local","robin","codex","mnemocore",null]'
    assert scope != MemoryScope(user_id="robin", agent_id="codex", project_id="other")


def test_scope_is_immutable():
    scope = MemoryScope(user_id="robin", agent_id="codex")
    with pytest.raises(FrozenInstanceError):
        scope.user_id = "other"


def test_memory_kind_values_are_stable():
    assert MemoryKind.OBSERVATION.value == "observation"
    assert MemoryKind.PROCEDURE.value == "procedure"
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python -m pytest tests/agent_memory/test_models.py -q`  
Expected: FAIL because `mnemocore.agent_memory` does not exist.

- [ ] **Step 3: Implement errors and immutable models**

`errors.py` must define:

```python
class AgentMemoryError(Exception):
    """Base error for the vNext agent-memory API."""

class ValidationError(AgentMemoryError):
    pass

class ScopeError(ValidationError):
    pass

class StorageError(AgentMemoryError):
    pass

class MemoryNotFoundError(AgentMemoryError):
    pass

class ClosedStoreError(StorageError):
    pass
```

`models.py` must use frozen, slotted dataclasses and string enums. `MemoryScope.scope_key` must be compact JSON over `[tenant_id, user_id, agent_id, project_id, session_id]`. Strip identifiers and reject blank required IDs, control characters, and identifiers longer than 256 characters.

```python
class MemoryKind(str, Enum):
    OBSERVATION = "observation"
    FACT = "fact"
    EPISODE = "episode"
    PROCEDURE = "procedure"
    PREFERENCE = "preference"
    SUMMARY = "summary"

class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    FORGOTTEN = "forgotten"

class MemoryEventType(str, Enum):
    REMEMBERED = "remembered"
    REINFORCED = "reinforced"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    FORGOTTEN = "forgotten"
    RESTORED = "restored"
```

`MemoryRecord` fields: `id`, `scope`, `kind`, `content`, `metadata`, `status`, `confidence`, `observed_at`, `valid_from`, `valid_to`, `created_at`, `updated_at`. Validate nonblank content, content length at most 100,000, confidence in `[0, 1]`, and `valid_to > valid_from` when both exist.

`MemoryEvent` fields: `id`, `scope`, `event_type`, `payload`, `occurred_at`, `created_at`, optional `memory_id`, optional `idempotency_key`.

`MemoryHistoryEntry` fields: `id`, `memory_id`, `event_id`, `action`, `status`, `created_at`, `details`.

`RecallResult` fields: `memory`, `score`, `score_components`, `reason`, `evidence_ids`.

Create a minimal `agent_memory/__init__.py` that exports Task 1 models and errors. Task 6 extends this file with client exports.

- [ ] **Step 4: Run model tests**

Run: `python -m pytest tests/agent_memory/test_models.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mnemocore/agent_memory/__init__.py src/mnemocore/agent_memory/errors.py src/mnemocore/agent_memory/models.py tests/agent_memory/test_models.py
git commit -m "feat(memory): add typed agent memory models"
```

### Task 2: Store Protocol and SQLite Schema

**Files:**
- Create: `src/mnemocore/agent_memory/store.py`
- Create: `src/mnemocore/agent_memory/sqlite_store.py`
- Create: `tests/agent_memory/test_sqlite_store.py`

**Interfaces:**
- Consumes: Task 1 models and errors.
- Produces: `MemoryStore` protocol and `SQLiteMemoryStore.open(path)` / `close()` lifecycle.

- [ ] **Step 1: Write failing schema lifecycle test**

```python
import sqlite3

import pytest

from mnemocore.agent_memory import MemoryScope
from mnemocore.agent_memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def scope():
    return MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore")


@pytest.mark.asyncio
async def test_open_creates_versioned_schema(tmp_path):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"memory_events", "memories", "memory_evidence", "memory_relations", "memory_history", "memory_fts"} <= tables
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py::test_open_creates_versioned_schema -q`  
Expected: FAIL because the store is missing.

- [ ] **Step 3: Define the async protocol**

`MemoryStore` must use these exact signatures. Use `typing.Protocol` and `runtime_checkable`; do not add behavior.

```python
@runtime_checkable
class MemoryStore(Protocol):
    async def remember(
        self, scope: MemoryScope, content: str, *,
        kind: MemoryKind = MemoryKind.OBSERVATION,
        metadata: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        confidence: float = 1.0,
        observed_at: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> MemoryRecord: ...
    async def get(
        self, scope: MemoryScope, memory_id: str, *, include_forgotten: bool = False
    ) -> MemoryRecord: ...
    async def list(
        self, scope: MemoryScope, *, kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE, limit: int = 100
    ) -> list[MemoryRecord]: ...
    async def recall(
        self, scope: MemoryScope, query: str, *, kinds: Sequence[MemoryKind] = (),
        limit: int = 10, as_of: Optional[str] = None
    ) -> list[RecallResult]: ...
    async def history(
        self, scope: MemoryScope, memory_id: str
    ) -> list[MemoryHistoryEntry]: ...
    async def forget(
        self, scope: MemoryScope, memory_id: str, *, reason: Optional[str] = None
    ) -> MemoryRecord: ...
    async def close(self) -> None: ...
```

- [ ] **Step 4: Implement schema initialization**

`SQLiteMemoryStore.open(path)` must:

1. resolve and create only the parent directory;
2. open SQLite connections with `timeout=10`;
3. execute `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, and `PRAGMA busy_timeout=10000`;
4. create schema version 1 in a transaction;
5. create a partial unique index on `(scope_key, idempotency_key)` where the key is not null;
6. create compound indexes for `(scope_key, status, kind, created_at)` and temporal validity;
7. create `memory_fts` using FTS5 `tokenize='unicode61'`;
8. raise `StorageError` with the database path when FTS5 or schema initialization fails.

Schema version 1 must store individual scope columns plus `scope_key` on events and memories. Required columns are:

```text
memory_events(id PK, memory_id, scope_key, tenant_id, user_id, agent_id,
  project_id, session_id, event_type, payload_json, idempotency_key,
  occurred_at, created_at)
memories(id PK, scope_key, tenant_id, user_id, agent_id, project_id,
  session_id, kind, content, metadata_json, status, confidence,
  observed_at, valid_from, valid_to, created_at, updated_at)
memory_evidence(memory_id, source_memory_id, event_id, relation, created_at)
memory_relations(id PK, scope_key, source_id, target_id, relation_type,
  valid_from, valid_to, confidence, event_id, created_at)
memory_history(id PK, memory_id, event_id, action, status, details_json, created_at)
memory_fts(memory_id UNINDEXED, content)
```

Add foreign keys from history/evidence/relations to their referenced rows where deletion semantics are unambiguous. The ledger event itself must never cascade-delete.

All public async methods run blocking SQLite work with `asyncio.to_thread` and guard lifecycle state with an `asyncio.Lock`. Each worker opens its own configured connection.

- [ ] **Step 5: Run schema test**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py::test_open_creates_versioned_schema -q`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mnemocore/agent_memory/store.py src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_store.py
git commit -m "feat(memory): add SQLite ledger schema"
```

### Task 3: Transactional Remember, History, and Persistence

**Files:**
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Modify: `tests/agent_memory/test_sqlite_store.py`

**Interfaces:**
- Consumes: `SQLiteMemoryStore.open`, `MemoryRecord`, `MemoryEvent`, exact `MemoryScope`.
- Produces: idempotent `remember`, `get`, `list`, and `history`.

- [ ] **Step 1: Add failing persistence and idempotency tests**

```python
@pytest.mark.asyncio
async def test_remember_is_persistent_and_idempotent(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    first = await store.remember(scope, "Use BM25 candidate union", idempotency_key="decision-1")
    second = await store.remember(scope, "Use BM25 candidate union", idempotency_key="decision-1")
    assert second.id == first.id
    assert len(await store.list(scope)) == 1
    await store.close()

    reopened = await SQLiteMemoryStore.open(path)
    assert (await reopened.get(scope, first.id)).content == "Use BM25 candidate union"
    history = await reopened.history(scope, first.id)
    assert [entry.action for entry in history] == ["remembered"]
    await reopened.close()
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py::test_remember_is_persistent_and_idempotent -q`  
Expected: FAIL because CRUD is not implemented.

- [ ] **Step 3: Implement one-transaction remember**

Inside `BEGIN IMMEDIATE`:

1. check the partial idempotency index and return the existing record when found;
2. insert a `remembered` ledger event whose payload includes the generated memory ID;
3. insert the active memory projection;
4. insert a history row linked to the event;
5. insert the FTS projection;
6. commit or roll back all writes.

Use UUID4 hex strings, canonical compact JSON, and Task 1 timestamp helpers. Convert rows with private `_row_to_*` helpers. Map `sqlite3.Error` to `StorageError` while preserving the original exception as `__cause__`.

- [ ] **Step 4: Implement exact-scope get/list/history**

Every query must include `scope_key = ?`. `get` raises `MemoryNotFoundError` for a missing or foreign-scope ID. `list` defaults to active records, orders newest first, validates `1 <= limit <= 1000`, and supports optional `MemoryKind` and `MemoryStatus` filters.

- [ ] **Step 5: Run store tests**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py -q`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_store.py
git commit -m "feat(memory): persist scoped memory events"
```

### Task 4: Scoped FTS5 Recall and Temporal Validity

**Files:**
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Modify: `tests/agent_memory/test_sqlite_store.py`

**Interfaces:**
- Consumes: stored memory and FTS projections from Task 3.
- Produces: `recall(scope, query, kinds=(), limit=10, as_of=None) -> list[RecallResult]`.

- [ ] **Step 1: Add failing relevance, scope, and temporal tests**

```python
@pytest.mark.asyncio
async def test_recall_is_relevant_scope_safe_and_temporal(tmp_path, scope):
    store = await SQLiteMemoryStore.open(tmp_path / "memory.db")
    await store.remember(scope, "BM25 must introduce lexical candidates")
    await store.remember(scope, "Redis stores transient queue signals")
    foreign = MemoryScope(user_id="other", agent_id="codex", project_id="mnemocore")
    await store.remember(foreign, "BM25 private foreign memory")

    results = await store.recall(scope, "BM25 lexical", limit=5)
    assert [r.memory.content for r in results] == ["BM25 must introduce lexical candidates"]
    assert results[0].score_components["bm25_rank"] == 1.0
    assert results[0].reason == "Matched lexical terms in the authorized scope"
```

Add a second test with one expired, one future, and one currently valid fact; assert only the currently valid fact is returned for the supplied `as_of` time.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py -k recall -q`  
Expected: FAIL because recall is not implemented.

- [ ] **Step 3: Implement safe FTS query construction**

Extract Unicode word tokens with `re.findall(r"\w+", query.casefold())`, reject an empty token set, quote each token for FTS, and join with `OR`. Never pass raw user syntax to `MATCH`.

Join `memory_fts` to `memories`, require exact `scope_key`, active status, optional kinds, and:

```sql
(valid_from IS NULL OR valid_from <= :as_of)
AND (valid_to IS NULL OR valid_to > :as_of)
```

Order by SQLite `bm25(memory_fts)`, over-fetch at `limit * 3`, deduplicate by memory ID, and emit the first `limit`. Use reciprocal lexical rank `1 / rank` as the normalized foundation score and preserve the raw BM25 value in `score_components`.

- [ ] **Step 4: Run recall tests**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py -k recall -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_store.py
git commit -m "feat(memory): add scope-safe lexical recall"
```

### Task 5: Logical Forget and Audit History

**Files:**
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Modify: `tests/agent_memory/test_sqlite_store.py`

**Interfaces:**
- Consumes: transactional ledger/projections.
- Produces: `forget(scope, memory_id, reason=None)`.

- [ ] **Step 1: Add failing forget test**

```python
@pytest.mark.asyncio
async def test_forget_tombstones_and_removes_from_recall(tmp_path, scope):
    store = await SQLiteMemoryStore.open(tmp_path / "memory.db")
    record = await store.remember(scope, "Never repeat this failed migration")
    await store.forget(scope, record.id, reason="incorrect")
    assert await store.recall(scope, "failed migration") == []
    forgotten = await store.get(scope, record.id, include_forgotten=True)
    assert forgotten.status.value == "forgotten"
    assert [h.action for h in await store.history(scope, record.id)] == ["remembered", "forgotten"]
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py::test_forget_tombstones_and_removes_from_recall -q`  
Expected: FAIL because forget is not implemented.

- [ ] **Step 3: Implement forget atomically**

Require exact scope and an active record. In one `BEGIN IMMEDIATE` transaction append a `forgotten` event, update status and `updated_at`, append history with the reason, and delete the FTS projection. A repeated forget returns the already-forgotten record without adding duplicate history.

- [ ] **Step 4: Run all SQLite tests**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_store.py
git commit -m "feat(memory): add audited logical forgetting"
```

### Task 6: Async Facade, Explicit Sync Wrapper, and Public Exports

**Files:**
- Create: `src/mnemocore/agent_memory/client.py`
- Create: `src/mnemocore/agent_memory/__init__.py`
- Create: `tests/agent_memory/test_client.py`

**Interfaces:**
- Consumes: `MemoryStore`, `SQLiteMemoryStore`, Task 1 types.
- Produces: `AgentMemory.open`, `AgentMemory.remember/recall/get/list/history/forget/close`, and `SyncAgentMemory.open` with equivalent sync methods.

- [ ] **Step 1: Add failing public API tests**

```python
import pytest

from mnemocore.agent_memory import AgentMemory, MemoryScope, SyncAgentMemory


@pytest.mark.asyncio
async def test_async_public_round_trip(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore")
    memory = await AgentMemory.open(tmp_path / "memory.db", scope=scope)
    stored = await memory.remember("Prefer minimal public APIs", idempotency_key="preference-1")
    recalled = await memory.recall("minimal APIs")
    assert recalled[0].memory.id == stored.id
    await memory.close()


def test_sync_public_round_trip(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex")
    with SyncAgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        stored = memory.remember("Sync wrapper stays explicit")
        assert memory.get(stored.id).content == stored.content
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/agent_memory/test_client.py -q`  
Expected: FAIL because the public facade is missing.

- [ ] **Step 3: Implement async delegation**

`AgentMemory` stores one immutable default scope and a `MemoryStore`. Its methods delegate without swallowing errors. `open` constructs `SQLiteMemoryStore`. Support `async with`. Reject calls after close with `ClosedStoreError`.

- [ ] **Step 4: Implement sync wrapper without loop bridging**

`SyncAgentMemory` owns a private `AgentMemory` and one persistent private event
loop created with `asyncio.new_event_loop()`. It reuses that loop with
`run_until_complete` for every operation and closes it deterministically with the
client; it does not call `asyncio.run` per method because the store's lifecycle
lock belongs to one loop. If called inside a running event loop, raise
`AgentMemoryError("Use AgentMemory inside async code")`. Support `with`; never
start a worker thread to run a coroutine.

- [ ] **Step 5: Define public exports**

Export only the clients, stable models/enums, and stable errors. Do not import the legacy `HAIMEngine`, Redis, Qdrant, FastAPI, or FAISS from this package.

- [ ] **Step 6: Run client and import tests**

Add an assertion that importing `mnemocore.agent_memory` does not add `mnemocore.core.engine`, `qdrant_client`, `redis`, or `fastapi` to `sys.modules`.

Run: `python -m pytest tests/agent_memory/test_client.py tests/agent_memory/test_models.py -q`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mnemocore/agent_memory/client.py src/mnemocore/agent_memory/__init__.py tests/agent_memory/test_client.py
git commit -m "feat(memory): expose persistent agent memory facade"
```

### Task 7: Quickstart, Focused Verification, and Baseline

**Files:**
- Create: `docs/AGENT_MEMORY_QUICKSTART.md`
- Modify: `README.md`
- Modify: `tests/agent_memory/test_client.py`

**Interfaces:**
- Consumes: the complete foundation API.
- Produces: one documented local workflow and verified compatibility with legacy imports.

- [ ] **Step 1: Add a subprocess smoke test**

Run a fresh Python subprocess with `PYTHONPATH=src` that opens a temporary database, remembers two records, closes, reopens, recalls one, prints its content, and exits zero. Assert stdout is exactly `Persistent across restarts`.

- [ ] **Step 2: Run and verify the smoke test**

Run: `python -m pytest tests/agent_memory/test_client.py -k subprocess -q`  
Expected: PASS only when packaging/imports and persistence work in a fresh process.

- [ ] **Step 3: Write the quickstart**

Document install, async and sync examples, exact-scope behavior, SQLite file location, logical forget semantics, current FTS-only retrieval, and the planned hybrid pipeline. Do not claim semantic retrieval or benchmark leadership yet.

Add a concise README link under Python Library Usage without replacing the legacy HAIM documentation.

- [ ] **Step 4: Run foundation verification**

```bash
python -m pytest tests/agent_memory -q
python -m pytest tests/test_light_memory.py tests/test_e2e_flow.py tests/test_agent_interface.py tests/test_mcp_server.py -q
python -m compileall -q src/mnemocore/agent_memory
git diff --check
```

Expected: all agent-memory tests pass; existing focused legacy tests retain their previous pass/skip behavior; compile and diff checks exit zero.

- [ ] **Step 5: Commit**

```bash
git add docs/AGENT_MEMORY_QUICKSTART.md README.md tests/agent_memory/test_client.py
git commit -m "docs: add persistent agent memory quickstart"
```

## Completion Gate

Foundation is complete only when:

- a memory survives process restart;
- duplicate idempotency keys do not duplicate memories;
- foreign scopes cannot get, list, recall, history, or forget records;
- validity windows affect recall without destructive deletion;
- forget is auditable and removes lexical visibility;
- public imports stay independent of heavy legacy/server modules;
- async and sync APIs pass focused tests;
- legacy focused tests do not regress;
- no new runtime dependency is added.
