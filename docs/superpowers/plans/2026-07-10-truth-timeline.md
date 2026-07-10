# Truth Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship exact-scope fact supersession, bitemporal lexical recall, event-backed memory receipts, and fully rebuildable provenance projections.

**Architecture:** Migrate the SQLite backend to schema v2 with a lifecycle projection and database-enforced provenance scope guards. Keep the immutable ledger canonical: supersession stores complete snapshots, while memories, lifecycle, history, evidence, relations, and FTS are transactional projections. Isolate schema/migration code and pure timeline validation from the already-large SQLite store.

**Tech Stack:** Python 3.10+, standard-library `sqlite3`, `dataclasses`, `asyncio`, `json`, `uuid`, `pytest`, `pytest-asyncio`.

## Global Constraints

- Every operation uses one exact canonical `MemoryScope`; foreign IDs are indistinguishable from missing IDs.
- Supersession accepts only active `MemoryKind.FACT` records.
- `effective_at` is timezone-aware UTC after normalization and is later than the old `valid_from`, when present.
- Temporal intervals are inclusive at the start and exclusive at the end.
- One immutable `SUPERSEDED` event contains complete old and replacement snapshots sufficient to rebuild every projection.
- Ledger, memories, lifecycle, history, evidence, relation, and FTS writes commit or roll back together.
- Every receipt and recall result resolves to immutable event evidence.
- Rebuild validates complete scope and ownership before projection cleanup.
- Preserve the legacy API, foundation databases, lightweight imports, and unrelated user files.
- Do not add a runtime dependency or copy another project's prompt or implementation.
- Use TDD and commit each independently working task.

---

## File Map

- `src/mnemocore/agent_memory/models.py`: frozen relation and receipt models.
- `src/mnemocore/agent_memory/errors.py`: lifecycle conflict error.
- `src/mnemocore/agent_memory/store.py`: expanded async protocol.
- `src/mnemocore/agent_memory/schema.py`: schema-v2 DDL, fingerprints, guards, migration, and validation.
- `src/mnemocore/agent_memory/timeline.py`: pure timestamp/query/supersession payload validation.
- `src/mnemocore/agent_memory/sqlite_store.py`: scoped transactions, bitemporal recall, receipts, and replay orchestration.
- `src/mnemocore/agent_memory/client.py`: async/sync facade parity.
- `src/mnemocore/agent_memory/__init__.py`: stable exports.
- `tests/agent_memory/test_models.py`: immutable public contracts.
- `tests/agent_memory/test_schema_v2.py`: migration and database scope guards.
- `tests/agent_memory/test_timeline.py`: pure temporal boundaries and payload validation.
- `tests/agent_memory/test_sqlite_timeline.py`: transactions, recall, receipts, concurrency, and rebuild.
- `tests/agent_memory/test_client.py`: facade and subprocess compatibility.
- `docs/AGENT_MEMORY_QUICKSTART.md`: supersession and timeline usage.

### Task 1: Public Timeline Models, Errors, and Protocol

**Files:**
- Modify: `src/mnemocore/agent_memory/models.py`
- Modify: `src/mnemocore/agent_memory/errors.py`
- Modify: `src/mnemocore/agent_memory/store.py`
- Modify: `src/mnemocore/agent_memory/__init__.py`
- Modify: `tests/agent_memory/test_models.py`

**Interfaces:**
- Produces: `MemoryConflictError`, `MemoryRelation`, `MemoryReceipt`, and expanded `MemoryStore` signatures.
- Consumes: existing frozen JSON validation and timestamp normalization.

- [ ] **Step 1: Write failing public-contract tests**

Test that `MemoryRelation` contains `id`, `scope`, `source_id`, `target_id`, `relation_type`, `valid_from`, `valid_to`, `confidence`, `event_id`, and `created_at`. Test that `MemoryReceipt` contains `memory`, tuple-normalized `evidence_event_ids`, tuple-normalized `evidence_memory_ids`, tuple-normalized `relations`, tuple-normalized `history`, and nonblank `explanation`. Mutating nested caller inputs or returned containers must fail or leave the model unchanged. Reject blank IDs/relation/explanation, non-finite confidence, confidence outside `[0, 1]`, and invalid validity intervals.

```python
receipt = MemoryReceipt(
    memory=record,
    evidence_event_ids=["event-1"],
    evidence_memory_ids=["source-1"],
    relations=[relation],
    history=[history],
    explanation="This fact supersedes source-1 at 2026-07-10T12:00:00Z.",
)
assert receipt.evidence_event_ids == ("event-1",)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/agent_memory/test_models.py -k "relation or receipt or conflict" -q`  
Expected: FAIL because the new public types are absent.

- [ ] **Step 3: Implement models, error, and exact protocol signatures**

Add `MemoryConflictError(AgentMemoryError)`. Extend `MemoryStore` with:

```python
async def supersede(
    self, scope: MemoryScope, memory_id: str, content: str, *,
    effective_at: str, reason: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    confidence: float = 1.0,
    idempotency_key: Optional[str] = None,
) -> MemoryRecord: ...

async def explain(
    self, scope: MemoryScope, memory_id: str, *,
    valid_at: Optional[str] = None,
    known_at: Optional[str] = None,
) -> MemoryReceipt: ...
```

Extend `recall` with keyword-only `valid_at` and `known_at`, while retaining `as_of`. Export only stable public types and errors.

- [ ] **Step 4: Run model and import tests**

Run: `python -m pytest tests/agent_memory/test_models.py tests/agent_memory/test_client.py -q`  
Expected: PASS; lightweight import assertions remain green.

- [ ] **Step 5: Commit**

```bash
git add src/mnemocore/agent_memory/models.py src/mnemocore/agent_memory/errors.py src/mnemocore/agent_memory/store.py src/mnemocore/agent_memory/__init__.py tests/agent_memory/test_models.py
git commit -m "feat(memory): define truth timeline contracts"
```

### Task 2: Schema v2 Migration and Scope Guards

**Files:**
- Create: `src/mnemocore/agent_memory/schema.py`
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Create: `tests/agent_memory/test_schema_v2.py`
- Modify: `tests/agent_memory/test_sqlite_store.py`

**Interfaces:**
- Consumes: exact v1 schema and immutable remembered/forgotten events.
- Produces: `SCHEMA_VERSION = 2`, `initialize_or_migrate(connection, path)`, and a validated v2 database.

- [ ] **Step 1: Write failing migration tests**

Create a real v1 fixture with one active and one forgotten memory using the foundation commit's schema/operations. On reopen, assert `PRAGMA user_version = 2`, the new lifecycle rows preserve remembered/forgotten knowledge intervals, and all foundation records/history still hydrate. Snapshot the file before a deliberately malformed-v1 migration and assert migration failure leaves `user_version`, tables, and rows unchanged.

- [ ] **Step 2: Write failing database-guard tests**

Direct SQL inserts into `memory_evidence` and `memory_relations` must abort when target, source, or event belongs to another scope. Same-scope rows must succeed. Exercise both INSERT and UPDATE guards.

- [ ] **Step 3: Verify failures**

Run: `python -m pytest tests/agent_memory/test_schema_v2.py -q`  
Expected: FAIL because schema v2 does not exist.

- [ ] **Step 4: Extract and implement schema ownership**

Move schema constants, exact fingerprints, trigger/index definitions, initialization, and migration code into `schema.py`. Add:

```sql
CREATE TABLE memory_lifecycle (
  memory_id TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  status TEXT NOT NULL,
  known_from TEXT NOT NULL,
  known_to TEXT,
  valid_from TEXT,
  valid_to TEXT,
  event_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(memory_id, known_from, status),
  FOREIGN KEY(memory_id) REFERENCES memories(id),
  FOREIGN KEY(event_id) REFERENCES memory_events(id)
)
```

Add `scope_key TEXT NOT NULL` to the v2 evidence table. Create compound lifecycle lookup indexes and BEFORE INSERT/UPDATE triggers that compare every evidence/relation endpoint's stored `scope_key`. Use `BEGIN IMMEDIATE`; validate v1 before migration and v2 before setting version/commit. Migrate valid existing evidence rows by deriving the target scope and rejecting mismatched endpoints.

- [ ] **Step 5: Backfill lifecycle deterministically**

Replay each exact-scope event stream in `(occurred_at, created_at, id)` order. Remembered events open active intervals. Forgotten events close active intervals and open forgotten intervals. Reject incomplete, out-of-order, cross-scope, duplicate, or foreign-owned streams before mutation.

- [ ] **Step 6: Maintain lifecycle in foundation operations**

`remember` inserts its active lifecycle row in the existing transaction. `forget` closes the current active interval and opens a forgotten interval in the existing transaction. The foundation rebuild path reconstructs remembered/forgotten lifecycle rows and includes them in preflight, cleanup, rollback, and equivalence tests. Deterministic lifecycle identities and timestamps must match migration backfill so reopen/rebuild does not drift.

- [ ] **Step 7: Run schema and foundation tests**

Run: `python -m pytest tests/agent_memory/test_schema_v2.py tests/agent_memory/test_sqlite_store.py -q`  
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/mnemocore/agent_memory/schema.py src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_schema_v2.py tests/agent_memory/test_sqlite_store.py
git commit -m "feat(memory): migrate timeline projections to schema v2"
```

### Task 3: Pure Timeline Policy and Atomic Supersession

**Files:**
- Create: `src/mnemocore/agent_memory/timeline.py`
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Create: `tests/agent_memory/test_timeline.py`
- Create: `tests/agent_memory/test_sqlite_timeline.py`

**Interfaces:**
- Consumes: v2 schema, `MemoryKind.FACT`, canonical snapshot serialization.
- Produces: temporal query normalization, complete supersession payload validation, and `SQLiteMemoryStore.supersede`.

- [ ] **Step 1: Write failing pure-policy tests**

Test UTC normalization, six-digit canonical storage, `as_of` alias behavior, rejection when both `as_of` and `valid_at` are supplied, half-open boundary selection, reason normalization, and complete supersession snapshot validation. Include exact microsecond boundaries and non-UTC aware inputs.

- [ ] **Step 2: Write failing supersession tests**

Assert one transaction creates exactly: one immutable `SUPERSEDED` event; one superseded source projection; one active replacement; closed/open lifecycle intervals; two deterministic history rows; one evidence row; one `supersedes` relation; and two FTS rows. Assert only active facts are accepted, foreign IDs raise `MemoryNotFoundError`, and invalid type/status/time raises `MemoryConflictError` without mutation.

- [ ] **Step 3: Verify failures**

Run: `python -m pytest tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py -k supersede -q`  
Expected: FAIL because timeline policy and supersede are absent.

- [ ] **Step 4: Implement pure policy helpers**

`timeline.py` must be standard-library-only and expose private-package helpers for:

```python
normalize_timeline_query(*, as_of, valid_at, known_at, now) -> tuple[str, str]
build_superseded_payload(source, replacement, *, reason, relation_id) -> Mapping[str, Any]
parse_superseded_payload(event, *, path) -> SupersessionReplay
```

Payloads contain complete normalized source/replacement snapshots, their IDs/scope, effective boundary, reason, relation ID/type, and evidence endpoints. Parsing must detach/validate every value and never trust denormalized payload scope.

- [ ] **Step 5: Implement atomic exact-scope supersede**

Use `BEGIN IMMEDIATE`. Perform idempotency lookup before retry payload validation; a prior matching supersession returns its replacement. Validate all new input before the first INSERT. A competing non-idempotent transition against a non-active source raises `MemoryConflictError`. Use deterministic history IDs derived from the event and record role. Preserve the old FTS row and insert the replacement row.

- [ ] **Step 6: Add rollback and concurrency tests**

Inject failures after every projection boundary and compare ledger, memories, lifecycle, history, evidence, relations, and FTS snapshots. Race two store instances: same idempotency key yields one replacement; different keys yield one success and one conflict.

- [ ] **Step 7: Run focused tests and commit**

Run: `python -m pytest tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py -q`  
Expected: PASS.

```bash
git add src/mnemocore/agent_memory/timeline.py src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_timeline.py tests/agent_memory/test_sqlite_timeline.py
git commit -m "feat(memory): add atomic fact supersession"
```

### Task 4: Bitemporal Recall and Memory Receipts

**Files:**
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Modify: `tests/agent_memory/test_sqlite_timeline.py`

**Interfaces:**
- Consumes: lifecycle/evidence/relation projections and timeline query normalization.
- Produces: bitemporal `recall` and exact-scope `explain`.

- [ ] **Step 1: Write the failing four-query temporal matrix**

Create one source fact and one later-known replacement with an earlier/later validity boundary. Assert current/current, current/past, past/past, and past/different-validity queries return the correct version. Test one microsecond before, exactly at, and one microsecond after the boundary. `as_of` must match `valid_at` compatibility behavior.

- [ ] **Step 2: Write failing receipt/provenance tests**

The replacement receipt must resolve its supersession event, source memory, relation, both histories where relevant, confidence, validity, and deterministic explanation. Recall results must contain the remembered/superseded evidence chain without duplicates. Foreign-scope `explain` raises the same not-found error as a random ID.

- [ ] **Step 3: Implement lifecycle-aware recall**

Join FTS, memories, and the lifecycle interval containing `known_at`; allow active/superseded states, exclude contradicted/forgotten, and apply the memory validity interval to `valid_at`. Keep parameterized safe MATCH construction, candidate over-fetch, deterministic `(bm25, memory_id)` tie-breaking, score components, and exact scope.

- [ ] **Step 4: Implement receipts**

Read exact-scope memory, evidence, relations, history, and immutable event IDs in one consistent read transaction. Validate hydrated endpoint scope even though database guards exist. Build explanation text from stable templates; never call an LLM or expose private content from evidence payloads.

- [ ] **Step 5: Run recall/receipt tests and commit**

Run: `python -m pytest tests/agent_memory/test_sqlite_timeline.py -k "recall or explain or receipt or temporal" -q`  
Expected: PASS.

```bash
git add src/mnemocore/agent_memory/sqlite_store.py tests/agent_memory/test_sqlite_timeline.py
git commit -m "feat(memory): add bitemporal recall receipts"
```

### Task 5: Supersession-Aware Event Rebuild

**Files:**
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Modify: `src/mnemocore/agent_memory/timeline.py`
- Modify: `tests/agent_memory/test_sqlite_timeline.py`

**Interfaces:**
- Consumes: complete remembered/forgotten/superseded event streams.
- Produces: event-only reconstruction of every v2 projection.

- [ ] **Step 1: Write failing rebuild equivalence test**

Snapshot memories, lifecycle, history, evidence, relations, FTS, recall results, and receipts after remember → supersede → forget-an-unrelated-record. Corrupt/delete every projection while retaining events, call `rebuild(scope)`, and assert canonical snapshot equivalence including stable IDs.

- [ ] **Step 2: Write adversarial preflight tests**

Before any cleanup, reject: incomplete snapshots; event/payload scope mismatch; source/replacement column mismatch; relation/evidence endpoint mismatch; invalid event ordering; duplicate replacement; cross-scope projection owner; cross-scope ledger-only owner; and boundary-invalid snapshots. Prove both scopes remain byte-for-byte unchanged.

- [ ] **Step 3: Extend replay and scoped cleanup**

Parse all events into an in-memory replay plan first. Validate ownership for every source, replacement, relation, and evidence ID across both `memories` and `memory_events`. Only then clean the exact scope and insert the complete projected state in one transaction. Retain cancellation-safe lifecycle locking.

- [ ] **Step 4: Test rollback after cleanup**

Install a temporary trigger that fails a late relation/FTS write. Assert `StorageError` preserves the SQLite cause and the pre-rebuild projection snapshot is fully restored.

- [ ] **Step 5: Run all store tests and commit**

Run: `python -m pytest tests/agent_memory/test_sqlite_store.py tests/agent_memory/test_sqlite_timeline.py -q`  
Expected: PASS.

```bash
git add src/mnemocore/agent_memory/sqlite_store.py src/mnemocore/agent_memory/timeline.py tests/agent_memory/test_sqlite_timeline.py
git commit -m "feat(memory): rebuild temporal provenance projections"
```

### Task 6: Facade Parity, Documentation, and Final Verification

**Files:**
- Modify: `src/mnemocore/agent_memory/client.py`
- Modify: `src/mnemocore/agent_memory/__init__.py`
- Modify: `tests/agent_memory/test_client.py`
- Modify: `docs/AGENT_MEMORY_QUICKSTART.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: complete store timeline API.
- Produces: async/sync supersede, explain, and bitemporal recall with documented restart behavior.

- [ ] **Step 1: Write failing async/sync parity tests**

Exercise `supersede`, `recall(valid_at=..., known_at=...)`, and `explain` through both clients. Verify closed clients, sync-inside-async rejection, private-loop reuse, and lightweight imports retain existing behavior.

- [ ] **Step 2: Add a subprocess timeline smoke test**

A fresh process opens a v1 foundation database, migrates it, supersedes a fact, closes, reopens, recalls both sides of the validity boundary, validates a receipt, prints `Truth timeline survives restart`, and exits zero.

- [ ] **Step 3: Implement facade delegation and docs**

Add exact method parity without swallowing errors. Update quickstart with the two clocks, boundary semantics, supersession example, receipt example, logical-forget limitation, and honest FTS-only statement. Keep README changes concise.

- [ ] **Step 4: Run final verification**

```bash
python -m pytest tests/agent_memory -q
python -m pytest tests/test_light_memory.py tests/test_e2e_flow.py tests/test_agent_interface.py tests/test_mcp_server.py -q
python -m compileall -q src/mnemocore/agent_memory
python -m flake8 src/mnemocore/agent_memory tests/agent_memory --select=E9,F63,F7,F82
git diff --check
```

Expected: all commands exit zero; no new runtime dependency; legacy focused behavior unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/mnemocore/agent_memory/client.py src/mnemocore/agent_memory/__init__.py tests/agent_memory/test_client.py docs/AGENT_MEMORY_QUICKSTART.md README.md
git commit -m "docs: ship truth timeline quickstart"
```

## Completion Gate

Truth Timeline is complete only when:

- current and point-in-time recall distinguish validity time from knowledge time;
- exact-boundary supersession returns one and only one valid fact;
- receipts resolve complete same-scope immutable evidence;
- migration and rebuild preserve or atomically restore every projection;
- cross-scope provenance is rejected by both application preflight and SQLite guards;
- concurrent/idempotent transitions cannot fork a fact lineage;
- async/sync/subprocess workflows pass;
- foundation and focused legacy tests do not regress.
