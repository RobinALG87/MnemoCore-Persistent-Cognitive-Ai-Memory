# MnemoCore Truth Timeline Design

**Status:** Approved continuation of Agent Memory vNext
**Date:** 2026-07-10
**Objective:** Make changing facts temporally correct, fully attributable, and rebuildable before adding broader semantic retrieval.

## 1. Decision

Three next-step routes were evaluated:

1. **Truth Timeline first (selected):** add supersession, bitemporal recall, complete lineage, and memory receipts. This prevents a stronger retriever from confidently returning stale facts.
2. **Hybrid retrieval first:** improves recall coverage quickly, but the current projection cannot reliably distinguish current truth from historical knowledge.
3. **Agent lifecycle first:** unlocks episodes and procedures, but those derived memories need the same provenance and temporal foundation.

This milestone implements one narrow vertical slice: evidence-backed fact supersession and explanation. Contradiction resolution, embeddings, sessions, compaction, and LLM extraction remain later milestones.

## 2. Product Behavior

Given an active fact, an agent can atomically replace it with a new fact effective at a precise real-world time:

```python
replacement = await memory.supersede(
    old_memory_id,
    "The deployment window is 16:00 UTC",
    effective_at="2026-07-10T12:00:00Z",
    reason="Operations changed the schedule",
    idempotency_key="deployment-window-v2",
)
```

Current recall returns the fact valid now. Historical recall separates two clocks:

- `valid_at`: when the proposition is true in the represented world.
- `known_at`: what MnemoCore had learned by that transaction time.

`as_of` remains a compatibility alias for `valid_at`; supplying both is a validation error. Omitting either clock defaults it to the current UTC time.

`explain(memory_id)` returns a frozen `MemoryReceipt` containing the record, source event IDs, source memory IDs, relations, lifecycle history, confidence, validity, and a deterministic plain-language explanation.

## 3. Invariants

1. Every operation uses one exact canonical `MemoryScope`; foreign IDs are indistinguishable from missing IDs.
2. Supersession accepts only active `MemoryKind.FACT` records.
3. `effective_at` is timezone-aware UTC after normalization and must be later than the old fact's `valid_from`, when present.
4. The old fact's `valid_to` becomes the exact effective boundary; the replacement's `valid_from` is the same boundary. Intervals remain inclusive at the start and exclusive at the end.
5. One immutable `SUPERSEDED` event contains complete old and replacement snapshots, reason, relation, and evidence data sufficient to rebuild every projection.
6. Ledger event, both memory projections, lifecycle projection, histories, evidence, relation, and FTS writes commit or roll back together.
7. Idempotent retry returns the same replacement without duplicate events, history, evidence, relations, or FTS rows.
8. Every returned receipt and recall result resolves to immutable event evidence.
9. Rebuild validates all event scopes, IDs, endpoints, and ownership before deleting any projection.
10. No new runtime dependency is required.

## 4. Storage Version 2

Schema version 2 adds a bitemporal lifecycle projection:

```text
memory_lifecycle(
  memory_id, scope_key, status,
  known_from, known_to,
  valid_from, valid_to,
  event_id, created_at
)
```

Rows represent half-open knowledge intervals `[known_from, known_to)`. A remembered memory starts active. Supersession closes the old active interval, adds an old superseded interval, and starts the replacement active interval at the supersession event time.

Version 2 also adds `scope_key` to `memory_evidence` and installs triggers that reject evidence or relation rows when any memory/event endpoint belongs to a different scope. Existing version-1 databases migrate transactionally: validate the complete v1 fingerprint, create the new structures, backfill active/forgotten lifecycle rows from immutable events, migrate the empty foundation evidence projection, install guards, validate the v2 fingerprint, then set `user_version = 2`. Any failure rolls back to an intact v1 database.

The FTS projection retains superseded facts so explicit historical recall can find them. Current status and bitemporal predicates—not index membership—determine visibility. Forgotten records remain absent from FTS; a historical `known_at` before forgetting is therefore not lexical-recallable in this milestone, while `explain` and direct audit history remain available.

## 5. Atomic Supersession

`supersede` runs under `BEGIN IMMEDIATE`:

1. Validate the old fact under exact scope and check idempotency.
2. Normalize timestamps, content, metadata, confidence, and reason before mutation.
3. Append one complete immutable `SUPERSEDED` event.
4. Close the old fact's validity and knowledge intervals; mark its current projection superseded.
5. Insert the active replacement and its lifecycle interval.
6. Append deterministic history rows for both records.
7. Insert exact-scope evidence and a typed `supersedes` relation.
8. Keep the old FTS row and insert the replacement FTS row.
9. Commit; map SQLite failures to path-bearing `StorageError` with the original cause.

Concurrent attempts serialize. After one succeeds, a different idempotency key against the now-superseded source fails with a typed lifecycle conflict rather than creating a fork.

## 6. Recall Policy

Lexical candidate generation remains FTS5-only in this milestone. Candidate hydration joins exact-scope lifecycle rows that contain `known_at`, then applies the memory's `[valid_from, valid_to)` interval to `valid_at`.

Eligible lifecycle states are `active` and `superseded`; the validity interval selects the correct version. `contradicted` and `forgotten` states are excluded. Results remain deterministically ordered by lexical rank and memory ID, preserve score components, and include every relevant source event ID.

This creates four distinct, testable queries:

- current knowledge about current truth;
- current knowledge about past truth;
- past knowledge about truth at that past time;
- past knowledge about a different validity time.

## 7. Public Contract

Add frozen public models:

- `MemoryRelation`: typed source/target link with scope, validity, confidence, event, and timestamps.
- `MemoryReceipt`: memory, evidence event IDs, evidence memory IDs, relations, history, and explanation.

Add async store/client methods with exact sync parity:

```python
supersede(memory_id, content, *, effective_at, reason=None,
          metadata=None, confidence=1.0, idempotency_key=None) -> MemoryRecord

explain(memory_id, *, valid_at=None, known_at=None) -> MemoryReceipt

recall(query, *, kinds=(), limit=10, as_of=None,
       valid_at=None, known_at=None) -> list[RecallResult]
```

Add `MemoryConflictError` for valid lifecycle conflicts. Validation, scope, storage, not-found, and closed-store errors retain their existing meanings.

## 8. Rebuild and Failure Safety

Rebuild preflights the complete exact-scope stream, including supersession snapshots, replacement IDs, relation endpoints, evidence endpoints, and foreign ledger/projection ownership. It then reconstructs memories, lifecycle rows, history, evidence, relations, and FTS in one transaction.

Malformed ordering, duplicate replacement IDs, missing source facts, mismatched payload IDs, cross-scope endpoints, invalid temporal boundaries, or incomplete snapshots fail before cleanup. A failure after cleanup rolls back every projection.

## 9. Verification

Required acceptance tests:

- boundary-microsecond truth selection before, at, and after supersession;
- independent `valid_at` and `known_at` matrix;
- exact-scope supersede/explain/recall isolation;
- retry, restart, and concurrent competing supersessions;
- injected rollback across ledger, memories, lifecycle, history, evidence, relations, and FTS;
- full projection corruption followed by event-only rebuild equivalence;
- schema v1-to-v2 migration success and rollback on malformed v1;
- direct SQL rejection of cross-scope evidence and relations;
- frozen receipt/relation models and lightweight imports;
- all existing agent-memory and focused legacy tests remain green.

## 10. Honest Scope

This milestone does not claim semantic retrieval, automatic contradiction detection, LLM extraction, or benchmark leadership. It establishes the temporal and provenance substrate those capabilities require.
