# Persistent Agent Memory Quickstart

MnemoCore's agent-memory API provides durable, exact-scoped memory backed by a
local SQLite file. The `mnemocore.agent_memory` implementation itself uses only
Python's standard library at runtime and is available through both asynchronous
and explicit synchronous clients; installing the complete MnemoCore package may
install additional dependencies used by its other modules.

## Install

From a cloned repository:

```bash
python -m pip install -e .
```

## Asynchronous client

```python
import asyncio

from mnemocore.agent_memory import AgentMemory, MemoryKind, MemoryScope


async def main() -> None:
    scope = MemoryScope(
        tenant_id="local",
        user_id="robin",
        agent_id="codex",
        project_id="mnemocore",
    )

    async with await AgentMemory.open("./data/agent-memory.db", scope=scope) as memory:
        stored = await memory.remember(
            "Prefer concise implementation updates",
            kind=MemoryKind.PREFERENCE,
            idempotency_key="update-style-v1",
        )
        results = await memory.recall("concise updates", limit=5)
        print(results[0].memory.content)

        # Forget is an auditable state transition, not physical erasure.
        await memory.forget(stored.id, reason="preference changed")


asyncio.run(main())
```

`AgentMemory.open()` creates the parent directory and SQLite database when
needed. Relative paths are resolved from the process working directory; use an
absolute path when several processes must agree on the same file location.

## Synchronous client

Use `SyncAgentMemory` only in synchronous code. Inside an active event loop,
use `AgentMemory` instead.

```python
from mnemocore.agent_memory import MemoryScope, SyncAgentMemory

scope = MemoryScope(user_id="robin", agent_id="local-agent")

with SyncAgentMemory.open("./data/agent-memory.db", scope=scope) as memory:
    stored = memory.remember(
        "SQLite memory survives process restarts",
        idempotency_key="durability-fact-v1",
    )
    print(memory.get(stored.id).content)
```

The sync client has the same `remember`, `supersede`, bitemporal `recall`,
`explain`, `get`, `list`, `history`, `forget`, and `rebuild` operations as the
async client. It owns one private event loop for its lifetime and rejects calls
made from inside a running event loop.

## Truth timeline: validity time and knowledge time

Timeline queries distinguish two clocks:

- `valid_at` asks when the fact is true in the modeled world.
- `known_at` asks what the memory ledger had recorded at that time.

Both accept timezone-aware ISO 8601 timestamps. `known_at` defaults to now.
The legacy `as_of` argument remains an alias for `valid_at`; do not pass both.

Validity windows are half-open: `[valid_from, valid_to)`. If a fact is
superseded at `2026-07-11T09:30:00.000001Z`, the source is valid up to but not
including that instant, and the replacement is valid from that exact instant.
An exact-boundary recall therefore returns the replacement only.

```python
source = await memory.remember(
    "Launch date is July 20",
    kind=MemoryKind.FACT,
    valid_from="2026-07-01T00:00:00Z",
)
replacement = await memory.supersede(
    source.id,
    "Launch date is July 27",
    effective_at="2026-07-11T09:30:00.000001Z",
    reason="vendor correction",
    idempotency_key="launch-date-v2",
)

# Use a knowledge instant at or after the supersession event.
known_at = replacement.updated_at.isoformat(timespec="microseconds").replace(
    "+00:00", "Z"
)
before = await memory.recall(
    "launch date",
    valid_at="2026-07-11T09:30:00.000000Z",
    known_at=known_at,
)
at_boundary = await memory.recall(
    "launch date",
    valid_at="2026-07-11T09:30:00.000001Z",
    known_at=known_at,
)

assert before[0].memory.id == source.id
assert at_boundary[0].memory.id == replacement.id
```

`explain()` returns the selected memory plus same-scope immutable evidence,
relations, history, and a deterministic explanation. It does not expose the
free-form supersession reason or private metadata in the explanation text.

```python
receipt = await memory.explain(
    replacement.id,
    valid_at="2026-07-11T09:30:00.000001Z",
    known_at=known_at,
)

assert receipt.memory.id == replacement.id
assert receipt.evidence_memory_ids == (source.id,)
print(receipt.evidence_event_ids)
print(receipt.explanation)
```

Closing and reopening a client preserves the timeline. Opening a foundation
schema-v1 database migrates it transactionally to schema v2 before use; facts
can then be superseded, closed, reopened in another process, and recalled on
either side of the boundary. Keep the SQLite file on durable storage and use
the same absolute path after restart.

## Scope and lifecycle semantics

A scope is the exact tuple of `tenant_id`, `user_id`, `agent_id`, optional
`project_id`, and optional `session_id`. Operations never fall back to a wider
scope: a client can only get, list, recall, inspect history for, or forget a
record in its complete scope. Reusing an idempotency key in the same scope
returns the existing memory instead of creating a duplicate.

`forget()` is logical and auditable, not physical deletion or a data-erasure
mechanism. It marks the projection as forgotten, records an immutable ledger
event plus an audit-history entry, and removes the memory from lexical search.
The content and event remain in the SQLite file and are available through
`get(..., include_forgotten=True)` for audit and rebuild. Applications with
hard-deletion or regulatory-erasure requirements need a separate lifecycle for
the database file and its backups.

If a projection or search index is damaged, `rebuild()` repairs the client's
exact scope from its immutable events. It does not read or rewrite another
scope.

## Retrieval status

Recall currently uses only scoped SQLite FTS5 lexical retrieval, followed by
validity-time, knowledge-time, and scope filtering. It does not perform vector,
embedding, semantic, or model-based retrieval, and no benchmark leadership is
claimed for this foundation. Queries therefore need useful lexical overlap
with stored content.

The planned hybrid pipeline will combine lexical and vector candidates, apply
scope and temporal policy before ranking, fuse and rerank candidates, and return
evidence-bearing results. That pipeline is future work; applications should
treat the current scores as lexical ranks.
