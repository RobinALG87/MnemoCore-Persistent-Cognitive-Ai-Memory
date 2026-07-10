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

## Scope and lifecycle semantics

A scope is the exact tuple of `tenant_id`, `user_id`, `agent_id`, optional
`project_id`, and optional `session_id`. Operations never fall back to a wider
scope: a client can only get, list, recall, inspect history for, or forget a
record in its complete scope. Reusing an idempotency key in the same scope
returns the existing memory instead of creating a duplicate.

`forget()` is logical and auditable. It marks the projection as forgotten,
records an immutable ledger event plus an audit-history entry, and removes the
memory from lexical search. The record remains available through
`get(..., include_forgotten=True)` for audit purposes.

If a projection or search index is damaged, `rebuild()` repairs the client's
exact scope from its immutable events. It does not read or rewrite another
scope.

## Retrieval status

Recall currently uses scoped SQLite FTS5 lexical retrieval with validity-window
filtering. It does not yet perform semantic or embedding retrieval, and no
benchmark leadership is claimed for this foundation.

The planned hybrid pipeline will combine lexical and vector candidates, apply
scope and temporal policy before ranking, fuse and rerank candidates, and return
evidence-bearing results. That pipeline is future work; applications should
treat the current scores as lexical ranks.
