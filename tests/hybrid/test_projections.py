from __future__ import annotations

import pytest

from mnemocore.agent_memory import MemoryKind, MemoryScope
from mnemocore.hybrid.projections import (
    ExactScopeError,
    MemoryTier,
    rebuild_graph_projection,
    rebuild_tier_projection,
)


def _scope(user_id: str) -> MemoryScope:
    return MemoryScope(tenant_id="tenant", user_id=user_id, agent_id="agent")


def test_tier_projection_is_deterministic_and_derived_only_from_exact_scope_records(tmp_path):
    from mnemocore.agent_memory import SyncAgentMemory

    scope = _scope("local")
    database = tmp_path / "memory.db"
    with SyncAgentMemory.open(database, scope=scope) as memory:
        observation = memory.remember("observed an orchard", kind=MemoryKind.OBSERVATION)
        procedure = memory.remember("follow the orchard procedure", kind=MemoryKind.PROCEDURE)
        fact = memory.remember("orchards have trees", kind=MemoryKind.FACT)
        records = memory.list(limit=10)

    first = rebuild_tier_projection(scope, records)
    second = rebuild_tier_projection(scope, reversed(records))

    assert first == second
    assert first.scope == scope
    assert first.memory_ids(MemoryTier.COLD) == (observation.id,)
    assert first.memory_ids(MemoryTier.HOT) == (procedure.id,)
    assert first.memory_ids(MemoryTier.WARM) == (fact.id,)


def test_graph_projection_is_deterministic_and_rejects_cross_scope_records(tmp_path):
    from mnemocore.agent_memory import SyncAgentMemory

    local_scope = _scope("local")
    foreign_scope = _scope("foreign")
    database = tmp_path / "memory.db"
    with SyncAgentMemory.open(database, scope=local_scope) as memory:
        target = memory.remember("target memory")
        source = memory.remember(
            "source memory", metadata={"related_memory_ids": [target.id]}
        )
        local_records = memory.list(limit=10)
    with SyncAgentMemory.open(database, scope=foreign_scope) as memory:
        foreign = memory.remember("foreign memory")

    projection = rebuild_graph_projection(local_scope, reversed(local_records))

    assert projection.scope == local_scope
    assert projection.node_ids == tuple(sorted((source.id, target.id)))
    assert projection.edges == ((source.id, target.id),)
    with pytest.raises(ExactScopeError, match="does not match the projection scope"):
        rebuild_graph_projection(local_scope, (*local_records, foreign))
