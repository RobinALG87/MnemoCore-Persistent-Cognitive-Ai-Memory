from __future__ import annotations

import sqlite3

import pytest

from mnemocore.agent_memory import (
    AgentMemory,
    MemoryKind,
    MemoryScope,
    StorageError,
    ValidationError,
)
from mnemocore.hybrid import HybridMemoryRuntime
from mnemocore.hybrid.plans import CognitivePlan, ProposedMemory


def _scope() -> MemoryScope:
    return MemoryScope(tenant_id="tenant", user_id="local", agent_id="agent")


@pytest.mark.asyncio
async def test_apply_rolls_back_every_proposal_when_a_later_write_fails(tmp_path):
    database = tmp_path / "memory.db"
    scope = _scope()
    async with await AgentMemory.open(database, scope=scope) as memory:
        source = await memory.remember("source for atomic plan")
        plan = CognitivePlan(
            scope=scope,
            provenance="cognitive-module",
            confidence=0.8,
            proposals=(
                ProposedMemory(
                    "first atomic memory",
                    MemoryKind.FACT,
                    "cognitive-module",
                    0.8,
                    source_memory_ids=(source.id,),
                ),
                ProposedMemory(
                    "forced failure memory",
                    MemoryKind.FACT,
                    "cognitive-module",
                    0.8,
                    source_memory_ids=(source.id,),
                ),
            ),
        )
        with sqlite3.connect(database) as connection:
            connection.execute("""
                CREATE TRIGGER fail_second_plan_write
                BEFORE INSERT ON memories
                WHEN NEW.content = 'forced failure memory'
                BEGIN SELECT RAISE(ABORT, 'injected mid-plan failure'); END;
                """)

        with pytest.raises(StorageError, match="injected mid-plan failure"):
            await HybridMemoryRuntime(memory, scope=scope).apply(plan)

        assert [record.id for record in await memory.list()] == [source.id]


@pytest.mark.asyncio
async def test_apply_rechecks_a_source_invalidated_after_proposal_without_writing(
    tmp_path,
):
    database = tmp_path / "memory.db"
    scope = _scope()
    async with await AgentMemory.open(database, scope=scope) as memory:
        source = await memory.remember("source active while plan is proposed")
        plan = CognitivePlan(
            scope=scope,
            provenance="cognitive-module",
            confidence=0.8,
            proposals=(
                ProposedMemory(
                    "must not survive invalidated source",
                    MemoryKind.FACT,
                    "cognitive-module",
                    0.8,
                    source_memory_ids=(source.id,),
                ),
            ),
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE memories SET status = ? WHERE id = ?",
                ("contradicted", source.id),
            )

        with pytest.raises(ValidationError, match="source must be active"):
            await HybridMemoryRuntime(memory, scope=scope).apply(plan)

        assert [record.content for record in await memory.list()] == []
