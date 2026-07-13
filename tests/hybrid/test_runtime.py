from __future__ import annotations

import asyncio

import pytest

from mnemocore.agent_memory import AgentMemory, MemoryScope, SyncAgentMemory
from mnemocore.hybrid import (
    SCORING_VERSION,
    HybridMemoryRuntime,
    SyncHybridMemoryRuntime,
)


def _scope(user_id: str) -> MemoryScope:
    return MemoryScope(tenant_id="tenant", user_id=user_id, agent_id="agent")


@pytest.mark.asyncio
async def test_recall_is_exact_scope_and_combines_lexical_and_binary_hdv(tmp_path):
    database = tmp_path / "memory.db"
    local_scope = _scope("local")
    foreign_scope = _scope("foreign")

    async with await AgentMemory.open(database, scope=local_scope) as local_memory:
        await local_memory.remember("orchard apples are crisp and sweet")
        runtime = HybridMemoryRuntime(local_memory)

        async with await AgentMemory.open(database, scope=foreign_scope) as foreign_memory:
            await foreign_memory.remember("orchard apples are a private foreign memory")

        results = await runtime.recall(local_scope, "crisp orchard apples")
        foreign_results = await runtime.recall(foreign_scope, "crisp orchard apples")

    assert [result.memory.content for result in results] == [
        "orchard apples are crisp and sweet"
    ]
    assert foreign_results == ()
    assert results[0].lexical_score > 0
    assert 0 <= results[0].hdv_score <= 1


@pytest.mark.asyncio
async def test_recall_exposes_a_content_free_scoring_version_and_is_deterministic(tmp_path):
    scope = _scope("local")
    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        await memory.remember("alpha beta gamma")
        await memory.remember("alpha beta delta")
        runtime = HybridMemoryRuntime(memory)

        first = await runtime.recall(scope, "alpha beta")
        second = await runtime.recall(scope, "alpha beta")

    assert SCORING_VERSION == "hybrid-lexical-binary-hdv-v1"
    assert [item.memory.id for item in first] == [item.memory.id for item in second]
    assert all(item.scoring_version == SCORING_VERSION for item in first)
    assert all(set(item.score_components) == {"lexical", "hdv", "hybrid"} for item in first)


def test_sync_and_async_runtimes_have_recall_parity(tmp_path):
    scope = _scope("local")
    database = tmp_path / "memory.db"
    with SyncAgentMemory.open(database, scope=scope) as memory:
        memory.remember("the red fox crosses the meadow")
        memory.remember("the blue whale crosses the ocean")
        runtime = SyncHybridMemoryRuntime(memory)

        sync_results = runtime.recall(scope, "red fox")

    async def async_recall():
        async with await AgentMemory.open(database, scope=scope) as memory:
            return await HybridMemoryRuntime(memory).recall(scope, "red fox")

    async_results = asyncio.run(async_recall())

    assert [result.memory.content for result in sync_results] == [
        "the red fox crosses the meadow"
    ]
    assert [result.memory.id for result in sync_results] == [
        result.memory.id for result in async_results
    ]
    assert sync_results[0].scoring_version == SCORING_VERSION
