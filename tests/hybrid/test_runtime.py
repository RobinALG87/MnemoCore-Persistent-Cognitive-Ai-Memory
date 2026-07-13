from __future__ import annotations

import asyncio

import pytest

from mnemocore.agent_memory import AgentMemory, MemoryScope, SyncAgentMemory
from mnemocore.hybrid import (
    ExactScopeError,
    SCORING_VERSION,
    HybridMemoryRuntime,
    PlanApplyReceipt,
    SyncHybridMemoryRuntime,
)
from mnemocore.hybrid.plans import CognitivePlan, ProposedMemory, ValidatedPlan, validate_plan
from mnemocore.agent_memory import MemoryKind


def _scope(user_id: str) -> MemoryScope:
    return MemoryScope(tenant_id="tenant", user_id=user_id, agent_id="agent")


@pytest.mark.asyncio
async def test_recall_is_exact_scope_and_combines_lexical_and_binary_hdv(tmp_path):
    database = tmp_path / "memory.db"
    local_scope = _scope("local")
    foreign_scope = _scope("foreign")

    async with await AgentMemory.open(database, scope=local_scope) as local_memory:
        await local_memory.remember("orchard apples are crisp and sweet")
        runtime = HybridMemoryRuntime(local_memory, scope=local_scope)

        async with await AgentMemory.open(database, scope=foreign_scope) as foreign_memory:
            await foreign_memory.remember("orchard apples are a private foreign memory")

        results = await runtime.recall(local_scope, "crisp orchard apples")
        with pytest.raises(ExactScopeError, match="does not match the runtime scope"):
            await runtime.recall(foreign_scope, "crisp orchard apples")

    assert [result.memory.content for result in results] == [
        "orchard apples are crisp and sweet"
    ]
    assert results[0].lexical_score > 0
    assert 0 <= results[0].hdv_score <= 1


@pytest.mark.asyncio
async def test_async_runtime_rejects_a_scope_that_differs_from_agent_memory(tmp_path):
    local_scope = _scope("local")
    foreign_scope = _scope("foreign")
    async with await AgentMemory.open(tmp_path / "memory.db", scope=local_scope) as memory:
        with pytest.raises(ExactScopeError, match="does not match the AgentMemory scope"):
            HybridMemoryRuntime(memory, scope=foreign_scope)


def test_sync_runtime_rejects_a_scope_that_differs_from_agent_memory(tmp_path):
    local_scope = _scope("local")
    foreign_scope = _scope("foreign")
    with SyncAgentMemory.open(tmp_path / "memory.db", scope=local_scope) as memory:
        with pytest.raises(ExactScopeError, match="does not match the AgentMemory scope"):
            SyncHybridMemoryRuntime(memory, scope=foreign_scope)


@pytest.mark.asyncio
async def test_recall_exposes_a_content_free_scoring_version_and_is_deterministic(tmp_path):
    scope = _scope("local")
    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        await memory.remember("alpha beta gamma")
        await memory.remember("alpha beta delta")
        runtime = HybridMemoryRuntime(memory, scope=scope)

        first = await runtime.recall(scope, "alpha beta")
        second = await runtime.recall(scope, "alpha beta")

    assert SCORING_VERSION == "hybrid-lexical-binary-hdv-v1"
    assert [item.memory.id for item in first] == [item.memory.id for item in second]
    assert all(item.scoring_version == SCORING_VERSION for item in first)
    assert all(set(item.score_components) == {"lexical", "hdv", "hybrid"} for item in first)


@pytest.mark.asyncio
async def test_binary_hdv_breaks_a_lexical_tie_deterministically(tmp_path):
    scope = _scope("local")
    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        await memory.remember("apple banana carrot")
        await memory.remember("apple banana")
        runtime = HybridMemoryRuntime(memory, scope=scope)

        results = await runtime.recall(scope, "apple banana")

    assert [item.lexical_score for item in results] == [1.0, 1.0]
    assert results[0].memory.content == "apple banana"
    assert results[0].hdv_score > results[1].hdv_score


def test_sync_and_async_runtimes_have_recall_parity(tmp_path):
    scope = _scope("local")
    database = tmp_path / "memory.db"
    with SyncAgentMemory.open(database, scope=scope) as memory:
        memory.remember("the red fox crosses the meadow")
        memory.remember("the blue whale crosses the ocean")
        runtime = SyncHybridMemoryRuntime(memory, scope=scope)

        sync_results = runtime.recall(scope, "red fox")

    async def async_recall():
        async with await AgentMemory.open(database, scope=scope) as memory:
            return await HybridMemoryRuntime(memory, scope=scope).recall(scope, "red fox")

    async_results = asyncio.run(async_recall())

    assert [result.memory.content for result in sync_results] == [
        "the red fox crosses the meadow"
    ]
    assert [result.memory.id for result in sync_results] == [
        result.memory.id for result in async_results
    ]
    assert sync_results[0].scoring_version == SCORING_VERSION


def test_sync_runtime_applies_the_same_validated_remember_only_plan(tmp_path):
    scope = _scope("local")
    plan = CognitivePlan(
        scope=scope,
        provenance="cognitive-module",
        confidence=0.8,
        proposals=(ProposedMemory("sync proposed memory", MemoryKind.FACT, "cognitive-module", 0.8),),
    )
    with SyncAgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        receipt = SyncHybridMemoryRuntime(memory, scope=scope).apply(validate_plan(scope, plan))

        assert receipt == PlanApplyReceipt(proposal_count=1, applied_count=1)
        assert {record.content for record in memory.list()} == {"sync proposed memory"}


@pytest.mark.asyncio
async def test_apply_persists_a_validated_plan_atomically_and_returns_content_free_receipt(tmp_path):
    scope = _scope("local")
    plan = CognitivePlan(
        scope=scope,
        provenance="cognitive-module",
        confidence=0.8,
        proposals=(
            ProposedMemory("first proposed memory", MemoryKind.FACT, "cognitive-module", 0.8),
            ProposedMemory("second proposed memory", MemoryKind.OBSERVATION, "cognitive-module", 0.8),
        ),
    )

    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        runtime = HybridMemoryRuntime(memory, scope=scope)
        receipt = await runtime.apply(ValidatedPlan(scope=scope, plan=plan, proposal_count=2))

        assert isinstance(receipt, PlanApplyReceipt)
        assert receipt.applied_count == 2
        assert receipt.proposal_count == 2
        assert not hasattr(receipt, "content")
        assert {record.content for record in await memory.list()} == {
            "first proposed memory",
            "second proposed memory",
        }


@pytest.mark.asyncio
async def test_apply_rejects_unvalidated_low_confidence_or_wrong_scope_before_writing(tmp_path):
    scope = _scope("local")
    foreign_scope = _scope("foreign")
    plan = CognitivePlan(
        scope=scope,
        provenance="cognitive-module",
        confidence=0.8,
        proposals=(ProposedMemory("proposed memory", MemoryKind.FACT, "cognitive-module", 0.8),),
    )
    low_confidence = CognitivePlan(
        scope=scope,
        provenance="cognitive-module",
        confidence=0.49,
        proposals=(ProposedMemory("low confidence memory", MemoryKind.FACT, "cognitive-module", 0.8),),
    )

    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        runtime = HybridMemoryRuntime(memory, scope=scope)
        forged_plan = CognitivePlan(
            scope=scope,
            provenance="cognitive-module",
            confidence=0.8,
            proposals=(ProposedMemory("valid before mutation", MemoryKind.FACT, "cognitive-module", 0.8),),
        )
        object.__setattr__(forged_plan.proposals[0], "content", " ")
        with pytest.raises(ValueError, match="content must not be blank"):
            await runtime.apply(ValidatedPlan(scope=scope, plan=forged_plan, proposal_count=1))
        with pytest.raises(ValueError, match="confidence"):
            await runtime.apply(low_confidence)
        with pytest.raises(ExactScopeError):
            await runtime.apply(CognitivePlan(
                scope=foreign_scope,
                provenance="cognitive-module",
                confidence=0.8,
                proposals=(ProposedMemory("foreign memory", MemoryKind.FACT, "cognitive-module", 0.8),),
            ))
        assert await memory.list() == []
