from dataclasses import dataclass

import pytest

from mnemocore.integrations import (
    AgentMemoryBridge,
    CrewAIMemoryTools,
    IntegrationPolicy,
    LangGraphMemory,
    MCPMemoryTools,
    OpenClawMemory,
    create_mcp_server,
)
from mnemocore.agent_memory import MemoryKind, MemoryReceipt, MemoryScope
from mnemocore.integrations.mcp import _json_safe


@dataclass
class FakeAsyncMemory:
    async def compile_context(self, query, *, token_budget, include_ancestors):
        return {
            "query": query,
            "token_budget": token_budget,
            "include_ancestors": include_ancestors,
            "text": f"memory for {query}",
        }

    async def remember(self, content, *, kind):
        return {"content": content, "kind": kind.value}


@dataclass
class FakeSyncMemory:
    def compile_context(self, query, *, token_budget, include_ancestors):
        return {"query": query, "token_budget": token_budget}

    def remember(self, content, *, kind):
        return {"content": content, "kind": kind.value}


@pytest.mark.asyncio
async def test_bridge_enforces_one_policy_for_context_and_writes():
    bridge = AgentMemoryBridge(
        FakeAsyncMemory(),
        policy=IntegrationPolicy(max_context_tokens=64, include_ancestors=False),
    )

    context = await bridge.context("fix retrieval")
    stored = await bridge.remember("observed failure")

    assert context["token_budget"] == 64
    assert context["include_ancestors"] is False
    assert stored["kind"] == "observation"


@pytest.mark.asyncio
async def test_langgraph_and_openclaw_return_framework_native_dicts():
    bridge = AgentMemoryBridge(FakeAsyncMemory())

    graph_update = await LangGraphMemory(bridge).context_node(
        {"goal": "plan migration"}
    )
    openclaw_update = await OpenClawMemory(bridge).before_turn(
        {"goal": "plan migration"}
    )

    assert graph_update["memory_context"]["query"] == "plan migration"
    assert openclaw_update["memory_context"]["query"] == "plan migration"


def test_crewai_tools_are_plain_sync_callables_with_bounded_context():
    tools = CrewAIMemoryTools(FakeSyncMemory(), max_context_tokens=32)

    context = tools.recall("ship release")
    stored = tools.remember("release passed", kind="observation")

    assert context["token_budget"] == 32
    assert stored["kind"] == "observation"


@pytest.mark.asyncio
async def test_mcp_tools_expose_small_allowlisted_surface():
    tools = MCPMemoryTools(AgentMemoryBridge(FakeAsyncMemory()))

    context = await tools.memory_recall("MCP contract")
    stored = await tools.memory_remember("MCP memory")

    assert context["query"] == "MCP contract"
    assert stored["kind"] == "observation"
    assert tools.allowed_tools == frozenset({"memory_recall", "memory_remember"})


def test_mcp_server_factory_is_lazy_and_constructible():
    pytest.importorskip("mcp")

    server = create_mcp_server(FakeAsyncMemory(), name="Test Memory")

    assert server is not None


def test_mcp_json_safe_handles_frozen_receipts():
    receipt = MemoryReceipt(
        memory_id="memory-1",
        scope=MemoryScope(user_id="robin", agent_id="codex"),
        kind=MemoryKind.PREFERENCE,
        score=1.0,
        score_components={"bm25": 1.0},
        reason="policy match",
        evidence_ids=("event-1",),
        estimated_tokens=3,
    )

    safe = _json_safe(receipt)

    assert safe["memory_id"] == "memory-1"
    assert safe["scope"]["user_id"] == "robin"
    assert safe["score_components"] == {"bm25": 1.0}
