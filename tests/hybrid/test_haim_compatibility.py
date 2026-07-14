"""v3 compatibility coverage for the deprecated HAIM bridge."""

import pytest

from mnemocore.agent_memory import AgentMemory, MemoryScope
from mnemocore.core.engine import HAIMEngineAdapter
from mnemocore.hybrid import ExactScopeError


def _scope(user_id: str) -> MemoryScope:
    return MemoryScope(
        tenant_id="local", user_id=user_id, agent_id="legacy", project_id="v3"
    )


@pytest.mark.asyncio
async def test_deprecated_adapter_bridges_store_query_and_delete_in_its_exact_scope(
    tmp_path,
):
    scope = _scope("robin")
    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        with pytest.deprecated_call(match="HAIMEngineAdapter is deprecated"):
            adapter = HAIMEngineAdapter(memory, scope=scope)

        memory_id = await adapter.store(
            "Robin prefers concise responses",
            metadata={"source": "legacy"},
            goal_id="response-style",
            project_id="v3",
        )

        results = await adapter.query("concise response", top_k=1)
        assert results[0][0] == memory_id
        assert 0.0 <= results[0][1] <= 1.0

        stored = await memory.get(memory_id)
        assert dict(stored.metadata) == {
            "source": "legacy",
            "goal_id": "response-style",
            "project_id": "v3",
        }

        assert await adapter.delete_memory(memory_id) is True
        assert await memory.list() == []


@pytest.mark.asyncio
async def test_deprecated_adapter_requires_the_clients_exact_explicit_scope(tmp_path):
    local_scope = _scope("robin")
    foreign_scope = _scope("alex")
    async with await AgentMemory.open(
        tmp_path / "memory.db", scope=local_scope
    ) as memory:
        with pytest.deprecated_call(match="HAIMEngineAdapter is deprecated"):
            with pytest.raises(
                ExactScopeError, match="does not match the AgentMemory scope"
            ):
                HAIMEngineAdapter(memory, scope=foreign_scope)
