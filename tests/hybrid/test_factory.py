from __future__ import annotations

import pytest

from mnemocore.agent_memory import ClosedStoreError, MemoryScope
from mnemocore.hybrid import ExactScopeError
from mnemocore.hybrid.factory import RuntimeFactory, RuntimeMetadata


def _scope(user_id: str = "user") -> MemoryScope:
    return MemoryScope(tenant_id="tenant", user_id=user_id, agent_id="agent")


@pytest.mark.asyncio
async def test_factory_opens_an_exact_scope_runtime_and_exposes_content_free_metadata(tmp_path):
    scope = _scope()
    factory = RuntimeFactory(tmp_path / "memory.db")

    runtime = await factory.open(scope=scope)

    assert factory.metadata == RuntimeMetadata(
        scope_key=scope.scope_key,
        storage_backend="sqlite",
        runtime_kind="hybrid-memory",
    )
    assert await runtime.recall(scope, "unseen query") == ()
    with pytest.raises(ExactScopeError, match="does not match the runtime scope"):
        await runtime.recall(_scope("other-user"), "unseen query")

    await factory.close()


@pytest.mark.asyncio
async def test_factory_close_closes_its_owned_runtime(tmp_path):
    scope = _scope()
    factory = RuntimeFactory(tmp_path / "memory.db")
    runtime = await factory.open(scope=scope)

    await factory.close()
    await factory.close()

    with pytest.raises(ClosedStoreError, match="closed"):
        await runtime.recall(scope, "unseen query")

