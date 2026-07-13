"""Async and explicit sync composition for the hybrid retrieval runtime."""

from __future__ import annotations

from pathlib import Path

from mnemocore.agent_memory import AgentMemory, MemoryScope, SyncAgentMemory

from .contracts import ExactScopeError, HybridRecallResult, RetrievalRequest
from .retrieval import DeterministicHybridRetriever


class HybridMemoryRuntime:
    """An async exact-scope retrieval runtime backed solely by AgentMemory."""

    def __init__(self, memory: AgentMemory, *, scope: MemoryScope) -> None:
        if not isinstance(memory, AgentMemory):
            raise TypeError("memory must be an AgentMemory")
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        self._memory = memory
        self._scope = scope
        self._retriever = DeterministicHybridRetriever()

    @classmethod
    async def open(cls, path: str | Path, *, scope: MemoryScope) -> "HybridMemoryRuntime":
        """Open a runtime around a single scope-bound AgentMemory client."""
        return cls(await AgentMemory.open(path, scope=scope), scope=scope)

    async def recall(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[HybridRecallResult, ...]:
        """Retrieve only records whose scope is exactly ``scope``.

        AgentMemory remains the only durable source; this layer holds no index
        or persisted projection and deliberately requests no ancestor scope.
        """
        request = RetrievalRequest(scope=scope, query=query, limit=limit)
        self._require_exact_scope(request.scope)
        records = await self._memory.list(limit=1000)
        return self._retriever.retrieve(request, records)

    def _require_exact_scope(self, scope: MemoryScope) -> None:
        if scope != self._scope:
            raise ExactScopeError("requested scope does not match the runtime scope")

    async def close(self) -> None:
        await self._memory.close()

    async def __aenter__(self) -> "HybridMemoryRuntime":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class SyncHybridMemoryRuntime:
    """Explicit synchronous facade with the same deterministic retrieval contract."""

    def __init__(self, memory: SyncAgentMemory, *, scope: MemoryScope) -> None:
        if not isinstance(memory, SyncAgentMemory):
            raise TypeError("memory must be a SyncAgentMemory")
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        self._memory = memory
        self._scope = scope
        self._retriever = DeterministicHybridRetriever()

    @classmethod
    def open(cls, path: str | Path, *, scope: MemoryScope) -> "SyncHybridMemoryRuntime":
        return cls(SyncAgentMemory.open(path, scope=scope), scope=scope)

    def recall(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[HybridRecallResult, ...]:
        request = RetrievalRequest(scope=scope, query=query, limit=limit)
        self._require_exact_scope(request.scope)
        records = self._memory.list(limit=1000)
        return self._retriever.retrieve(request, records)

    def _require_exact_scope(self, scope: MemoryScope) -> None:
        if scope != self._scope:
            raise ExactScopeError("requested scope does not match the runtime scope")

    def close(self) -> None:
        self._memory.close()

    def __enter__(self) -> "SyncHybridMemoryRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
