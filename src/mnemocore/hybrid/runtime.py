"""Async and explicit sync composition for the hybrid retrieval runtime."""

from __future__ import annotations

from pathlib import Path

from mnemocore.agent_memory import AgentMemory, MemoryScope, SyncAgentMemory

from .contracts import HybridRecallResult, RetrievalRequest
from .retrieval import DeterministicHybridRetriever


class HybridMemoryRuntime:
    """An async exact-scope retrieval runtime backed solely by AgentMemory."""

    def __init__(self, memory: AgentMemory) -> None:
        if not isinstance(memory, AgentMemory):
            raise TypeError("memory must be an AgentMemory")
        self._memory = memory
        self._retriever = DeterministicHybridRetriever()

    @classmethod
    async def open(cls, path: str | Path, *, scope: MemoryScope) -> "HybridMemoryRuntime":
        """Open a runtime around a single scope-bound AgentMemory client."""
        return cls(await AgentMemory.open(path, scope=scope))

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
        records = await self._memory.list(limit=1000)
        return self._retriever.retrieve(request, records)

    async def close(self) -> None:
        await self._memory.close()

    async def __aenter__(self) -> "HybridMemoryRuntime":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class SyncHybridMemoryRuntime:
    """Explicit synchronous facade with the same deterministic retrieval contract."""

    def __init__(self, memory: SyncAgentMemory) -> None:
        if not isinstance(memory, SyncAgentMemory):
            raise TypeError("memory must be a SyncAgentMemory")
        self._memory = memory
        self._retriever = DeterministicHybridRetriever()

    @classmethod
    def open(cls, path: str | Path, *, scope: MemoryScope) -> "SyncHybridMemoryRuntime":
        return cls(SyncAgentMemory.open(path, scope=scope))

    def recall(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[HybridRecallResult, ...]:
        request = RetrievalRequest(scope=scope, query=query, limit=limit)
        records = self._memory.list(limit=1000)
        return self._retriever.retrieve(request, records)

    def close(self) -> None:
        self._memory.close()

    def __enter__(self) -> "SyncHybridMemoryRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
