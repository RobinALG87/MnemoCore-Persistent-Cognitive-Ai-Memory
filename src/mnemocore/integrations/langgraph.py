"""LangGraph-compatible async node adapters without a LangGraph dependency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .bridge import AgentMemoryBridge, IntegrationError


class LangGraphMemory:
    """Add memory context to a LangGraph state through ordinary async nodes."""

    def __init__(
        self,
        bridge: AgentMemoryBridge,
        *,
        query_key: str = "goal",
        output_key: str = "memory_context",
    ) -> None:
        self.bridge = bridge
        self.query_key = query_key
        self.output_key = output_key

    async def context_node(self, state: Mapping[str, Any]) -> dict[str, Any]:
        query = state.get(self.query_key) or state.get("query")
        if not isinstance(query, str) or not query.strip():
            raise IntegrationError(f"LangGraph state needs a non-empty {self.query_key!r}")
        context = await self.bridge.context(query)
        return {self.output_key: context}

    async def observation_node(self, state: Mapping[str, Any]) -> dict[str, Any]:
        observation = state.get("observation")
        if not isinstance(observation, str) or not observation.strip():
            raise IntegrationError("LangGraph state needs a non-empty 'observation'")
        stored = await self.bridge.observe(observation)
        return {"memory_observation": stored}
