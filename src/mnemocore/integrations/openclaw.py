"""OpenClaw-style turn hooks using a small JSON-compatible event contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .bridge import AgentMemoryBridge, IntegrationError


class OpenClawMemory:
    """Translate OpenClaw task/turn payloads into safe memory operations."""

    def __init__(self, bridge: AgentMemoryBridge) -> None:
        self.bridge = bridge

    async def before_turn(self, event: Mapping[str, Any]) -> dict[str, Any]:
        query = event.get("goal") or event.get("query")
        if not isinstance(query, str) or not query.strip():
            raise IntegrationError("OpenClaw event needs a non-empty 'goal' or 'query'")
        context = await self.bridge.context(query)
        return {"memory_context": context}

    async def after_turn(self, event: Mapping[str, Any]) -> dict[str, Any]:
        observation = event.get("observation")
        if observation is None:
            return {}
        if not isinstance(observation, str) or not observation.strip():
            raise IntegrationError("OpenClaw observation must be a non-empty string")
        return {"memory_observation": await self.bridge.observe(observation)}
