"""CrewAI-compatible synchronous tools with optional lazy CrewAI decoration."""

from __future__ import annotations

from typing import Any, Callable

from ..agent_memory import MemoryKind, SyncAgentMemory
from .bridge import IntegrationError, IntegrationPolicy, _normalize_kind


class CrewAIMemoryTools:
    """Expose bounded recall and controlled remember tools to a CrewAI crew."""

    def __init__(
        self,
        memory: SyncAgentMemory,
        *,
        policy: IntegrationPolicy | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self.memory = memory
        if policy is not None and max_context_tokens is not None:
            raise IntegrationError(
                "pass either policy or max_context_tokens, not both"
            )
        self.policy = policy or IntegrationPolicy(
            max_context_tokens=(
                1200 if max_context_tokens is None else max_context_tokens
            )
        )

    def recall(self, query: str) -> Any:
        return self.memory.compile_context(
            query,
            token_budget=self.policy.max_context_tokens,
            include_ancestors=self.policy.include_ancestors,
        )

    def remember(self, content: str, *, kind: MemoryKind | str = MemoryKind.OBSERVATION) -> Any:
        if not self.policy.allow_writes:
            raise IntegrationError("memory writes are disabled by integration policy")
        return self.memory.remember(content, kind=_normalize_kind(kind))

    def as_tools(self) -> tuple[Callable[..., Any], ...]:
        """Return plain callables; decorate them with CrewAI only when installed."""
        try:
            from crewai.tools import tool
        except ImportError:
            return (self.recall, self.remember)

        @tool("mnemocore_recall")
        def crew_recall(query: str) -> Any:
            return self.recall(query)

        @tool("mnemocore_remember")
        def crew_remember(content: str, kind: str = "observation") -> Any:
            return self.remember(content, kind=kind)

        return crew_recall, crew_remember
