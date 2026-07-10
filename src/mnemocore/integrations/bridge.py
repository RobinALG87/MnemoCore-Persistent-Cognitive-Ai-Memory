"""Shared, policy-controlled bridge used by framework adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..agent_memory import AgentMemory, MemoryKind, MemorySession, ValidationError


class IntegrationError(ValidationError):
    """Raised when an adapter request violates its local integration policy."""


@dataclass(frozen=True, slots=True)
class IntegrationPolicy:
    """Small guardrail set shared by all adapters."""

    max_context_tokens: int = 1200
    include_ancestors: bool = True
    allow_writes: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_context_tokens, int)
            or isinstance(self.max_context_tokens, bool)
            or not 1 <= self.max_context_tokens <= 100_000
        ):
            raise IntegrationError("max_context_tokens must be between 1 and 100000")


class AgentMemoryBridge:
    """Framework-neutral async facade with one enforced integration policy."""

    def __init__(
        self,
        memory: AgentMemory,
        *,
        policy: Optional[IntegrationPolicy] = None,
    ) -> None:
        self.memory = memory
        self.policy = policy or IntegrationPolicy()

    async def context(
        self,
        query: str,
        *,
        token_budget: Optional[int] = None,
        include_ancestors: Optional[bool] = None,
    ) -> Any:
        budget = self.policy.max_context_tokens if token_budget is None else token_budget
        if not isinstance(budget, int) or isinstance(budget, bool):
            raise IntegrationError("token_budget must be an integer")
        if not 1 <= budget <= self.policy.max_context_tokens:
            raise IntegrationError(
                f"token_budget must be between 1 and {self.policy.max_context_tokens}"
            )
        return await self.memory.compile_context(
            query,
            token_budget=budget,
            include_ancestors=(
                self.policy.include_ancestors
                if include_ancestors is None
                else include_ancestors
            ),
        )

    async def remember(
        self,
        content: str,
        *,
        kind: MemoryKind | str = MemoryKind.OBSERVATION,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        self._ensure_writes()
        normalized_kind = _normalize_kind(kind)
        kwargs: dict[str, Any] = {"kind": normalized_kind}
        if metadata is not None:
            kwargs["metadata"] = metadata
        return await self.memory.remember(content, **kwargs)

    async def observe(self, content: str, *, metadata: Optional[dict[str, Any]] = None) -> Any:
        return await self.remember(content, kind=MemoryKind.OBSERVATION, metadata=metadata)

    async def finish(
        self,
        session: MemorySession,
        *,
        outcome: str = "success",
        reward: float = 0.0,
        notes: Optional[str] = None,
    ) -> Any:
        self._ensure_writes()
        return await session.finish(outcome=outcome, reward=reward, notes=notes)

    def _ensure_writes(self) -> None:
        if not self.policy.allow_writes:
            raise IntegrationError("memory writes are disabled by integration policy")


def _normalize_kind(kind: MemoryKind | str) -> MemoryKind:
    try:
        return kind if isinstance(kind, MemoryKind) else MemoryKind(kind)
    except (TypeError, ValueError) as error:
        raise IntegrationError(f"unknown memory kind: {kind!r}") from error
