"""Async and explicit sync composition for the hybrid retrieval runtime."""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

from mnemocore.agent_memory import AgentMemory, MemoryScope, MemoryWrite, SyncAgentMemory

from .contracts import ExactScopeError, HybridRecallResult, RetrievalRequest
from .retrieval import DeterministicHybridRetriever
from .plans import (
    MINIMUM_APPLY_CONFIDENCE,
    CognitivePlan,
    ProposedMemory,
    ValidatedPlan,
    validate_plan,
)


@dataclass(frozen=True, slots=True)
class PlanApplyReceipt:
    """Content-free count-only receipt for one atomically applied plan."""

    proposal_count: int
    applied_count: int

    def __post_init__(self) -> None:
        if self.proposal_count < 1 or self.applied_count != self.proposal_count:
            raise ValueError("applied_count must equal a positive proposal_count")


def _plan_writes(validated_plan: ValidatedPlan) -> tuple[MemoryWrite, ...]:
    plan = validated_plan.plan
    return tuple(
        MemoryWrite(
            content=proposal.content,
            kind=proposal.kind,
            confidence=proposal.confidence,
        )
        for proposal in plan.proposals
    )


def _freshly_validate(scope: MemoryScope, candidate: CognitivePlan | ValidatedPlan) -> ValidatedPlan:
    if isinstance(candidate, ValidatedPlan):
        candidate = candidate.plan
    return validate_plan(
        scope,
        candidate,
        min_confidence=MINIMUM_APPLY_CONFIDENCE,
        revalidate=True,
    )


def _require_client_scope(memory: AgentMemory, scope: MemoryScope) -> None:
    """Reject an explicit runtime scope that is not the client's bound scope."""
    bound_scope = memory._scope
    if bound_scope != scope:
        raise ExactScopeError("runtime scope does not match the AgentMemory scope")


class HybridMemoryRuntime:
    """An async exact-scope retrieval runtime backed solely by AgentMemory."""

    def __init__(self, memory: AgentMemory, *, scope: MemoryScope) -> None:
        if not isinstance(memory, AgentMemory):
            raise TypeError("memory must be an AgentMemory")
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        _require_client_scope(memory, scope)
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

    async def apply(self, plan: CognitivePlan | ValidatedPlan) -> PlanApplyReceipt:
        """Freshly validate and atomically apply a remember-only cognitive plan."""
        validated_plan = _freshly_validate(self._scope, plan)
        records = await self._memory.remember_many(_plan_writes(validated_plan))
        return PlanApplyReceipt(
            proposal_count=validated_plan.proposal_count,
            applied_count=len(records),
        )

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
        _require_client_scope(memory._client, scope)
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

    def apply(self, plan: CognitivePlan | ValidatedPlan) -> PlanApplyReceipt:
        """Synchronously validate and atomically apply a remember-only plan."""
        validated_plan = _freshly_validate(self._scope, plan)
        records = self._memory.remember_many(_plan_writes(validated_plan))
        return PlanApplyReceipt(
            proposal_count=validated_plan.proposal_count,
            applied_count=len(records),
        )

    def _require_exact_scope(self, scope: MemoryScope) -> None:
        if scope != self._scope:
            raise ExactScopeError("requested scope does not match the runtime scope")

    def close(self) -> None:
        self._memory.close()

    def __enter__(self) -> "SyncHybridMemoryRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
