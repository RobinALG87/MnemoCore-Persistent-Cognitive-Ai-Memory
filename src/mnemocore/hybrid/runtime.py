"""Async and explicit sync composition for the hybrid retrieval runtime."""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

from mnemocore.agent_memory import (
    AgentMemory,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryWrite,
    SyncAgentMemory,
)

from .contracts import (
    SCORING_VERSION,
    ExactScopeError,
    HybridRecallResult,
    RetrievalObservability,
    RetrievalRequest,
)
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


DEFAULT_CANDIDATE_BUDGET = 5_000
MAX_CANDIDATE_BUDGET = 10_000
_CANDIDATE_PAGE_SIZE = 1_000


def _validate_candidate_budget(candidate_budget: int) -> int:
    if (
        not isinstance(candidate_budget, int)
        or isinstance(candidate_budget, bool)
        or not 1 <= candidate_budget <= MAX_CANDIDATE_BUDGET
    ):
        raise ValueError(
            f"candidate_budget must be between 1 and {MAX_CANDIDATE_BUDGET}"
        )
    return candidate_budget


def _plan_writes(validated_plan: ValidatedPlan) -> tuple[MemoryWrite, ...]:
    plan = validated_plan.plan
    return tuple(
        MemoryWrite(
            content=proposal.content,
            kind=proposal.kind,
            confidence=proposal.confidence,
            metadata={
                "cognitive": {
                    "synthetic": True,
                    "plan_provenance": plan.provenance,
                    "proposal_provenance": proposal.provenance,
                    "source_memory_ids": proposal.source_memory_ids,
                }
            },
        )
        for proposal in plan.proposals
    )


def _freshly_validate(
    scope: MemoryScope, candidate: CognitivePlan | ValidatedPlan
) -> ValidatedPlan:
    if isinstance(candidate, ValidatedPlan):
        candidate = candidate.plan
    return validate_plan(
        scope,
        candidate,
        min_confidence=MINIMUM_APPLY_CONFIDENCE,
        revalidate=True,
    )


async def _resolve_plan_sources(
    memory: AgentMemory, validated_plan: ValidatedPlan
) -> None:
    """Require each cognitive source to be active in this bound exact scope."""
    resolved_ids: set[str] = set()
    for proposal in validated_plan.plan.proposals:
        for memory_id in proposal.source_memory_ids:
            if memory_id not in resolved_ids:
                await memory.get(memory_id)
                resolved_ids.add(memory_id)


def _resolve_plan_sources_sync(
    memory: SyncAgentMemory, validated_plan: ValidatedPlan
) -> None:
    """Synchronous counterpart of exact-scope cognitive source resolution."""
    resolved_ids: set[str] = set()
    for proposal in validated_plan.plan.proposals:
        for memory_id in proposal.source_memory_ids:
            if memory_id not in resolved_ids:
                memory.get(memory_id)
                resolved_ids.add(memory_id)


def _require_client_scope(memory: AgentMemory, scope: MemoryScope) -> None:
    """Reject an explicit runtime scope that is not the client's bound scope."""
    bound_scope = memory._scope
    if bound_scope != scope:
        raise ExactScopeError("runtime scope does not match the AgentMemory scope")


class HybridMemoryRuntime:
    """An async exact-scope retrieval runtime backed solely by AgentMemory."""

    def __init__(
        self,
        memory: AgentMemory,
        *,
        scope: MemoryScope,
        candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
    ) -> None:
        if not isinstance(memory, AgentMemory):
            raise TypeError("memory must be an AgentMemory")
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        _require_client_scope(memory, scope)
        self._memory = memory
        self._scope = scope
        self._candidate_budget = _validate_candidate_budget(candidate_budget)
        self._retriever = DeterministicHybridRetriever()
        self._last_retrieval_observability = RetrievalObservability(candidate_count=0)

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        scope: MemoryScope,
        candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
    ) -> "HybridMemoryRuntime":
        """Open a runtime around a single scope-bound AgentMemory client."""
        return cls(
            await AgentMemory.open(path, scope=scope),
            scope=scope,
            candidate_budget=candidate_budget,
        )

    @property
    def last_retrieval_observability(self) -> RetrievalObservability:
        """Return content-free metadata for the most recent recall."""
        return self._last_retrieval_observability

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
        records: list[MemoryRecord] = []
        offset = 0
        while len(records) < self._candidate_budget:
            page_limit = min(
                _CANDIDATE_PAGE_SIZE, self._candidate_budget - len(records)
            )
            page = await self._memory.list(limit=page_limit, offset=offset)
            records.extend(page)
            if len(page) < page_limit:
                break
            offset += len(page)
        self._last_retrieval_observability = RetrievalObservability(
            candidate_count=len(records),
            scoring_version=SCORING_VERSION,
        )
        return self._retriever.retrieve(request, records)

    async def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.OBSERVATION,
        metadata: dict | None = None,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        """Persist one record in this runtime's exact scope."""
        return await self._memory.remember(
            content, kind=kind, metadata=metadata, confidence=confidence
        )

    async def get(self, memory_id: str) -> MemoryRecord:
        """Get one active record from this runtime's exact scope."""
        return await self._memory.get(memory_id)

    async def forget(
        self, memory_id: str, *, reason: str | None = None
    ) -> MemoryRecord:
        """Forget one record in this runtime's exact scope."""
        return await self._memory.forget(memory_id, reason=reason)

    async def apply(self, plan: CognitivePlan | ValidatedPlan) -> PlanApplyReceipt:
        """Freshly validate and atomically apply a remember-only cognitive plan."""
        validated_plan = _freshly_validate(self._scope, plan)
        await _resolve_plan_sources(self._memory, validated_plan)
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

    def __init__(
        self,
        memory: SyncAgentMemory,
        *,
        scope: MemoryScope,
        candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
    ) -> None:
        if not isinstance(memory, SyncAgentMemory):
            raise TypeError("memory must be a SyncAgentMemory")
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        _require_client_scope(memory._client, scope)
        self._memory = memory
        self._scope = scope
        self._candidate_budget = _validate_candidate_budget(candidate_budget)
        self._retriever = DeterministicHybridRetriever()
        self._last_retrieval_observability = RetrievalObservability(candidate_count=0)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        scope: MemoryScope,
        candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
    ) -> "SyncHybridMemoryRuntime":
        return cls(
            SyncAgentMemory.open(path, scope=scope),
            scope=scope,
            candidate_budget=candidate_budget,
        )

    @property
    def last_retrieval_observability(self) -> RetrievalObservability:
        """Return content-free metadata for the most recent recall."""
        return self._last_retrieval_observability

    def recall(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[HybridRecallResult, ...]:
        request = RetrievalRequest(scope=scope, query=query, limit=limit)
        self._require_exact_scope(request.scope)
        records: list[MemoryRecord] = []
        offset = 0
        while len(records) < self._candidate_budget:
            page_limit = min(
                _CANDIDATE_PAGE_SIZE, self._candidate_budget - len(records)
            )
            page = self._memory.list(limit=page_limit, offset=offset)
            records.extend(page)
            if len(page) < page_limit:
                break
            offset += len(page)
        self._last_retrieval_observability = RetrievalObservability(
            candidate_count=len(records),
            scoring_version=SCORING_VERSION,
        )
        return self._retriever.retrieve(request, records)

    def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.OBSERVATION,
        metadata: dict | None = None,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        """Persist one record in this runtime's exact scope."""
        return self._memory.remember(
            content, kind=kind, metadata=metadata, confidence=confidence
        )

    def get(self, memory_id: str) -> MemoryRecord:
        """Get one active record from this runtime's exact scope."""
        return self._memory.get(memory_id)

    def forget(
        self, memory_id: str, *, reason: str | None = None
    ) -> MemoryRecord:
        """Forget one record in this runtime's exact scope."""
        return self._memory.forget(memory_id, reason=reason)

    def apply(self, plan: CognitivePlan | ValidatedPlan) -> PlanApplyReceipt:
        """Synchronously validate and atomically apply a remember-only plan."""
        validated_plan = _freshly_validate(self._scope, plan)
        _resolve_plan_sources_sync(self._memory, validated_plan)
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
