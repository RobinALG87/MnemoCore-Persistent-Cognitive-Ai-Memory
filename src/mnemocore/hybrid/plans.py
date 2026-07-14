"""Typed, non-mutating validation for cognitive memory proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mnemocore.agent_memory import MemoryKind, MemoryScope

from .contracts import ExactScopeError

MINIMUM_APPLY_CONFIDENCE = 0.5


def _require_provenance(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("provenance must not be blank")
    return value.strip()


def _require_confidence(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("confidence must be between 0 and 1")
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return float(value)


@dataclass(frozen=True, slots=True)
class ProposedMemory:
    """An ephemeral cognitive proposal; it has no persistence behavior."""

    content: str
    kind: MemoryKind
    provenance: str
    confidence: float
    source_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must not be blank")
        if not isinstance(self.kind, MemoryKind):
            raise TypeError("kind must be a MemoryKind")
        object.__setattr__(self, "provenance", _require_provenance(self.provenance))
        object.__setattr__(self, "confidence", _require_confidence(self.confidence))
        if not isinstance(self.source_memory_ids, tuple):
            raise TypeError("source_memory_ids must be a tuple")
        object.__setattr__(
            self,
            "source_memory_ids",
            tuple(
                _require_provenance(memory_id) for memory_id in self.source_memory_ids
            ),
        )


@dataclass(frozen=True, slots=True)
class CognitivePlan:
    """A scope-bound proposal created by a cognitive module, never a writer."""

    scope: MemoryScope
    provenance: str
    confidence: float
    proposals: tuple[ProposedMemory, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        object.__setattr__(self, "provenance", _require_provenance(self.provenance))
        object.__setattr__(self, "confidence", _require_confidence(self.confidence))
        if not isinstance(self.proposals, tuple) or not self.proposals:
            raise ValueError("proposals must be a non-empty tuple")
        if any(not isinstance(proposal, ProposedMemory) for proposal in self.proposals):
            raise TypeError("proposals must contain only ProposedMemory")


@dataclass(frozen=True, slots=True)
class ValidatedPlan:
    """A receipt for runtime use; validation deliberately performs no writes."""

    scope: MemoryScope
    plan: CognitivePlan
    proposal_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        if not isinstance(self.plan, CognitivePlan):
            raise TypeError("plan must be a CognitivePlan")
        if self.scope != self.plan.scope:
            raise ValueError("scope does not match the plan scope")
        if self.proposal_count != len(self.plan.proposals):
            raise ValueError("proposal_count must match the plan proposals")


def validate_plan(
    scope: MemoryScope,
    plan: CognitivePlan,
    *,
    min_confidence: float = 0.0,
    revalidate: bool = False,
) -> ValidatedPlan:
    """Validate a plan for one runtime scope without applying or persisting it."""
    if not isinstance(scope, MemoryScope):
        raise TypeError("scope must be a MemoryScope")
    if not isinstance(plan, CognitivePlan):
        raise TypeError("plan must be a CognitivePlan")
    minimum = _require_confidence(min_confidence)
    if plan.scope != scope:
        raise ExactScopeError("plan scope does not match the runtime scope")
    validated_plan = plan
    if revalidate:
        # Rebuild from public values so a caller cannot bypass dataclass
        # validation by mutating a frozen object through low-level mechanisms.
        validated_plan = CognitivePlan(
            scope=plan.scope,
            provenance=plan.provenance,
            confidence=plan.confidence,
            proposals=tuple(
                ProposedMemory(
                    content=proposal.content,
                    kind=proposal.kind,
                    provenance=proposal.provenance,
                    confidence=proposal.confidence,
                    source_memory_ids=proposal.source_memory_ids,
                )
                for proposal in plan.proposals
            ),
        )
    if validated_plan.confidence < minimum or any(
        proposal.confidence < minimum for proposal in validated_plan.proposals
    ):
        raise ValueError("plan and proposals must meet the minimum confidence")
    return ValidatedPlan(
        scope=scope,
        plan=validated_plan,
        proposal_count=len(validated_plan.proposals),
    )
