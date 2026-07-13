from __future__ import annotations

import pytest

from mnemocore.agent_memory import MemoryKind, MemoryScope
from mnemocore.hybrid.plans import (
    CognitivePlan,
    ProposedMemory,
    ValidatedPlan,
    validate_plan,
)


def _scope(user_id: str) -> MemoryScope:
    return MemoryScope(tenant_id="tenant", user_id=user_id, agent_id="agent")


def test_plan_validation_returns_a_typed_non_mutating_runtime_receipt():
    scope = _scope("local")
    plan = CognitivePlan(
        scope=scope,
        provenance="cognitive-module",
        confidence=0.8,
        proposals=(
            ProposedMemory(
                content="a plan-derived observation",
                kind=MemoryKind.OBSERVATION,
                provenance="cognitive-module",
                confidence=0.7,
            ),
        ),
    )

    receipt = validate_plan(scope, plan)

    assert receipt.scope == scope
    assert receipt.plan is plan
    assert receipt.proposal_count == 1


def test_plan_validation_rejects_cross_scope_and_invalid_provenance_or_confidence():
    local_scope = _scope("local")
    foreign_scope = _scope("foreign")
    valid_proposal = ProposedMemory(
        content="proposed memory",
        kind=MemoryKind.FACT,
        provenance="cognitive-module",
        confidence=0.5,
    )
    foreign_plan = CognitivePlan(
        scope=foreign_scope,
        provenance="cognitive-module",
        confidence=0.5,
        proposals=(valid_proposal,),
    )

    with pytest.raises(ValueError, match="does not match the runtime scope"):
        validate_plan(local_scope, foreign_plan)
    with pytest.raises(ValueError, match="provenance must not be blank"):
        CognitivePlan(
            scope=local_scope,
            provenance=" ",
            confidence=0.5,
            proposals=(valid_proposal,),
        )
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        ProposedMemory(
            content="invalid confidence",
            kind=MemoryKind.FACT,
            provenance="cognitive-module",
            confidence=1.1,
        )


def test_validated_plan_rejects_direct_construction_with_a_different_scope():
    scope = _scope("local")
    plan = CognitivePlan(
        scope=scope,
        provenance="cognitive-module",
        confidence=0.5,
        proposals=(
            ProposedMemory("memory", MemoryKind.FACT, "cognitive-module", 0.5),
        ),
    )

    with pytest.raises(ValueError, match="does not match the plan scope"):
        ValidatedPlan(scope=_scope("foreign"), plan=plan, proposal_count=1)


def test_validated_plan_rejects_direct_construction_with_an_incorrect_count():
    scope = _scope("local")
    plan = CognitivePlan(
        scope=scope,
        provenance="cognitive-module",
        confidence=0.5,
        proposals=(
            ProposedMemory("memory", MemoryKind.FACT, "cognitive-module", 0.5),
        ),
    )

    with pytest.raises(ValueError, match="proposal_count must match the plan proposals"):
        ValidatedPlan(scope=scope, plan=plan, proposal_count=0)
