"""Ephemeral deterministic projections rebuilt from exact-scope memory records."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from mnemocore.agent_memory import MemoryKind, MemoryRecord, MemoryScope

from .contracts import ExactScopeError


class MemoryTier(str, Enum):
    """A stable, content-independent tier assignment for memory kinds."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


_TIER_BY_KIND = {
    MemoryKind.DECISION: MemoryTier.HOT,
    MemoryKind.PROCEDURE: MemoryTier.HOT,
    MemoryKind.FACT: MemoryTier.WARM,
    MemoryKind.PREFERENCE: MemoryTier.WARM,
    MemoryKind.SUMMARY: MemoryTier.WARM,
    MemoryKind.OBSERVATION: MemoryTier.COLD,
    MemoryKind.EPISODE: MemoryTier.COLD,
}


@dataclass(frozen=True, slots=True)
class TierProjection:
    """An in-memory tier view whose contents can always be rebuilt."""

    scope: MemoryScope
    tiers: tuple[tuple[MemoryTier, tuple[str, ...]], ...]

    def memory_ids(self, tier: MemoryTier) -> tuple[str, ...]:
        if not isinstance(tier, MemoryTier):
            raise TypeError("tier must be a MemoryTier")
        return dict(self.tiers)[tier]


@dataclass(frozen=True, slots=True)
class GraphProjection:
    """An in-memory relation graph derived from record-owned metadata."""

    scope: MemoryScope
    node_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def _exact_scope_records(scope: MemoryScope, records: Iterable[MemoryRecord]) -> tuple[MemoryRecord, ...]:
    if not isinstance(scope, MemoryScope):
        raise TypeError("scope must be a MemoryScope")
    materialized = tuple(records)
    for record in materialized:
        if not isinstance(record, MemoryRecord):
            raise TypeError("records must contain only MemoryRecord")
        if record.scope != scope:
            raise ExactScopeError("record scope does not match the projection scope")
    return materialized


def rebuild_tier_projection(scope: MemoryScope, records: Iterable[MemoryRecord]) -> TierProjection:
    """Build a stable tier projection without persisting any duplicate state."""
    by_tier: dict[MemoryTier, list[str]] = {tier: [] for tier in MemoryTier}
    for record in _exact_scope_records(scope, records):
        by_tier[_TIER_BY_KIND[record.kind]].append(record.id)
    return TierProjection(
        scope=scope,
        tiers=tuple((tier, tuple(sorted(by_tier[tier]))) for tier in MemoryTier),
    )


def rebuild_graph_projection(scope: MemoryScope, records: Iterable[MemoryRecord]) -> GraphProjection:
    """Build a stable graph using only ``related_memory_ids`` in records' metadata."""
    exact_records = _exact_scope_records(scope, records)
    node_ids = tuple(sorted(record.id for record in exact_records))
    nodes = frozenset(node_ids)
    edges: set[tuple[str, str]] = set()
    for record in exact_records:
        related = record.metadata.get("related_memory_ids", ())
        if not isinstance(related, (tuple, list)):
            raise ValueError("related_memory_ids must be a sequence of memory ids")
        for target_id in related:
            if not isinstance(target_id, str) or not target_id:
                raise ValueError("related_memory_ids must contain non-empty strings")
            if target_id in nodes:
                edges.add((record.id, target_id))
    return GraphProjection(scope=scope, node_ids=node_ids, edges=tuple(sorted(edges)))
