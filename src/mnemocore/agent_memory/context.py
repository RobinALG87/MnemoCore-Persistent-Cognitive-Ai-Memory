"""Deterministic, dependency-free context compilation for agent memory."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .errors import ValidationError
from .models import (
    ContextItem,
    ContextPack,
    MemoryKind,
    MemoryReceipt,
    RecallResult,
)

CONTEXT_LEVELS: tuple[tuple[str, tuple[MemoryKind, ...]], ...] = (
    ("core", (MemoryKind.PREFERENCE,)),
    ("working", (MemoryKind.OBSERVATION, MemoryKind.DECISION)),
    ("procedural", (MemoryKind.PROCEDURE,)),
    ("episodic", (MemoryKind.EPISODE,)),
    ("semantic", (MemoryKind.FACT, MemoryKind.SUMMARY)),
)


def estimate_tokens(content: str) -> int:
    """Return a deterministic conservative local estimate without a tokenizer."""
    return max(1, math.ceil(len(content) / 4))


def compile_context_pack(
    query: str,
    *,
    token_budget: int,
    results_by_level: Mapping[str, Sequence[RecallResult]],
) -> ContextPack:
    """Select explainable recall results into a fixed-priority bounded briefing."""
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("query must not be blank")
    if (
        not isinstance(token_budget, int)
        or isinstance(token_budget, bool)
        or token_budget < 1
    ):
        raise ValidationError("token_budget must be a positive integer")

    remaining = token_budget
    selected: dict[str, list[ContextItem]] = {
        level: [] for level, _ in CONTEXT_LEVELS
    }
    seen_memory_ids: set[str] = set()
    for level, _ in CONTEXT_LEVELS:
        for result in results_by_level.get(level, ()):
            memory = result.memory
            if memory.id in seen_memory_ids:
                continue
            token_count = estimate_tokens(memory.content)
            if token_count > remaining:
                continue
            selected[level].append(
                ContextItem(
                    content=memory.content,
                    receipt=MemoryReceipt(
                        memory_id=memory.id,
                        scope=memory.scope,
                        kind=memory.kind,
                        score=result.score,
                        score_components=result.score_components,
                        reason=result.reason,
                        evidence_ids=result.evidence_ids,
                        estimated_tokens=token_count,
                    ),
                )
            )
            seen_memory_ids.add(memory.id)
            remaining -= token_count

    return ContextPack(
        query=query,
        token_budget=token_budget,
        estimated_tokens=token_budget - remaining,
        core=tuple(selected["core"]),
        working=tuple(selected["working"]),
        episodic=tuple(selected["episodic"]),
        semantic=tuple(selected["semantic"]),
        procedural=tuple(selected["procedural"]),
    )
