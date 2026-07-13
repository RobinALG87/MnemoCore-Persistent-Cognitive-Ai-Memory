"""Public, content-free contracts for deterministic hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from mnemocore.agent_memory import MemoryRecord, MemoryScope


SCORING_VERSION = "hybrid-lexical-binary-hdv-v1"


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """An exact-scope retrieval request.

    A scope is deliberately mandatory: the hybrid runtime never broadens a
    request to a parent, tenant, or global scope.
    """

    scope: MemoryScope
    query: str
    limit: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must not be blank")
        if not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit < 1:
            raise ValueError("limit must be a positive integer")


@dataclass(frozen=True, slots=True)
class HybridRecallResult:
    """A deterministic retrieval result with inspectable, content-free scores."""

    memory: MemoryRecord
    lexical_score: float
    hdv_score: float
    score: float
    scoring_version: str = SCORING_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.memory, MemoryRecord):
            raise TypeError("memory must be a MemoryRecord")
        for name in ("lexical_score", "hdv_score", "score"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.scoring_version != SCORING_VERSION:
            raise ValueError("scoring_version must identify the current scoring algorithm")

    @property
    def score_components(self) -> Mapping[str, float]:
        """Return stable component names without exposing any content or signals."""
        return MappingProxyType(
            {
                "lexical": float(self.lexical_score),
                "hdv": float(self.hdv_score),
                "hybrid": float(self.score),
            }
        )
