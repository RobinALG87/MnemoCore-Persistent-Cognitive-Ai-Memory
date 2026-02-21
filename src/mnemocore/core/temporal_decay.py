"""
Adaptive Temporal Decay Module (Phase 5.0)
==========================================
Replaces the global λ decay parameter with per-memory adaptive decay
based on individual stability S_i, using the Ebbinghaus-inspired formula:

    R = e^(-T / S_i)

where:
    R  = retention score in [0, 1]
    T  = time since last access (days)
    S_i = memory stability = S_base × (1 + k × access_count)

S_i increases with every successful retrieval, making frequently-accessed
memories much more decay-resistant, while rarely-used memories decay faster.

This module provides:
  - AdaptiveDecayEngine: computes retention and updates stability
  - Background scan logic surfaced to TierManager for eviction decisions
  - Review candidate flagging (≥ REVIEW_THRESHOLD) for spaced repetition

Designed to be dependency-injected into TierManager in Phase 5.0.

Public API:
    engine = AdaptiveDecayEngine()
    retention = engine.retention(node)          # float in [0, 1]
    engine.update_after_access(node)            # call after each retrieval
    candidates = engine.scan_review_candidates(nodes, threshold=0.3)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List

from loguru import logger

if TYPE_CHECKING:
    from .node import MemoryNode


# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

# Base stability (days before first retrieval triggers significant decay)
S_BASE: float = 1.0

# Growth coefficient: how fast stability increases per access
# stability = S_base * (1 + K_GROWTH * log(1 + access_count))
K_GROWTH: float = 0.5

# Retention threshold below which a memory becomes a review candidate
REVIEW_THRESHOLD: float = 0.40

# Minimum retention before a memory is considered for eviction
EVICTION_THRESHOLD: float = 0.10


# ------------------------------------------------------------------ #
#  Core engine                                                        #
# ------------------------------------------------------------------ #

class AdaptiveDecayEngine:
    """
    Computes per-memory adaptive retention scores and manages stability growth.

    Stateless: all state lives in MemoryNode.stability and MemoryNode.access_count.
    Safe to instantiate multiple times; no shared mutable state.
    """

    def __init__(
        self,
        s_base: float = S_BASE,
        k_growth: float = K_GROWTH,
        review_threshold: float = REVIEW_THRESHOLD,
        eviction_threshold: float = EVICTION_THRESHOLD,
    ) -> None:
        self.s_base = s_base
        self.k_growth = k_growth
        self.review_threshold = review_threshold
        self.eviction_threshold = eviction_threshold

    # ---- Core math ----------------------------------------------- #

    def stability(self, node: "MemoryNode") -> float:
        """
        Compute S_i for a node.

        S_i = S_base * (1 + K_GROWTH * log(1 + access_count))

        This grows logarithmically so stability never explodes but keeps
        compounding with regular access.
        """
        ac = max(getattr(node, "access_count", 1), 1)
        return self.s_base * (1.0 + self.k_growth * math.log1p(ac))

    def retention(self, node: "MemoryNode") -> float:
        """
        Compute current retention R = e^(-T / S_i).

        T is measured in days since last access.
        Returns a value in (0, 1].
        """
        # Days since last access
        last = getattr(node, "last_accessed", None)
        if last is None:
            last = getattr(node, "created_at", datetime.now(timezone.utc))
        if getattr(last, "tzinfo", None) is None:
            last = last.replace(tzinfo=timezone.utc)

        t_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
        s_i = self.stability(node)

        r = math.exp(-t_days / max(s_i, 0.01))
        return float(r)

    # ---- Mutation helpers ---------------------------------------- #

    def update_after_access(self, node: "MemoryNode") -> None:
        """
        Increase per-memory stability after a successful retrieval.
        Writes back to node.stability.
        """
        node.stability = self.stability(node)
        node.review_candidate = False
        logger.debug(
            f"Node {getattr(node, 'id', '?')[:8]} stability updated → {node.stability:.3f}"
        )

    def update_review_candidate(self, node: "MemoryNode") -> bool:
        """
        Flag a node as review_candidate if its retention is below the threshold.
        Returns True if flagged.
        """
        r = self.retention(node)
        if r <= self.review_threshold:
            node.review_candidate = True
            logger.debug(
                f"Node {getattr(node, 'id', '?')[:8]} flagged review_candidate "
                f"(retention={r:.3f}, threshold={self.review_threshold})"
            )
            return True
        node.review_candidate = False
        return False

    def should_evict(self, node: "MemoryNode") -> bool:
        """
        True if a node's retention has fallen below the eviction threshold.
        TierManager should use this instead of the global decay_lambda.
        """
        return self.retention(node) <= self.eviction_threshold

    # ---- Batch scanning ------------------------------------------ #

    def scan_review_candidates(
        self, nodes: "List[MemoryNode]"
    ) -> "List[MemoryNode]":
        """
        Scan a list of nodes and flag those that are review candidates.
        Returns the sub-list of candidates.
        """
        candidates: List["MemoryNode"] = []
        for node in nodes:
            if self.update_review_candidate(node):
                candidates.append(node)

        logger.info(
            f"AdaptiveDecay scan: {len(candidates)}/{len(nodes)} "
            f"nodes flagged as review candidates"
        )
        return candidates

    def eviction_candidates(
        self, nodes: "List[MemoryNode]"
    ) -> "List[MemoryNode]":
        """
        Return nodes whose retention has fallen below EVICTION_THRESHOLD.
        TierManager calls this to decide which nodes to demote or remove.
        """
        return [n for n in nodes if self.should_evict(n)]


# ------------------------------------------------------------------ #
#  Module-level singleton                                             #
# ------------------------------------------------------------------ #

_ENGINE: AdaptiveDecayEngine | None = None


def get_adaptive_decay_engine() -> AdaptiveDecayEngine:
    """Return the shared AdaptiveDecayEngine singleton."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AdaptiveDecayEngine()
    return _ENGINE
