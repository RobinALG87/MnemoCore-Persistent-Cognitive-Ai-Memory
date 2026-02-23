"""
Tier Scoring Module (Phase 6 Refactor)
=======================================
Memory scoring and promotion decisions between tiers.

Handles:
- LTP strength calculation and recency weighting
- Similarity-based scoring for searches
- Promotion/demotion decision logic
- Temporal scoring for episodic chains
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.config import HAIMConfig

if TYPE_CHECKING:
    from mnemocore.core.node import MemoryNode


@dataclass
class TierScore:
    """Score result for a memory node."""
    node_id: str
    ltp_strength: float
    recency_score: float
    importance_score: float
    combined_score: float
    tier: str

    def __lt__(self, other):
        return self.combined_score < other.combined_score


@dataclass
class SearchResult:
    """Search result with similarity score."""
    node_id: str
    similarity: float
    tier: str
    metadata: Optional[Dict] = None


class LTPCalculator:
    """
    Long-Term Potentiation (LTP) strength calculator.

    Formula: S = I * log(1 + A) * e^(-lambda * T)

    Where:
    - I = Importance (epistemic + pragmatic values)
    - A = Access count
    - T = Time since creation (days)
    - lambda = Decay rate
    """

    def __init__(self, config: HAIMConfig):
        self.config = config

    def calculate(self, node: "MemoryNode") -> float:
        """
        Calculate LTP strength for a memory node.

        Args:
            node: The memory node to score.

        Returns:
            LTP strength value (0.0 to potentially unbounded).
        """
        import math

        # I = Importance (derived from values or default)
        importance = max(
            self.config.ltp.initial_importance,
            (node.epistemic_value + node.pragmatic_value) / 2
        )

        # A = Access count (logarithmic scaling)
        access_factor = math.log1p(node.access_count)

        # T = Time since creation (days)
        age_days = node.age_days()

        # Decay factor
        decay = math.exp(-self.config.ltp.decay_lambda * age_days)

        ltp = importance * access_factor * decay

        # Update the node's cached value
        node.ltp_strength = ltp

        return ltp

    def calculate_batch(self, nodes: List["MemoryNode"]) -> Dict[str, float]:
        """
        Calculate LTP strength for multiple nodes.

        Args:
            nodes: List of memory nodes to score.

        Returns:
            Dict mapping node_id to LTP strength.
        """
        results = {}
        for node in nodes:
            results[node.id] = self.calculate(node)
        return results


class RecencyScorer:
    """
    Recency-based scoring for memory retrieval.

    Newer memories and recently accessed memories get higher scores.
    """

    def __init__(self, config: HAIMConfig):
        self.config = config
        self.now = datetime.now(timezone.utc)

    def score(self, node: "MemoryNode") -> float:
        """
        Calculate recency score (0.0 to 1.0).

        Combines creation time and last access time.
        """
        # Time since creation (normalized to 0-1 range over 30 days)
        creation_age_hours = node.age_seconds() / 3600.0
        creation_score = max(0.0, 1.0 - creation_age_hours / (30 * 24))

        # Time since last access (normalized to 0-1 range over 7 days)
        access_age_hours = (self.now - node.last_accessed).total_seconds() / 3600.0
        access_score = max(0.0, 1.0 - access_age_hours / (7 * 24))

        # Combine: 70% last access, 30% creation
        return access_score * 0.7 + creation_score * 0.3


class ImportanceScorer:
    """
    Importance-based scoring combining multiple signals.

    Uses epistemic and pragmatic values along with access patterns.
    """

    def __init__(self, config: HAIMConfig):
        self.config = config

    def score(self, node: "MemoryNode") -> float:
        """
        Calculate importance score (0.0 to 1.0).

        Combines epistemic, pragmatic, and access-derived importance.
        """
        # Base importance from values
        base = (node.epistemic_value + node.pragmatic_value) / 2

        # Boost based on access count (logarithmic)
        import math
        access_boost = min(0.3, math.log1p(node.access_count) * 0.1)

        # Stability factor (from Phase 5.0)
        stability_factor = min(0.2, (node.stability - 1.0) * 0.1)

        return min(1.0, base + access_boost + stability_factor)


class TierScoringManager:
    """
    Manager for memory scoring and tier decisions.

    Coordinates scoring, promotion, and demotion decisions.
    """

    def __init__(self, config: HAIMConfig):
        self.config = config
        self.ltp_calculator = LTPCalculator(config)
        self.recency_scorer = RecencyScorer(config)
        self.importance_scorer = ImportanceScorer(config)

    def calculate_node_score(self, node: "MemoryNode") -> TierScore:
        """
        Calculate comprehensive score for a node.

        Returns a TierScore with LTP, recency, importance, and combined scores.
        """
        ltp = self.ltp_calculator.calculate(node)
        recency = self.recency_scorer.score(node)
        importance = self.importance_scorer.score(node)

        # Combined score: 60% LTP, 25% recency, 15% importance
        combined = ltp * 0.6 + recency * 0.25 + importance * 0.15

        return TierScore(
            node_id=node.id,
            ltp_strength=ltp,
            recency_score=recency,
            importance_score=importance,
            combined_score=combined,
            tier=node.tier,
        )

    def should_promote(self, node: "MemoryNode", from_tier: str, to_tier: str) -> bool:
        """
        Determine if a node should be promoted between tiers.

        Args:
            node: The memory node to evaluate.
            from_tier: Source tier ("warm", "cold").
            to_tier: Target tier ("hot", "warm").

        Returns:
            True if promotion is recommended.
        """
        # WARM -> HOT promotion
        if from_tier == "warm" and to_tier == "hot":
            threshold = self.config.tiers_hot.ltp_threshold_min
            delta = self.config.hysteresis.promote_delta
            return node.ltp_strength > (threshold + delta)

        # COLD -> WARM promotion
        if from_tier == "cold" and to_tier == "warm":
            threshold = self.config.tiers_warm.ltp_threshold_min
            return node.ltp_strength > threshold

        return False

    def should_demote(self, node: "MemoryNode", from_tier: str, to_tier: str) -> bool:
        """
        Determine if a node should be demoted between tiers.

        Args:
            node: The memory node to evaluate.
            from_tier: Source tier ("hot", "warm").
            to_tier: Target tier ("warm", "cold").

        Returns:
            True if demotion is recommended.
        """
        # HOT -> WARM demotion
        if from_tier == "hot" and to_tier == "warm":
            threshold = self.config.tiers_hot.ltp_threshold_min
            delta = self.config.hysteresis.demote_delta
            return node.ltp_strength < (threshold - delta)

        # WARM -> COLD demotion
        if from_tier == "warm" and to_tier == "cold":
            threshold = self.config.tiers_warm.ltp_threshold_min
            if node.ltp_strength < threshold:
                return True
            # Also check age
            archive_days = self.config.tiers_warm.archive_threshold_days
            return node.age_days() > archive_days

        return False

    def rank_for_promotion(self, nodes: List["MemoryNode"], limit: int = 10) -> List[TierScore]:
        """
        Rank nodes by promotion priority.

        Returns the top candidates for promotion to a higher tier.
        """
        scores = [self.calculate_node_score(n) for n in nodes]
        scores.sort(key=lambda s: s.combined_score, reverse=True)
        return scores[:limit]

    def rank_for_eviction(self, nodes: List["MemoryNode"], limit: int = 10) -> List[TierScore]:
        """
        Rank nodes by eviction priority.

        Returns the top candidates for eviction (lowest scores first).
        """
        scores = [self.calculate_node_score(n) for n in nodes]
        scores.sort(key=lambda s: s.combined_score)
        return scores[:limit]

    def calculate_similarity(self, query: BinaryHDV, target: BinaryHDV) -> float:
        """
        Calculate similarity between two BinaryHDV vectors.

        Returns a value between 0.0 and 1.0.
        """
        return query.similarity(target)

    def calculate_similarity_batch(
        self, query: BinaryHDV, targets: List[BinaryHDV]
    ) -> List[Tuple[int, float]]:
        """
        Calculate similarity between query and multiple targets.

        Returns list of (index, similarity) tuples.
        """
        results = []
        for i, target in enumerate(targets):
            sim = self.calculate_similarity(query, target)
            results.append((i, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class TemporalScorer:
    """
    Scorer for temporal/episodic memory relationships.

    Handles scoring based on temporal proximity and episodic chains.
    """

    def __init__(self, config: HAIMConfig):
        self.config = config

    def calculate_temporal_proximity(
        self, node_a: "MemoryNode", node_b: "MemoryNode"
    ) -> float:
        """
        Calculate temporal proximity score between two memories.

        Returns 1.0 for identical timestamps, decaying to 0.0 over time.
        """
        time_diff = abs(node_a.created_at.timestamp() - node_b.created_at.timestamp())
        # Decay over 24 hours
        return max(0.0, 1.0 - time_diff / (24 * 3600))

    def score_chain_continuity(self, chain: List["MemoryNode"]) -> float:
        """
        Score the continuity of an episodic chain.

        Higher scores indicate better temporal continuity.
        """
        if len(chain) < 2:
            return 1.0

        continuity_scores = []
        for i in range(len(chain) - 1):
            score = self.calculate_temporal_proximity(chain[i], chain[i + 1])
            continuity_scores.append(score)

        return sum(continuity_scores) / len(continuity_scores)

    def predict_next_in_chain(
        self, current: "MemoryNode", candidates: List["MemoryNode"]
    ) -> Optional["MemoryNode"]:
        """
        Predict the most likely next memory in an episodic chain.

        Uses temporal proximity and LTP strength to predict.
        """
        if not candidates:
            return None

        scored = []
        for candidate in candidates:
            # Skip if it's the same node or from before current
            if candidate.id == current.id:
                continue
            if candidate.created_at < current.created_at:
                continue

            temporal = self.calculate_temporal_proximity(current, candidate)
            ltp = min(1.0, candidate.ltp_strength)
            combined = temporal * 0.7 + ltp * 0.3
            scored.append((candidate, combined))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]


class SearchScorer:
    """
    Scorer for search results across tiers.

    Combines similarity with tier-aware scoring.
    """

    def __init__(self, config: HAIMConfig):
        self.config = config
        self.tier_scorer = TierScoringManager(config)

    def score_search_result(
        self, node_id: str, similarity: float, tier: str, ltp: float = 0.5
    ) -> SearchResult:
        """
        Create a scored search result.

        Applies tier-specific boost factors.
        """
        # HOT tier gets a boost
        tier_boost = {"hot": 1.1, "warm": 1.0, "cold": 0.9}.get(tier, 1.0)

        # Adjusted similarity
        adjusted_similarity = min(1.0, similarity * tier_boost)

        return SearchResult(
            node_id=node_id,
            similarity=adjusted_similarity,
            tier=tier,
            metadata={"ltp_strength": ltp},
        )

    def merge_and_rank(
        self,
        hot_results: List[Tuple[str, float]],
        warm_results: List[Tuple[str, float]],
        cold_results: List[Tuple[str, float]],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Merge and rank search results from all tiers.

        HOT results take precedence for duplicates.
        """
        combined: Dict[str, SearchResult] = {}

        for node_id, sim in hot_results:
            combined[node_id] = self.score_search_result(node_id, sim, "hot")

        for node_id, sim in warm_results:
            if node_id not in combined:
                combined[node_id] = self.score_search_result(node_id, sim, "warm")
            else:
                # Keep the higher score (HOT won due to boost)
                existing = combined[node_id]
                new_sim = min(1.0, sim * 1.0)  # No boost for WARM
                if new_sim > existing.similarity:
                    combined[node_id] = self.score_search_result(
                        node_id, sim, "warm"
                    )

        for node_id, sim in cold_results:
            if node_id not in combined:
                combined[node_id] = self.score_search_result(
                    node_id, sim, "cold"
                )

        # Sort by similarity
        results = list(combined.values())
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]


__all__ = [
    "TierScore",
    "SearchResult",
    "LTPCalculator",
    "RecencyScorer",
    "ImportanceScorer",
    "TierScoringManager",
    "TemporalScorer",
    "SearchScorer",
]
