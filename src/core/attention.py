"""
Contextual Query Masking via XOR Attention (Phase 4.0)
======================================================
Implements an XOR-based soft attention mechanism over Binary HDV space.

How it works:
  1. A "context key" is constructed by bundling recent HOT-tier vectors.
  2. A XOR attention mask is generated:  mask = query XOR context_key
     This creates a residual vector that is ORTHOGONAL to the context,
     effectively suppressing already-known dimensions and amplifying novel ones.
  3. Query results are re-ranked by a composite score:
        composite = alpha * raw_similarity + beta * novelty_boost(mask, mem_hdv)
  4. The mask is also available for downstream gap-detection.

Motivation (VSA theory):
  - XOR in binary HDV space is the self-inverse binding operator.
  - query.xor(context) ≈ "what about this query is NOT already represented in context?"
  - Hamming similarity(mask, candidate) ≈ novelty of candidate relative to context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .binary_hdv import BinaryHDV, majority_bundle

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Tunable hyperparameters for XOR attention."""
    alpha: float = 0.6          # Weight for raw similarity
    beta: float = 0.4           # Weight for novelty score from XOR mask
    context_sample_n: int = 50  # How many HOT nodes to include in context key
    min_novelty_boost: float = 0.0  # Floor for novelty contribution
    enabled: bool = True

    def validate(self) -> None:
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0, 1]"
        assert 0.0 <= self.beta <= 1.0, "beta must be in [0, 1]"
        assert abs((self.alpha + self.beta) - 1.0) < 1e-6, "alpha + beta must equal 1.0"


@dataclass
class AttentionResult:
    """Enriched result from contextual reranking."""
    node_id: str
    raw_score: float
    novelty_score: float
    composite_score: float
    attention_mask: Optional[BinaryHDV] = field(default=None, repr=False)


class XORAttentionMasker:
    """
    Contextual query masking using XOR binding in binary HDV space.

    Usage:
        masker = XORAttentionMasker(config)
        mask = masker.build_attention_mask(query_vec, context_vecs)
        reranked = masker.rerank(raw_scores, memory_vectors, mask)
    """

    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()

    def build_context_key(self, context_nodes_hdv: List[BinaryHDV]) -> BinaryHDV:
        """
        Bundle HOT-tier vectors into a single context summary key.
        Uses majority vote bundling (sum > threshold → 1, else → 0).
        Falls back to zero-vector if no context is available.
        """
        if not context_nodes_hdv:
            return BinaryHDV.zeros(context_nodes_hdv[0].dimension if context_nodes_hdv else 16384)

        return majority_bundle(context_nodes_hdv)

    def build_attention_mask(
        self,
        query_vec: BinaryHDV,
        context_key: BinaryHDV,
    ) -> BinaryHDV:
        """
        Compute XOR attention mask: mask = query XOR context_key.

        The mask represents "query minus context" — bits that are unique
        to the query compared to what the system already holds in working memory.

        High Hamming similarity between mask and a candidate → that candidate
        is novel / peripheral relative to the current context.
        """
        mask = query_vec.xor_bind(context_key)
        logger.debug(
            "Built XOR attention mask — "
            f"query/context Hamming dist = {query_vec.normalized_distance(context_key):.4f}"
        )
        return mask

    def novelty_score(self, mask: BinaryHDV, candidate_hdv: BinaryHDV) -> float:
        """
        Calculate novelty of a candidate relative to the context.

        Defined as: Hamming similarity(mask, candidate) in [0, 1].
        Higher value → candidate is more "attention-worthy" given the query context.
        """
        return mask.similarity(candidate_hdv)

    def rerank(
        self,
        raw_scores: Dict[str, float],
        memory_vectors: Dict[str, BinaryHDV],
        mask: BinaryHDV,
    ) -> List[AttentionResult]:
        """
        Re-rank retrieved memories using the composite XOR attention score.

        Args:
            raw_scores: {node_id: raw_similarity} from initial retrieval.
            memory_vectors: {node_id: BinaryHDV} for novelty calculation.
            mask: XOR attention mask built from query and context.

        Returns:
            Sorted list of AttentionResult (highest composite first).
        """
        cfg = self.config
        results: List[AttentionResult] = []

        for node_id, raw in raw_scores.items():
            hdv = memory_vectors.get(node_id)
            if hdv is None:
                novelty = cfg.min_novelty_boost
            else:
                novelty = max(self.novelty_score(mask, hdv), cfg.min_novelty_boost)

            composite = cfg.alpha * raw + cfg.beta * novelty

            results.append(
                AttentionResult(
                    node_id=node_id,
                    raw_score=raw,
                    novelty_score=novelty,
                    composite_score=composite,
                    attention_mask=mask,
                )
            )

        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    def extract_scores(
        self, results: List[AttentionResult]
    ) -> List[Tuple[str, float]]:
        """Convert AttentionResult list to the standard (node_id, score) tuple format."""
        return [(r.node_id, r.composite_score) for r in results]
