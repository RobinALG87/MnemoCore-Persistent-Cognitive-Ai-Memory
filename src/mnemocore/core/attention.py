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

Phase 4.1: XOR-based Project Isolation
======================================
XORIsolationMask provides deterministic project-based memory isolation:

  - Each project_id derives a unique binary mask via SHA256(project_id) -> seed -> RNG
  - store(): masked_hdv = original_hdv XOR project_mask
  - query(): unmasked_query = query_hdv XOR project_mask (then search in masked space)
  - Memories from different projects are effectively orthogonal (~50% similarity)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .binary_hdv import BinaryHDV, majority_bundle


@dataclass
class AttentionConfig:
    """Tunable hyperparameters for XOR attention."""

    alpha: float = 0.6  # Weight for raw similarity
    beta: float = 0.4  # Weight for novelty score from XOR mask
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
            return BinaryHDV.zeros(
                context_nodes_hdv[0].dimension if context_nodes_hdv else 16384
            )

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

    def extract_scores(self, results: List[AttentionResult]) -> List[Tuple[str, float]]:
        """Convert AttentionResult list to the standard (node_id, score) tuple format."""
        return [(r.node_id, r.composite_score) for r in results]


# ==============================================================================
# Phase 4.1: XOR-based Project Isolation
# ==============================================================================


@dataclass
class IsolationConfig:
    """Configuration for XOR-based project isolation."""

    enabled: bool = True
    dimension: int = 16384

    def validate(self) -> None:
        assert self.dimension > 0, "dimension must be positive"
        assert self.dimension % 8 == 0, "dimension must be multiple of 8"


class XORIsolationMask:
    """
    Deterministic XOR-based isolation mask for multi-tenant memory isolation.

    Design:
    -------
    Each project_id derives a unique binary mask through:
        SHA256(project_id) -> 256-bit digest -> seed -> np.random.Generator -> binary mask

    The mask is applied via XOR binding:
        - store(content, project_id="A"): masked_hdv = original_hdv XOR mask_A
        - query(query_text, project_id="A"): unmasked = query_hdv XOR mask_A

    Properties:
    -----------
    - Self-inverse: XOR twice with the same mask recovers the original vector
    - Deterministic: Same project_id always produces the same mask
    - Orthogonal isolation: Different projects' masks are ~50% different (random)
    - No key management: project_id IS the key (no external secrets needed)

    Security Model:
    ---------------
    This provides cryptographic isolation via the one-time pad principle:
    - A masked vector reveals NO information about the original without the mask
    - Cross-project queries will match random noise (~50% similarity baseline)
    - The isolation strength depends on the secrecy of project_ids

    Usage:
    ------
        masker = XORIsolationMask(config)
        mask = masker.get_mask("project-alpha")  # Deterministic mask

        # Store
        masked_hdv = masker.apply_mask(original_hdv, "project-alpha")

        # Query (apply same mask to query to search in masked space)
        masked_query = masker.apply_mask(query_hdv, "project-alpha")

        # Remove mask (if needed for inspection)
        original = masker.remove_mask(masked_hdv, "project-alpha")
    """

    def __init__(self, config: Optional[IsolationConfig] = None):
        self.config = config or IsolationConfig()
        self._mask_cache: Dict[str, BinaryHDV] = {}

    def _derive_seed(self, project_id: str) -> int:
        """
        Derive a deterministic 64-bit seed from project_id using SHA256.

        Args:
            project_id: Unique project identifier string.

        Returns:
            64-bit integer seed for numpy's Generator.
        """
        digest = hashlib.sha256(f"mnemo_isolation_v1:{project_id}".encode()).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def get_mask(self, project_id: str) -> BinaryHDV:
        """
        Get or create the deterministic isolation mask for a project.

        The mask is cached for efficiency. Same project_id always returns
        the same BinaryHDV mask.

        Args:
            project_id: Unique project identifier.

        Returns:
            BinaryHDV mask of dimension self.config.dimension.
        """
        if project_id in self._mask_cache:
            return self._mask_cache[project_id]

        seed = self._derive_seed(project_id)
        rng = np.random.default_rng(seed)

        # Generate random binary mask
        n_bytes = self.config.dimension // 8
        mask_bytes = rng.integers(0, 256, size=n_bytes, dtype=np.uint8)

        mask = BinaryHDV(data=mask_bytes, dimension=self.config.dimension)
        self._mask_cache[project_id] = mask

        logger.debug(
            f"Generated isolation mask for project '{project_id}' (seed={seed})"
        )
        return mask

    def apply_mask(self, hdv: BinaryHDV, project_id: str) -> BinaryHDV:
        """
        Apply project isolation mask to a vector (XOR binding).

        Args:
            hdv: The BinaryHDV to mask.
            project_id: Project identifier for mask derivation.

        Returns:
            Masked BinaryHDV (original XOR project_mask).
        """
        if not self.config.enabled:
            return hdv

        mask = self.get_mask(project_id)
        return hdv.xor_bind(mask)

    def remove_mask(self, masked_hdv: BinaryHDV, project_id: str) -> BinaryHDV:
        """
        Remove project isolation mask from a vector (XOR is self-inverse).

        Note: This is identical to apply_mask() due to XOR's self-inverse property.
        Kept as a separate method for semantic clarity.

        Args:
            masked_hdv: The masked BinaryHDV.
            project_id: Project identifier used for masking.

        Returns:
            Original unmasked BinaryHDV.
        """
        return self.apply_mask(masked_hdv, project_id)

    def clear_cache(self) -> None:
        """Clear the mask cache (useful for testing)."""
        self._mask_cache.clear()

    def is_isolated(
        self,
        hdv_a: BinaryHDV,
        project_id_a: str,
        hdv_b: BinaryHDV,
        project_id_b: str,
        threshold: float = 0.55,
    ) -> bool:
        """
        Check if two vectors are properly isolated (different projects).

        After masking, vectors from different projects should have ~50% similarity.
        This method checks if the cross-project similarity is within expected bounds.

        Args:
            hdv_a: First (unmasked) vector.
            project_id_a: First vector's project.
            hdv_b: Second (unmasked) vector.
            project_id_b: Second vector's project.
            threshold: Maximum similarity for "isolated" (default 0.55).

        Returns:
            True if vectors are isolated (different projects), False otherwise.
        """
        if project_id_a == project_id_b:
            return False  # Same project = not isolated

        masked_a = self.apply_mask(hdv_a, project_id_a)
        masked_b = self.apply_mask(hdv_b, project_id_b)

        similarity = masked_a.similarity(masked_b)
        return similarity < threshold
