"""
Reconstructive Memory Module (Phase 6.0)
========================================
Implements reconstructive recall - the cognitive process of synthesizing
memories from fragments when exact matches are unavailable.

Based on cognitive science principles:
  - Memory is reconstructive, not reproductive (Bartlett, 1932)
  - Schema-based filling of gaps (Rumelhart, 1980)
  - Associative inference through related concepts

Key features:
  1. Fragment retrieval with similarity-based ranking
  2. Schema synthesis from multiple partial matches
  3. Confidence scoring for reconstructed vs stored memories
  4. Integration with GapDetector and GapFiller for validation
  5. Metadata tagging to distinguish reconstruction sources

Public API:
    reconstructor = ReconstructiveRecall(engine, config)

    # Direct recall with reconstruction
    result = await reconstructor.recall("What did we discuss about X?")

    # Check if memory was reconstructed
    if result.is_reconstructed:
        print(f"Confidence: {result.confidence:.2f} (reconstructed)")
    else:
        print("Exact memory retrieved")

    # Get fragments used for reconstruction
    for frag in result.fragments:
        print(f"  Fragment: {frag.content} (similarity: {frag.similarity:.2f})")
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from loguru import logger
import numpy as np

if TYPE_CHECKING:
    from ..core.engine import HAIMEngine
    from ..core.binary_hdv import BinaryHDV
    from ..core.node import MemoryNode
    from ..core.gap_detector import GapDetector, GapRecord


# ------------------------------------------------------------------ #
#  Data Classes                                                      #
# ------------------------------------------------------------------ #


@dataclass
class MemoryFragment:
    """
    A fragment of memory retrieved during reconstructive recall.

    Attributes:
        node_id: Unique identifier of the source memory node.
        content: Text content of the fragment.
        similarity: Semantic similarity score to the query [0.0, 1.0].
        source_tier: Storage tier where fragment was found (HOT/WARM/COLD).
        is_reconstructed_source: Whether this fragment itself was reconstructed.
        metadata: Original metadata from the source node.
    """
    node_id: str
    content: str
    similarity: float
    source_tier: str = "unknown"
    is_reconstructed_source: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "similarity": round(self.similarity, 4),
            "source_tier": self.source_tier,
            "is_reconstructed_source": self.is_reconstructed_source,
            "metadata": self.metadata,
        }


@dataclass
class ReconstructedMemory:
    """
    A synthesized memory produced by reconstructive recall.

    Attributes:
        content: The reconstructed/synthesized text content.
        fragments: List of MemoryFragment objects used in synthesis.
        confidence: Overall confidence score [0.0, 1.0].
        is_reconstructed: Always True for this class (vs stored memories).
        reconstruction_method: Method used (synthesis/inference/extrapolation).
        created_at: Timestamp of reconstruction.
        query_id: Hash of the original query.
        gap_detected: Whether a knowledge gap was identified.
    """
    content: str
    fragments: List[MemoryFragment]
    confidence: float
    is_reconstructed: bool = True
    reconstruction_method: str = "synthesis"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_id: str = ""
    gap_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "fragments": [f.to_dict() for f in self.fragments],
            "confidence": round(self.confidence, 4),
            "is_reconstructed": self.is_reconstructed,
            "reconstruction_method": self.reconstruction_method,
            "created_at": self.created_at.isoformat(),
            "query_id": self.query_id,
            "gap_detected": self.gap_detected,
            "fragment_count": len(self.fragments),
        }


@dataclass
class ReconstructionResult:
    """
    Complete result of a reconstructive recall operation.

    Attributes:
        reconstructed: Optional ReconstructedMemory if synthesis occurred.
        direct_matches: List of (node_id, similarity) for direct matches.
        fragments: All fragments retrieved (used or unused).
        confidence_breakdown: Detailed confidence scoring.
        is_reconstructed: Whether the primary result is reconstructed.
        gap_records: Any GapRecords created during the process.
    """
    reconstructed: Optional[ReconstructedMemory]
    direct_matches: List[Tuple[str, float]]
    fragments: List[MemoryFragment]
    confidence_breakdown: Dict[str, float]
    is_reconstructed: bool
    gap_records: List[GapRecord] = field(default_factory=list)

    def get_primary_content(self) -> str:
        """Get the primary content to return to the user."""
        if self.reconstructed:
            return self.reconstructed.content
        elif self.direct_matches:
            # Return content of best direct match
            return f"[Direct match available: {self.direct_matches[0][0]}]"
        return "[No relevant memories found]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reconstructed": self.reconstructed.to_dict() if self.reconstructed else None,
            "direct_matches": [(nid, round(sim, 4)) for nid, sim in self.direct_matches],
            "fragments": [f.to_dict() for f in self.fragments],
            "confidence_breakdown": self.confidence_breakdown,
            "is_reconstructed": self.is_reconstructed,
            "gap_count": len(self.gap_records),
        }


@dataclass
class ReconstructionConfig:
    """
    Configuration for reconstructive recall behavior.

    Attributes:
        min_similarity_threshold: Minimum similarity for fragment inclusion.
        max_fragments: Maximum number of fragments to use in synthesis.
        synthesis_threshold: Below this avg similarity, trigger reconstruction.
        confidence_weight_fragment: Weight for fragment similarity in confidence.
        confidence_weight_count: Weight for fragment count in confidence.
        confidence_weight_coherence: Weight for semantic coherence.
        enable_gap_detection: Whether to detect knowledge gaps.
        enable_persistent_storage: Whether to store reconstructed memories.
        reconstruction_tag: Metadata tag for stored reconstructions.
        max_synthesis_length: Maximum character length for synthesized content.
    """
    min_similarity_threshold: float = 0.35
    max_fragments: int = 7
    synthesis_threshold: float = 0.55
    confidence_weight_fragment: float = 0.5
    confidence_weight_count: float = 0.2
    confidence_weight_coherence: float = 0.3
    enable_gap_detection: bool = True
    enable_persistent_storage: bool = False
    reconstruction_tag: str = "reconstructive_recall"
    max_synthesis_length: int = 500


# ------------------------------------------------------------------ #
#  Reconstruction Engine                                             #
# ------------------------------------------------------------------ #


class ReconstructiveRecall:
    """
    Reconstructive memory engine that synthesizes memories from fragments.

    This module implements the cognitive science principle that memory recall
    is a constructive process - we actively rebuild memories from available
    fragments rather than playing back stored recordings.

    Key capabilities:
      1. Fragment retrieval with semantic similarity scoring
      2. Schema-based synthesis from partial matches
      3. Confidence calculation distinguishing stored vs reconstructed
      4. Gap detection for missing knowledge
      5. Optional persistent storage of reconstructions
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        config: Optional[ReconstructionConfig] = None,
        gap_detector: Optional["GapDetector"] = None,
    ):
        """
        Initialize the reconstructive recall engine.

        Args:
            engine: HAIMEngine instance for memory access.
            config: Optional configuration override.
            gap_detector: Optional GapDetector for knowledge gap detection.
        """
        self.engine = engine
        self.config = config or ReconstructionConfig()
        self.gap_detector = gap_detector

        # Statistics tracking
        self._stats = {
            "total_recalls": 0,
            "reconstructed_count": 0,
            "direct_match_count": 0,
            "gaps_detected": 0,
        }

    # ------------------------------------------------------------------ #
    #  Main API Methods                                                  #
    # ------------------------------------------------------------------ #

    async def recall(
        self,
        query: str,
        top_k: int = 10,
        enable_synthesis: bool = True,
        project_id: Optional[str] = None,
    ) -> ReconstructionResult:
        """
        Perform reconstructive recall for a given query.

        This is the primary entry point for reconstructive memory retrieval.
        It attempts to find direct matches first, then falls back to
        fragment-based synthesis if matches are insufficient.

        Args:
            query: The query text to recall memories for.
            top_k: Number of candidate fragments to retrieve.
            enable_synthesis: Whether to synthesize from fragments.
            project_id: Optional project ID for isolation masking.

        Returns:
            ReconstructionResult containing the best available answer.
        """
        self._stats["total_recalls"] += 1

        # Generate query ID for tracking
        query_id = hashlib.shake_256(query.lower().strip().encode()).hexdigest(12)

        # Step 1: Retrieve candidate fragments
        fragments = await self._retrieve_fragments(
            query, top_k=top_k, project_id=project_id
        )

        # Step 2: Analyze retrieval quality
        direct_matches = [(f.node_id, f.similarity) for f in fragments]

        # Step 3: Determine if reconstruction is needed
        avg_similarity = (
            sum(f.similarity for f in fragments[:3]) / min(3, len(fragments))
            if fragments else 0.0
        )

        reconstructed: Optional[ReconstructedMemory] = None
        gap_records: List[GapRecord] = []
        is_reconstructed = False

        # Trigger reconstruction if below threshold
        if enable_synthesis and avg_similarity < self.config.synthesis_threshold:
            is_reconstructed = True
            self._stats["reconstructed_count"] += 1

            # Detect knowledge gap if enabled
            if self.config.enable_gap_detection and self.gap_detector:
                gap_records = await self.gap_detector.assess_query(
                    query, direct_matches[:5], attention_mask=None
                )
                self._stats["gaps_detected"] += len(gap_records)

            # Synthesize reconstruction from fragments
            reconstructed = await self._synthesize_from_fragments(
                query, fragments, query_id, gap_detected=bool(gap_records)
            )

            # Optionally store the reconstruction
            if self.config.enable_persistent_storage and reconstructed:
                await self._store_reconstruction(reconstructed, query)
        else:
            self._stats["direct_match_count"] += 1

        # Calculate confidence breakdown
        confidence_breakdown = self._calculate_confidence_breakdown(
            fragments, reconstructed
        )

        return ReconstructionResult(
            reconstructed=reconstructed,
            direct_matches=direct_matches,
            fragments=fragments,
            confidence_breakdown=confidence_breakdown,
            is_reconstructed=is_reconstructed,
            gap_records=gap_records,
        )

    async def recall_with_fragments(
        self,
        query: str,
        fragment_texts: List[str],
        query_id: Optional[str] = None,
    ) -> ReconstructedMemory:
        """
        Reconstruct memory from explicitly provided fragment texts.

        Useful for external systems that want to leverage the synthesis
        capabilities with their own fragment sources.

        Args:
            query: The original query/context.
            fragment_texts: List of fragment text strings to synthesize.
            query_id: Optional query identifier for tracking.

        Returns:
            ReconstructedMemory synthesized from the provided fragments.
        """
        if query_id is None:
            query_id = hashlib.shake_256(query.lower().strip().encode()).hexdigest(12)

        # Create fragment objects from text
        fragments = [
            MemoryFragment(
                node_id=f"external_{i}_{hashlib.shake_256(t.encode()).hexdigest(8)}",
                content=t,
                similarity=0.7,  # Assume moderate similarity for external
                source_tier="external",
                is_reconstructed_source=False,
            )
            for i, t in enumerate(fragment_texts)
        ]

        return await self._synthesize_from_fragments(
            query, fragments, query_id, gap_detected=False
        )

    async def validate_reconstruction(
        self,
        reconstruction: ReconstructedMemory,
        is_helpful: bool,
        feedback: Optional[str] = None,
    ) -> None:
        """
        Record feedback on a reconstructed memory.

        Positive feedback can strengthen the synthesis patterns,
        while negative feedback may trigger gap detection.

        Args:
            reconstruction: The ReconstructedMemory to validate.
            is_helpful: Whether the reconstruction was useful.
            feedback: Optional textual feedback.
        """
        # Record retrieval feedback for fragments used
        for fragment in reconstruction.fragments:
            if not fragment.is_reconstructed_source:
                await self.engine.record_retrieval_feedback(
                    fragment.node_id,
                    helpful=is_helpful,
                    eig_signal=reconstruction.confidence,
                )

        # If not helpful, register as a gap
        if not is_helpful and self.gap_detector:
            query_for_gap = reconstruction.content[:100]  # Use content as proxy
            await self.gap_detector.register_negative_feedback(query_for_gap)

        logger.info(
            f"Reconstruction validated: helpful={is_helpful}, "
            f"confidence={reconstruction.confidence:.3f}"
        )

    # ------------------------------------------------------------------ #
    #  Fragment Retrieval                                                #
    # ------------------------------------------------------------------ #

    async def _retrieve_fragments(
        self,
        query: str,
        top_k: int,
        project_id: Optional[str] = None,
    ) -> List[MemoryFragment]:
        """
        Retrieve memory fragments relevant to the query.

        Performs a semantic search and converts results to MemoryFragment
        objects with similarity scores and metadata.

        Args:
            query: Query text.
            top_k: Number of fragments to retrieve.
            project_id: Optional project ID for isolation.

        Returns:
            List of MemoryFragment objects sorted by similarity.
        """
        # Query the engine
        search_results = await self.engine.query(
            query,
            top_k=top_k,
            associative_jump=True,
            track_gaps=False,  # We handle gaps separately
            project_id=project_id,
            include_neighbors=True,
        )

        fragments: List[MemoryFragment] = []

        for node_id, similarity in search_results:
            # Filter by threshold
            if similarity < self.config.min_similarity_threshold:
                continue

            # Get the full node
            node = await self.engine.get_memory(node_id)
            if not node:
                continue

            # Determine tier
            tier = getattr(node, "tier", "unknown")

            # Check if this was itself reconstructed
            is_reconstructed = node.metadata.get("is_reconstructed", False) if node.metadata else False

            fragment = MemoryFragment(
                node_id=node_id,
                content=node.content,
                similarity=similarity,
                source_tier=tier,
                is_reconstructed_source=is_reconstructed,
                metadata=node.metadata or {},
            )

            fragments.append(fragment)

        # Sort by similarity descending
        fragments.sort(key=lambda f: f.similarity, reverse=True)

        # Limit to max_fragments
        if len(fragments) > self.config.max_fragments:
            fragments = fragments[: self.config.max_fragments]

        return fragments

    # ------------------------------------------------------------------ #
    #  Synthesis Methods                                                 #
    # ------------------------------------------------------------------ #

    async def _synthesize_from_fragments(
        self,
        query: str,
        fragments: List[MemoryFragment],
        query_id: str,
        gap_detected: bool,
    ) -> ReconstructedMemory:
        """
        Synthesize a coherent memory from fragments.

        This implements the core reconstructive process: combining partial
        information into a coherent whole using semantic synthesis.

        Args:
            query: Original query text.
            fragments: Retrieved memory fragments.
            query_id: Query identifier.
            gap_detected: Whether a knowledge gap was identified.

        Returns:
            ReconstructedMemory with synthesized content.
        """
        if not fragments:
            # No fragments available - create placeholder
            return ReconstructedMemory(
                content=f"[No relevant memories found for: {query[:100]}]",
                fragments=[],
                confidence=0.0,
                reconstruction_method="none",
                query_id=query_id,
                gap_detected=gap_detected,
            )

        # Select synthesis method based on fragment characteristics
        method = self._determine_synthesis_method(fragments)

        # Perform synthesis
        if method == "extraction":
            content = await self._synthesis_extraction(query, fragments)
        elif method == "interpolation":
            content = await self._synthesis_interpolation(query, fragments)
        else:  # synthesis
            content = await self._synthesis_combination(query, fragments)

        # Truncate if needed
        original_content = content
        if len(content) > self.config.max_synthesis_length:
            content = content[: self.config.max_synthesis_length - 3] + "..."

        # Calculate confidence
        confidence = self._calculate_reconstruction_confidence(fragments, content)

        return ReconstructedMemory(
            content=content,
            fragments=fragments,
            confidence=confidence,
            reconstruction_method=method,
            query_id=query_id,
            gap_detected=gap_detected,
        )

    def _determine_synthesis_method(self, fragments: List[MemoryFragment]) -> str:
        """
        Determine the best synthesis method based on fragment characteristics.

        Methods:
          - extraction: Use best fragment directly (high similarity single match)
          - interpolation: Blend between two related fragments
          - synthesis: Combine multiple fragments into new representation
        """
        if not fragments:
            return "synthesis"

        # Single high-quality fragment -> extraction
        if len(fragments) == 1 or fragments[0].similarity > 0.75:
            return "extraction"

        # Two closely related fragments -> interpolation
        if len(fragments) == 2:
            return "interpolation"

        # Multiple fragments -> full synthesis
        return "synthesis"

    async def _synthesis_extraction(
        self,
        query: str,
        fragments: List[MemoryFragment],
    ) -> str:
        """
        Extract the best fragment with minor contextual enhancement.

        Used when a single high-quality match is found.
        """
        best = fragments[0]

        # Add context if multiple fragments available
        context = ""
        if len(fragments) > 1:
            related_topics = set()
            for f in fragments[1:4]:
                # Extract key terms (simple heuristic)
                words = re.findall(r'\b\w{4,}\b', f.content.lower())
                related_topics.update(words[:3])

            if related_topics:
                context = f" Related: {', '.join(list(related_topics)[:5])}."

        return f"{best.content}{context}"

    async def _synthesis_interpolation(
        self,
        query: str,
        fragments: List[MemoryFragment],
    ) -> str:
        """
        Interpolate between two related fragments.

        Creates a blended response that captures the essence of both.
        """
        if len(fragments) < 2:
            return fragments[0].content if fragments else "[Insufficient data]"

        f1, f2 = fragments[0], fragments[1]

        # Weight by similarity
        total_sim = f1.similarity + f2.similarity
        w1 = f1.similarity / total_sim if total_sim > 0 else 0.5
        w2 = f2.similarity / total_sim if total_sim > 0 else 0.5

        # Simple concatenation with weighting indication
        # In a full implementation, this would use semantic blending
        if w1 > 0.7:
            return f"{f1.content} Additionally: {f2.content}"
        elif w2 > 0.7:
            return f"{f2.content} Context: {f1.content}"
        else:
            return f"Combining related information: {f1.content} {f2.content}"

    async def _synthesis_combination(
        self,
        query: str,
        fragments: List[MemoryFragment],
    ) -> str:
        """
        Combine multiple fragments into a coherent synthesis.

        This is the most complex synthesis method, creating a new
        representation that wasn't explicitly stored but is implied
        by the combination of fragments.
        """
        if not fragments:
            return "[No information available]"

        # Group fragments by similarity tiers
        high_conf = [f for f in fragments if f.similarity > 0.6]
        medium_conf = [f for f in fragments if 0.4 < f.similarity <= 0.6]
        low_conf = [f for f in fragments if f.similarity <= 0.4]

        parts = []

        # Primary content from high-confidence fragments
        if high_conf:
            primary_texts = [f.content for f in high_conf[:3]]
            parts.append(" ".join(primary_texts))

        # Supplementary context from medium-confidence
        if medium_conf:
            secondary = [f.content for f in medium_conf[:2]]
            if secondary:
                parts.append(f"Additional context: {' '.join(secondary)}")

        # Low-confidence hints
        if low_conf and not parts:
            # Only use low-confidence if nothing else available
            hints = [f.content for f in low_conf[:3]]
            parts.append(f"Related information: {' '.join(hints)}")

        if not parts:
            return "[Limited information available for reconstruction]"

        return " ".join(parts)

    # ------------------------------------------------------------------ #
    #  Confidence Calculation                                            #
    # ------------------------------------------------------------------ #

    def _calculate_reconstruction_confidence(
        self,
        fragments: List[MemoryFragment],
        synthesized_content: str,
    ) -> float:
        """
        Calculate overall confidence score for a reconstructed memory.

        Confidence is based on:
          1. Fragment similarity scores (weighted average)
          2. Number of fragments (more = more confident, up to a point)
          3. Semantic coherence of the synthesis

        Args:
            fragments: Fragments used in reconstruction.
            synthesized_content: The synthesized text.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        if not fragments:
            return 0.0

        # 1. Fragment similarity component
        if fragments:
            avg_sim = sum(f.similarity for f in fragments) / len(fragments)
            fragment_score = avg_sim
        else:
            fragment_score = 0.0

        # 2. Fragment count component (diminishing returns)
        count_score = min(1.0, len(fragments) / 5.0)

        # 3. Coherence component (length-based heuristic)
        coherence_score = min(1.0, len(synthesized_content) / 100.0)

        # Weighted combination
        confidence = (
            self.config.confidence_weight_fragment * fragment_score +
            self.config.confidence_weight_count * count_score +
            self.config.confidence_weight_coherence * coherence_score
        )

        return round(float(confidence), 4)

    def _calculate_confidence_breakdown(
        self,
        fragments: List[MemoryFragment],
        reconstructed: Optional[ReconstructedMemory],
    ) -> Dict[str, float]:
        """
        Calculate detailed confidence breakdown for result.

        Returns individual components for transparency.
        """
        if fragments:
            avg_similarity = sum(f.similarity for f in fragments) / len(fragments)
            max_similarity = max(f.similarity for f in fragments)
            fragment_count = len(fragments)
        else:
            avg_similarity = 0.0
            max_similarity = 0.0
            fragment_count = 0

        return {
            "avg_fragment_similarity": round(avg_similarity, 4),
            "max_fragment_similarity": round(max_similarity, 4),
            "fragment_count": fragment_count,
            "reconstruction_confidence": reconstructed.confidence if reconstructed else 0.0,
            "overall_confidence": (
                reconstructed.confidence if reconstructed else max_similarity
            ),
        }

    # ------------------------------------------------------------------ #
    #  Persistent Storage                                                #
    # ------------------------------------------------------------------ #

    async def _store_reconstruction(
        self,
        reconstruction: ReconstructedMemory,
        original_query: str,
    ) -> Optional[str]:
        """
        Store a reconstructed memory for future retrieval.

        Reconstructed memories are tagged with metadata so they can be
        distinguished from directly stored memories.

        Args:
            reconstruction: The ReconstructedMemory to store.
            original_query: The query that triggered reconstruction.

        Returns:
            Node ID if stored, None otherwise.
        """
        metadata = {
            "is_reconstructed": True,
            "reconstruction_method": reconstruction.reconstruction_method,
            "query_id": reconstruction.query_id,
            "original_query": original_query,
            "fragment_count": len(reconstruction.fragments),
            "fragment_sources": [f.node_id for f in reconstruction.fragments],
            "tags": [self.config.reconstruction_tag, "synthesized"],
            "created_at": reconstruction.created_at.isoformat(),
        }

        try:
            node_id = await self.engine.store(
                reconstruction.content,
                metadata=metadata,
            )
            logger.info(
                f"Stored reconstruction {node_id} "
                f"(confidence: {reconstruction.confidence:.3f})"
            )
            return node_id
        except Exception as e:
            logger.warning(f"Failed to store reconstruction: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Statistics and Utilities                                          #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics about reconstructive recall operations."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_recalls": 0,
            "reconstructed_count": 0,
            "direct_match_count": 0,
            "gaps_detected": 0,
        }


# ------------------------------------------------------------------ #
#  Utility Functions                                                  #
# ------------------------------------------------------------------ #


def create_reconstruction_config(
    min_similarity: Optional[float] = None,
    max_fragments: Optional[int] = None,
    synthesis_threshold: Optional[float] = None,
    enable_storage: Optional[bool] = None,
) -> ReconstructionConfig:
    """
    Convenience function to create a ReconstructionConfig with overrides.

    Args:
        min_similarity: Override minimum similarity threshold.
        max_fragments: Override maximum fragments to use.
        synthesis_threshold: Override synthesis trigger threshold.
        enable_storage: Override persistent storage setting.

    Returns:
        ReconstructionConfig with specified overrides applied.
    """
    config = ReconstructionConfig()

    if min_similarity is not None:
        config.min_similarity_threshold = min_similarity
    if max_fragments is not None:
        config.max_fragments = max_fragments
    if synthesis_threshold is not None:
        config.synthesis_threshold = synthesis_threshold
    if enable_storage is not None:
        config.enable_persistent_storage = enable_storage

    return config


def is_reconstructed_memory(node: "MemoryNode") -> bool:
    """
    Check if a memory node was created via reconstructive recall.

    Utility function for filtering or identifying reconstructed memories.

    Args:
        node: MemoryNode to check.

    Returns:
        True if the memory was reconstructed, False otherwise.
    """
    if not node or not node.metadata:
        return False
    return node.metadata.get("is_reconstructed", False)


def get_reconstruction_metadata(node: "MemoryNode") -> Dict[str, Any]:
    """
    Extract reconstruction-specific metadata from a memory node.

    Returns a dictionary with reconstruction details if the node was
    reconstructed, or an empty dict otherwise.

    Args:
        node: MemoryNode to extract metadata from.

    Returns:
        Dictionary with reconstruction metadata.
    """
    if not is_reconstructed_memory(node):
        return {}

    meta = node.metadata or {}
    return {
        "method": meta.get("reconstruction_method", "unknown"),
        "query_id": meta.get("query_id", ""),
        "fragment_count": meta.get("fragment_count", 0),
        "confidence": meta.get("reconstruction_confidence", 0.0),
    }
