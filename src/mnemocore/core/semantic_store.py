"""
Semantic Store Service
======================
Manages semantic concepts, abstractions, and long-term general knowledge.
Integrates with Qdrant for persistent vector search and local cache for fast access.

Phase 5.1: Full Qdrant persistence, concept consolidation from episodes,
reliability decay, and HDV-based similarity search.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import threading
import asyncio
import logging
import uuid

from .memory_model import SemanticConcept
from .binary_hdv import BinaryHDV

logger = logging.getLogger(__name__)


class SemanticStoreService:
    """
    Semantic memory store — the neocortex analog in CLS theory.

    Provides long-term storage of abstracted concepts derived from
    episodic experiences. Supports both local brute-force and Qdrant-backed
    similarity search. Concepts are append-only with reliability tracking.
    """

    def __init__(self, qdrant_store=None, config=None):
        """
        Args:
            qdrant_store: QdrantStore instance for HDV persistence and search.
            config: SemanticConfig from HAIMConfig (optional, uses defaults).
        """
        self._qdrant_store = qdrant_store
        self._config = config

        # Local cache of semantic concepts — fast path for frequent access
        self._concepts: Dict[str, SemanticConcept] = {}
        self._lock = threading.RLock()

        # Stats
        self._upsert_count: int = 0
        self._search_count: int = 0
        self._qdrant_sync_count: int = 0

    @property
    def concept_count(self) -> int:
        """Number of concepts in local cache."""
        with self._lock:
            return len(self._concepts)

    def upsert_concept(self, concept: SemanticConcept) -> None:
        """
        Add or update a semantic concept locally.

        For Qdrant persistence, call upsert_concept_persistent() (async).
        """
        with self._lock:
            self._concepts[concept.id] = concept
            self._upsert_count += 1
            logger.debug(f"Upserted semantic concept {concept.id} ({concept.label})")

    async def upsert_concept_persistent(self, concept: SemanticConcept) -> None:
        """
        Add or update a concept in both local cache and Qdrant.

        Converts the BinaryHDV prototype to bipolar float for Qdrant DOT distance.
        """
        self.upsert_concept(concept)

        if not self._qdrant_store:
            return

        try:
            from qdrant_client import models as qmodels
            import numpy as np

            # Convert binary HDV {0,1} to bipolar {-1,+1} for Qdrant DOT distance
            raw = concept.prototype_hdv.to_numpy()
            bipolar = (raw.astype(float) * 2.0 - 1.0).tolist()

            point = qmodels.PointStruct(
                id=concept.id,
                vector=bipolar,
                payload={
                    "content": f"{concept.label}: {concept.description}",
                    "concept_label": concept.label,
                    "concept_tags": concept.tags,
                    "reliability": concept.reliability,
                    "support_episode_count": len(concept.support_episode_ids),
                    "unix_timestamp": int(concept.last_updated_at.timestamp()),
                    "type": "semantic_concept",
                },
            )

            collection = getattr(self._qdrant_store, "collection_hot", "haim_hot")
            await self._qdrant_store.upsert(collection=collection, points=[point])
            self._qdrant_sync_count += 1
            logger.debug(f"Persisted concept {concept.id} to Qdrant")

        except Exception as e:
            logger.warning(f"Failed to persist concept {concept.id} to Qdrant: {e}")

    def get_concept(self, concept_id: str) -> Optional[SemanticConcept]:
        """Retrieve a specific concept by its ID from local cache."""
        with self._lock:
            return self._concepts.get(concept_id)

    def find_nearby_concepts(
        self, hdv: BinaryHDV, top_k: int = 5, min_similarity: float = 0.5
    ) -> List[SemanticConcept]:
        """
        Find conceptually similar semantic anchors via local brute-force.

        For large concept stores, prefer find_nearby_concepts_qdrant() which
        uses Qdrant's HNSW index for O(log n) search.
        """
        self._search_count += 1
        with self._lock:
            results = []
            for concept in self._concepts.values():
                sim = hdv.similarity(concept.prototype_hdv)
                if sim >= min_similarity:
                    results.append((sim, concept))

            results.sort(key=lambda x: (-x[0], x[1].id))
            return [c for _, c in results[:top_k]]

    async def find_nearby_concepts_qdrant(
        self, hdv: BinaryHDV, top_k: int = 5, min_similarity: float = 0.5
    ) -> List[SemanticConcept]:
        """
        Find similar concepts using Qdrant ANN search (O(log n) via HNSW).

        Falls back to local brute-force if Qdrant is unavailable.
        """
        if not self._qdrant_store:
            return self.find_nearby_concepts(hdv, top_k, min_similarity)

        try:
            import numpy as np

            raw = hdv.to_numpy()
            bipolar = (raw.astype(float) * 2.0 - 1.0).tolist()

            collection = getattr(self._qdrant_store, "collection_hot", "haim_hot")
            scored_points = await self._qdrant_store.search(
                collection=collection,
                query_vector=bipolar,
                limit=top_k,
                score_threshold=min_similarity,
                metadata_filter={"type": "semantic_concept"},
            )

            results = []
            with self._lock:
                for sp in scored_points:
                    concept = self._concepts.get(str(sp.id))
                    if concept:
                        results.append(concept)

            return results

        except Exception as e:
            logger.warning(f"Qdrant concept search failed, falling back to local: {e}")
            return self.find_nearby_concepts(hdv, top_k, min_similarity)

    def adjust_concept_reliability(self, concept_id: str, delta: float) -> None:
        """Bump or decay the reliability of a concept (e.g. via Pulse immunology loops)."""
        with self._lock:
            concept = self._concepts.get(concept_id)
            if not concept:
                return

            concept.reliability = max(0.0, min(1.0, concept.reliability + delta))
            concept.last_updated_at = datetime.now(timezone.utc)

    def consolidate_from_content(
        self,
        content: str,
        hdv: BinaryHDV,
        episode_ids: List[str],
        tags: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[SemanticConcept]:
        """
        Attempt to consolidate content into an existing or new semantic concept.

        If a sufficiently similar concept exists (>= min_similarity), merges
        the episode support. Otherwise creates a new concept.

        Returns the upserted concept, or None if content is empty.
        """
        if not content or not content.strip():
            return None

        min_sim = 0.5
        if self._config:
            min_sim = getattr(self._config, "min_similarity_threshold", 0.5)

        # Check for existing similar concept
        nearby = self.find_nearby_concepts(hdv, top_k=1, min_similarity=min_sim)

        if nearby:
            existing = nearby[0]
            # Merge episode support
            for ep_id in episode_ids:
                if ep_id not in existing.support_episode_ids:
                    existing.support_episode_ids.append(ep_id)
            # Boost reliability on reinforcement
            existing.reliability = min(1.0, existing.reliability + 0.02)
            existing.last_updated_at = datetime.now(timezone.utc)
            logger.debug(f"Merged into existing concept {existing.id} ({existing.label})")
            return existing

        # Create new concept
        concept = SemanticConcept(
            id=f"sc_{uuid.uuid4().hex[:12]}",
            label=content[:80].strip(),
            description=content[:500].strip(),
            tags=tags or [],
            prototype_hdv=hdv,
            support_episode_ids=list(episode_ids),
            reliability=0.5,
            last_updated_at=datetime.now(timezone.utc),
            metadata={"agent_id": agent_id} if agent_id else {},
        )

        self.upsert_concept(concept)
        logger.info(f"Created new semantic concept {concept.id}: {concept.label[:50]}")
        return concept

    def decay_all_reliability(self, decay_rate: float = 0.01) -> int:
        """
        Apply reliability decay to all concepts. Returns count of decayed concepts.

        Concepts below min threshold could be candidates for archival.
        """
        decayed = 0
        with self._lock:
            for concept in self._concepts.values():
                if concept.reliability > 0.0:
                    concept.reliability = max(0.0, concept.reliability - decay_rate)
                    decayed += 1
        return decayed

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics."""
        with self._lock:
            return {
                "concept_count": len(self._concepts),
                "upsert_count": self._upsert_count,
                "search_count": self._search_count,
                "qdrant_sync_count": self._qdrant_sync_count,
                "qdrant_connected": self._qdrant_store is not None,
            }

    def get_all_concepts(self) -> List[SemanticConcept]:
        """Return all concepts (snapshot). For diagnostics/export."""
        with self._lock:
            return list(self._concepts.values())

