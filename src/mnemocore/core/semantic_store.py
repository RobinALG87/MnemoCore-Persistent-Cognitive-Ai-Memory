"""
Semantic Store Service
======================
Manages semantic concepts, abstractions, and long-term general knowledge.
Integrates tightly with Qdrant/vector storage for conceptual search and retrieval.
"""

from typing import Dict, List, Optional
import threading
import logging

from .memory_model import SemanticConcept
from .binary_hdv import BinaryHDV

logger = logging.getLogger(__name__)


class SemanticStoreService:
    def __init__(self, qdrant_store=None):
        """
        Args:
            qdrant_store: The underlying QdrantStore instance for HDV similarity searches.
        """
        self._qdrant_store = qdrant_store
        
        # Local cache of semantic concepts, usually backed by proper storage
        # In a full implementation, `upsert_concept` also stores to Qdrant/disk.
        self._concepts: Dict[str, SemanticConcept] = {}
        self._lock = threading.RLock()

    def upsert_concept(self, concept: SemanticConcept) -> None:
        """Add or update a semantic concept and ensure it is indexed for search."""
        with self._lock:
            self._concepts[concept.id] = concept
            logger.debug(f"Upserted semantic concept {concept.id} ({concept.label})")
            
            # Note: A complete implementation would push the prototype HDV
            # and concept metadata into Qdrant for semantic search here.
            pass

    def get_concept(self, concept_id: str) -> Optional[SemanticConcept]:
        """Retrieve a specific concept by its ID."""
        with self._lock:
            return self._concepts.get(concept_id)

    def find_nearby_concepts(
        self, hdv: BinaryHDV, top_k: int = 5, min_similarity: float = 0.5
    ) -> List[SemanticConcept]:
        """
        Find conceptually similar semantic anchors.
        Delegates to Qdrant or performs local brute-force if in minimal mode.
        """
        with self._lock:
            # Temporary local-only similarity threshold search
            results = []
            for concept in self._concepts.values():
                sim = hdv.similarity(concept.prototype_hdv)
                if sim >= min_similarity:
                    results.append((sim, concept))

            # Sort descending by similarity, then ascending by creation/id
            results.sort(key=lambda x: (-x[0], x[1].id))
            return [c for _, c in results[:top_k]]

    def adjust_concept_reliability(self, concept_id: str, delta: float) -> None:
        """Bump or decay the reliability of a concept (e.g. via Pulse immunology loops)."""
        with self._lock:
            concept = self._concepts.get(concept_id)
            if not concept:
                return
            
            concept.reliability = max(0.0, min(1.0, concept.reliability + delta))

