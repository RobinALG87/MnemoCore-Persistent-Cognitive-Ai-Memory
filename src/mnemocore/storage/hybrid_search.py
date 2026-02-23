"""
Hybrid Search Module - Storage Layer Wrapper
==============================================
This module provides a convenient interface to the hybrid search functionality.

The core implementation is in mnemocore.core.hybrid_search, but this wrapper
provides storage-layer specific utilities and integration helpers.

Phase 4.6: Enhanced semantic retrieval with multi-modal search support.
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

# Re-export core hybrid search components
from mnemocore.core.hybrid_search import (
    HybridSearchEngine as _HybridSearchEngine,
    HybridSearchConfig as _HybridSearchConfig,
    SearchResult,
    SparseEncoder,
    ReciprocalRankFusion,
)


class HybridSearchConfig(_HybridSearchConfig):
    """Storage-layer specific hybrid search configuration."""
    pass


class HybridSearchEngine(_HybridSearchEngine):
    """Storage-layer specific hybrid search engine with additional utilities."""

    def __init__(self, config: Optional[HybridSearchConfig] = None):
        super().__init__(config)
        logger.info("Initialized HybridSearchEngine for storage layer")

    async def search_qdrant_points(
        self,
        query: str,
        qdrant_points: List[Any],
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Search hybrid using Qdrant PointStruct objects directly.

        Args:
            query: Search query string
            qdrant_points: List of Qdrant PointStruct results
            limit: Maximum results to return

        Returns:
            List of SearchResult with combined scores
        """
        dense_results = [
            (str(p.id), getattr(p, "score", 0.0))
            for p in qdrant_points
        ]

        payloads = {
            str(p.id): p.payload or {}
            for p in qdrant_points
        }

        return await self.search(
            query=query,
            dense_results=dense_results,
            dense_payloads=payloads,
            limit=limit,
        )


def create_hybrid_search_engine(
    mode: str = "hybrid",
    alpha: float = 0.7,
    rrf_k: int = 60,
) -> HybridSearchEngine:
    """
    Factory function to create a configured hybrid search engine.

    Args:
        mode: Search mode ("dense", "sparse", or "hybrid")
        alpha: Weight for dense search in hybrid mode (0-1)
        rrf_k: RRF constant for rank fusion

    Returns:
        Configured HybridSearchEngine instance
    """
    config = HybridSearchConfig(
        mode=mode,
        hybrid_alpha=alpha,
        rrf_k=rrf_k,
    )
    return HybridSearchEngine(config)


__all__ = [
    "HybridSearchEngine",
    "HybridSearchConfig",
    "SearchResult",
    "SparseEncoder",
    "ReciprocalRankFusion",
    "create_hybrid_search_engine",
]
