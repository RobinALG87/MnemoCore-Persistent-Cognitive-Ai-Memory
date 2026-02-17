"""
Qdrant Vector Store (Sync)
==========================
Wrapper for QdrantClient handling:
1. Collection management (HOT vs WARM tiers)
2. Binary Quantization (BQ) configuration
3. Sync batch upsert/search (for integration with sync Engine)
"""

import logging
from typing import List, Any, Optional

from qdrant_client import QdrantClient, models

from .config import get_config

logger = logging.getLogger(__name__)

class QdrantStore:
    _instance = None

    def __init__(self):
        self.config = get_config()
        self.client = QdrantClient(
            url=self.config.qdrant.url,
            api_key=self.config.qdrant.api_key
        )
        self.dim = self.config.dimensionality

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_collections(self):
        """Ensure HOT and WARM collections exist with proper schema."""
        cfg = self.config.qdrant
        
        # Define BQ config if enabled
        quantization_config = None
        if cfg.binary_quantization:
            quantization_config = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=cfg.always_ram
                )
            )

        # Create HOT collection (optimized for latency)
        if not self.client.collection_exists(cfg.collection_hot):
            logger.info(f"Creating HOT collection: {cfg.collection_hot}")
            self.client.create_collection(
                collection_name=cfg.collection_hot,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE, # Cosine on bipolar (-1,1) ~ Hamming
                    on_disk=False
                ),
                quantization_config=quantization_config,
                hnsw_config=models.HnswConfigDiff(
                    m=cfg.hnsw_m,
                    ef_construct=cfg.hnsw_ef_construct,
                    on_disk=False
                )
            )

        # Create WARM collection (optimized for scale/disk)
        if not self.client.collection_exists(cfg.collection_warm):
            logger.info(f"Creating WARM collection: {cfg.collection_warm}")
            self.client.create_collection(
                collection_name=cfg.collection_warm,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                quantization_config=quantization_config,
                hnsw_config=models.HnswConfigDiff(
                    m=cfg.hnsw_m, 
                    ef_construct=cfg.hnsw_ef_construct,
                    on_disk=True
                )
            )

    def upsert(self, collection: str, points: List[models.PointStruct]):
        """Sync batch upsert."""
        try:
            self.client.upsert(
                collection_name=collection,
                points=points
            )
        except Exception as e:
            logger.error(f"Qdrant upsert failed for {collection}: {e}")
            raise

    def search(self, collection: str, query_vector: List[float], limit: int = 5, score_threshold: float = 0.0) -> List[models.ScoredPoint]:
        """Sync semantic search."""
        return self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
    def get_point(self, collection: str, point_id: str) -> Optional[models.Record]:
        """Get a single point by ID."""
        records = self.client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_vectors=True,
            with_payload=True
        )
        if records:
            return records[0]
        return None

    def scroll(self, collection: str, limit: int = 100, offset: Any = None, with_vectors: bool = False) -> Any:
        """Scroll/Iterate over collection (for consolidation)."""
        return self.client.scroll(
            collection_name=collection,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=True,
            offset=offset
        )

    def delete(self, collection: str, point_ids: List[str]):
        """Delete points by ID."""
        self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=point_ids)
        )

    def close(self):
        self.client.close()
