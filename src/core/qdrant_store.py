import logging
from typing import List, Any, Optional
import asyncio

from qdrant_client import AsyncQdrantClient, models

from .config import get_config
from .resilience import vector_circuit_breaker

logger = logging.getLogger(__name__)

class QdrantStore:
    _instance = None

    def __init__(self):
        self.config = get_config()
        self.client = AsyncQdrantClient(
            url=self.config.qdrant.url,
            api_key=self.config.qdrant.api_key
        )
        self.dim = self.config.dimensionality

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def ensure_collections(self):
        """Ensure HOT and WARM collections exist with proper schema."""
        try:
            return await vector_circuit_breaker.call_async(self._ensure_collections)
        except Exception:
            # ensure_collections if critical, but if breaker is open, we carry on if we can
            logger.error("Circuit breaker blocked or failed ensure_collections")
            raise

    async def _ensure_collections(self):
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
        if not await self.client.collection_exists(cfg.collection_hot):
            logger.info(f"Creating HOT collection: {cfg.collection_hot}")
            await self.client.create_collection(
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
        if not await self.client.collection_exists(cfg.collection_warm):
            logger.info(f"Creating WARM collection: {cfg.collection_warm}")
            await self.client.create_collection(
                collection_name=cfg.collection_warm,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.MANHATTAN, # Manhattan on 0/1 is exactly Hamming
                    on_disk=True
                ),
                quantization_config=quantization_config,
                hnsw_config=models.HnswConfigDiff(
                    m=cfg.hnsw_m, 
                    ef_construct=cfg.hnsw_ef_construct,
                    on_disk=True
                )
            )

    async def upsert(self, collection: str, points: List[models.PointStruct]):
        """Async batch upsert."""
        try:
            await vector_circuit_breaker.call_async(self.client.upsert, collection_name=collection, points=points)
        except Exception as e:
            logger.exception(f"Qdrant upsert failed or blocked for {collection}")
            raise

    async def search(self, collection: str, query_vector: List[float], limit: int = 5, score_threshold: float = 0.0) -> List[models.ScoredPoint]:
        """Async semantic search."""
        try:
            return await vector_circuit_breaker.call_async(self.client.search,
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
        except Exception:
            logger.error(f"Qdrant search failed or blocked for {collection}")
            return []
        
    async def get_point(self, collection: str, point_id: str) -> Optional[models.Record]:
        """Get a single point by ID."""
        try:
            records = await vector_circuit_breaker.call_async(self.client.retrieve,
                collection_name=collection,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )
            if records:
                return records[0]
        except Exception:
            logger.error(f"Qdrant get_point failed or blocked for {point_id}")
        return None

    async def scroll(self, collection: str, limit: int = 100, offset: Any = None, with_vectors: bool = False) -> Any:
        """Scroll/Iterate over collection (for consolidation)."""
        try:
            return await vector_circuit_breaker.call_async(self.client.scroll,
                collection_name=collection,
                limit=limit,
                with_vectors=with_vectors,
                with_payload=True,
                offset=offset
            )
        except Exception:
            logger.error(f"Qdrant scroll failed or blocked for {collection}")
            return [], None

    async def delete(self, collection: str, point_ids: List[str]):
        """Delete points by ID."""
        try:
            await vector_circuit_breaker.call_async(self.client.delete,
                collection_name=collection,
                points_selector=models.PointIdsList(points=point_ids)
            )
        except Exception:
            logger.error(f"Qdrant delete failed or blocked for {point_ids}")
            raise

    async def close(self):
        await self.client.close()
