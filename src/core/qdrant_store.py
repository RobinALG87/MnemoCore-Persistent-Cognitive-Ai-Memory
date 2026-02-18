"""
Qdrant Vector Store Layer
=========================
Provides async access to Qdrant for vector storage and similarity search.
"""

import logging
from typing import List, Any, Optional
import asyncio

from qdrant_client import AsyncQdrantClient, models

from .config import get_config
from .reliability import qdrant_breaker
from .exceptions import (
    CircuitOpenError,
    StorageConnectionError,
    wrap_storage_exception,
)

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
        """
        Ensure HOT and WARM collections exist with proper schema.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Qdrant connection fails.
        """
        try:
            return await qdrant_breaker.call(self._ensure_collections)
        except CircuitOpenError:
            logger.error("Circuit breaker blocked ensure_collections")
            raise
        except Exception as e:
            logger.error(f"Qdrant ensure_collections failed: {e}")
            raise wrap_storage_exception("qdrant", "ensure_collections", e)

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
                    distance=models.Distance.COSINE,
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
                    distance=models.Distance.MANHATTAN,
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
        """
        Async batch upsert.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Qdrant connection fails.
        """
        try:
            await qdrant_breaker.call(
                self.client.upsert, collection_name=collection, points=points
            )
        except CircuitOpenError:
            logger.error(f"Qdrant upsert blocked for {collection}: circuit breaker open")
            raise
        except Exception as e:
            logger.exception(f"Qdrant upsert failed for {collection}")
            raise wrap_storage_exception("qdrant", "upsert", e)

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0
    ) -> List[models.ScoredPoint]:
        """
        Async semantic search.

        Returns:
            List of scored points (empty list on errors).

        Note:
            This method returns an empty list on errors rather than raising,
            as search failures should not crash the calling code.
        """
        try:
            return await qdrant_breaker.call(
                self.client.search,
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
        except CircuitOpenError:
            logger.warning(f"Qdrant search blocked for {collection}: circuit breaker open")
            return []
        except Exception as e:
            logger.error(f"Qdrant search failed for {collection}: {e}")
            return []

    async def get_point(self, collection: str, point_id: str) -> Optional[models.Record]:
        """
        Get a single point by ID.

        Returns:
            Record if found, None if not found.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Qdrant connection fails.
        """
        try:
            records = await qdrant_breaker.call(
                self.client.retrieve,
                collection_name=collection,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )
            if records:
                return records[0]
            return None  # Not found - expected case
        except CircuitOpenError:
            logger.error(f"Qdrant get_point blocked for {point_id}: circuit breaker open")
            raise
        except Exception as e:
            logger.error(f"Qdrant get_point failed for {point_id}: {e}")
            raise wrap_storage_exception("qdrant", "get_point", e)

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: Any = None,
        with_vectors: bool = False
    ) -> Any:
        """
        Scroll/Iterate over collection (for consolidation).

        Returns:
            Tuple of (points, next_offset). Returns ([], None) on errors.

        Note:
            This method returns empty results on errors rather than raising,
            as scroll is typically used for background operations.
        """
        try:
            return await qdrant_breaker.call(
                self.client.scroll,
                collection_name=collection,
                limit=limit,
                with_vectors=with_vectors,
                with_payload=True,
                offset=offset
            )
        except CircuitOpenError:
            logger.warning(f"Qdrant scroll blocked for {collection}: circuit breaker open")
            return [], None
        except Exception as e:
            logger.error(f"Qdrant scroll failed for {collection}: {e}")
            return [], None

    async def delete(self, collection: str, point_ids: List[str]):
        """
        Delete points by ID.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Qdrant connection fails.
        """
        try:
            await qdrant_breaker.call(
                self.client.delete,
                collection_name=collection,
                points_selector=models.PointIdsList(points=point_ids)
            )
        except CircuitOpenError:
            logger.error(f"Qdrant delete blocked for {point_ids}: circuit breaker open")
            raise
        except Exception as e:
            logger.error(f"Qdrant delete failed for {point_ids}: {e}")
            raise wrap_storage_exception("qdrant", "delete", e)

    async def close(self):
        await self.client.close()
