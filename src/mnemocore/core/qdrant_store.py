"""
Qdrant Vector Store Layer
=========================
Provides async access to Qdrant for vector storage and similarity search.

Phase 4.3: Temporal Recall - supports time-based filtering and indexing.
"""

from typing import List, Any, Optional, Tuple, Dict
from datetime import datetime
import asyncio

from qdrant_client import AsyncQdrantClient, models
from loguru import logger

from .reliability import qdrant_breaker
from .exceptions import (
    CircuitOpenError,
    StorageConnectionError,
    wrap_storage_exception,
)


class QdrantStore:
    """
    Qdrant Vector Store Layer.
    No longer a singleton - instances should be created via dependency injection.
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        dimensionality: int,
        collection_hot: str = "haim_hot",
        collection_warm: str = "haim_warm",
        binary_quantization: bool = True,
        always_ram: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ):
        self.url = url
        self.api_key = api_key
        self.dim = dimensionality
        self.collection_hot = collection_hot
        self.collection_warm = collection_warm
        self.binary_quantization = binary_quantization
        self.always_ram = always_ram
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.client = AsyncQdrantClient(url=url, api_key=api_key)

    async def ensure_collections(self):
        """
        Ensure HOT and WARM collections exist with proper schema.

        Performs a connectivity ping before collection setup so that startup
        failures produce a clear, actionable error message.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Qdrant is unreachable or connection fails.
        """
        # Phase 4.3: Verify connectivity before attempting collection setup.
        # This converts a cryptic ConnectionRefusedError into a clear message.
        try:
            await self.client.get_collections()
        except Exception as e:
            msg = (
                f"MnemoCore cannot reach Qdrant at '{self.url}'. "
                f"Ensure Qdrant is running and the URL/API key are correct. "
                f"Original error: {e}"
            )
            logger.error(msg)
            raise StorageConnectionError(msg) from e

        try:
            return await qdrant_breaker.call(self._ensure_collections)
        except CircuitOpenError:
            logger.error("Circuit breaker blocked ensure_collections")
            raise
        except Exception as e:
            logger.error(f"Qdrant ensure_collections failed: {e}")
            raise wrap_storage_exception("qdrant", "ensure_collections", e)

    async def _ensure_collections(self):
        # Define BQ config if enabled
        quantization_config = None
        if self.binary_quantization:
            quantization_config = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=self.always_ram
                )
            )

        # Create HOT collection (optimized for latency)
        if not await self.client.collection_exists(self.collection_hot):
            logger.info(f"Creating HOT collection: {self.collection_hot}")
            await self.client.create_collection(
                collection_name=self.collection_hot,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.DOT,
                    on_disk=False
                ),
                quantization_config=quantization_config,
                hnsw_config=models.HnswConfigDiff(
                    m=self.hnsw_m,
                    ef_construct=self.hnsw_ef_construct,
                    on_disk=False
                )
            )

        # Create WARM collection (optimized for scale/disk)
        if not await self.client.collection_exists(self.collection_warm):
            logger.info(f"Creating WARM collection: {self.collection_warm}")
            await self.client.create_collection(
                collection_name=self.collection_warm,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.DOT,
                    on_disk=True
                ),
                quantization_config=quantization_config,
                hnsw_config=models.HnswConfigDiff(
                    m=self.hnsw_m,
                    ef_construct=self.hnsw_ef_construct,
                    on_disk=True
                )
            )

        # Phase 4.3: Create payload index on unix_timestamp for temporal queries
        for collection_name in [self.collection_hot, self.collection_warm]:
            try:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="unix_timestamp",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
                logger.info(f"Created unix_timestamp index on {collection_name}")
            except Exception as e:
                # Index may already exist - that's fine
                if "already exists" not in str(e).lower():
                    logger.debug(f"Timestamp index on {collection_name}: {e}")

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
        score_threshold: float = 0.0,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[models.ScoredPoint]:
        """
        Async semantic search.

        Args:
            collection: Collection name to search.
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            time_range: Optional (start, end) datetime tuple for temporal filtering.
                       Phase 4.3: Enables "memories from last 48 hours" queries.

        Returns:
            List of scored points (empty list on errors).

        Note:
            This method returns an empty list on errors rather than raising,
            as search failures should not crash the calling code.
        """
        try:
            must_conditions = []
            if time_range:
                start_ts = int(time_range[0].timestamp())
                end_ts = int(time_range[1].timestamp())
                must_conditions.append(
                    models.FieldCondition(
                        key="unix_timestamp",
                        range=models.Range(
                            gte=start_ts,
                            lte=end_ts,
                        ),
                    )
                )
            
            if metadata_filter:
                for k, v in metadata_filter.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=k,
                            match=models.MatchValue(value=v)
                        )
                    )
            
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

            # Support for Binary Quantization rescoring (BUG-04)
            search_params = None
            if self.binary_quantization:
                search_params = models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0
                    )
                )

            return await qdrant_breaker.call(
                self.client.search,
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                search_params=search_params,
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

    async def get_collection_info(self, collection: str) -> Optional[Any]:
        """
        Get collection info (e.g. points count).
        Wraps client.get_collection() with reliability and error handling.
        """
        try:
            return await qdrant_breaker.call(
                self.client.get_collection,
                collection_name=collection
            )
        except CircuitOpenError:
            logger.warning(f"Qdrant get_collection_info blocked for {collection}: circuit breaker open")
            return None
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection}: {e}")
            return None

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

    # Phase 4.3: Temporal utilities

    async def get_temporal_neighbors(
        self,
        collection: str,
        unix_timestamp: int,
        window: int = 2,
    ) -> List[models.Record]:
        """
        Get memories created within a time window around a timestamp.

        Args:
            collection: Collection name.
            unix_timestamp: Central timestamp to search around.
            window: Number of seconds to look before and after (default 2s).

        Returns:
            List of records ordered by timestamp.

        Note:
            This enables "what happened just before/after" queries for
            sequential context window feature.
        """
        try:
            # Look for memories in a small time window
            start_ts = unix_timestamp - window
            end_ts = unix_timestamp + window

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="unix_timestamp",
                        range=models.Range(
                            gte=start_ts,
                            lte=end_ts,
                        ),
                    ),
                ]
            )

            results = await qdrant_breaker.call(
                self.client.scroll,
                collection_name=collection,
                limit=10,
                with_vectors=False,
                with_payload=True,
                query_filter=query_filter,
            )

            # Sort by timestamp
            records = results[0] if results else []
            records.sort(key=lambda r: r.payload.get("unix_timestamp", 0))

            return records

        except Exception as e:
            logger.error(f"Failed to get temporal neighbors: {e}")
            return []

    async def get_by_previous_id(
        self,
        collection: str,
        previous_id: str,
    ) -> Optional[models.Record]:
        """
        Get a memory that follows another (episodic chaining).

        Args:
            collection: Collection name.
            previous_id: The previous_id to search for.

        Returns:
            The memory that has this previous_id, or None.
        """
        try:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="previous_id",
                        match=models.MatchValue(value=previous_id),
                    ),
                ]
            )

            results = await qdrant_breaker.call(
                self.client.scroll,
                collection_name=collection,
                limit=1,
                with_vectors=False,
                with_payload=True,
                query_filter=query_filter,
            )

            if results and results[0]:
                return results[0][0]
            return None

        except Exception as e:
            logger.error(f"Failed to get by previous_id: {e}")
            return None
