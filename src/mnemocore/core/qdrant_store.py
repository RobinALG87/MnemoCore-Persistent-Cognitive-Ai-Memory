"""
Qdrant Vector Store Layer
=========================
Provides async access to Qdrant for vector storage and similarity search.

Phase 4.3: Temporal Recall - supports time-based filtering and indexing.
"""

from typing import List, Any, Optional, Tuple, Dict
from datetime import datetime
import asyncio
import numpy as np

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

        for collection_name, on_disk in [
            (self.collection_hot, False),
            (self.collection_warm, True)
        ]:
            if await self.client.collection_exists(collection_name):
                # Check for distance mismatch (Phase 4.5 alignment)
                info = await self.client.get_collection(collection_name)
                current_distance = info.config.params.vectors.distance
                if current_distance != models.Distance.DOT:
                    logger.warning(
                        f"Collection {collection_name} has distance {current_distance}, "
                        f"but DOT is required. Recreating collection."
                    )
                    await self.client.delete_collection(collection_name)
                else:
                    continue

            logger.info(f"Creating collection: {collection_name} (DOT)")
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.DOT,
                    on_disk=on_disk
                ),
                quantization_config=quantization_config,
                hnsw_config=models.HnswConfigDiff(
                    m=self.hnsw_m,
                    ef_construct=self.hnsw_ef_construct,
                    on_disk=on_disk
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
        """Async semantic search."""
        try:
            # Transform query to bipolar if it's (0, 1) (Phase 4.5)
            q_vec = np.array(query_vector)
            if np.all((q_vec == 0) | (q_vec == 1)):
                q_vec = q_vec * 2.0 - 1.0
            
            must_conditions = []
            query_filter = None
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

            response = await qdrant_breaker.call(
                self.client.query_points,
                collection_name=collection,
                query=q_vec.tolist(),
                limit=limit,
                query_filter=query_filter,
                search_params=search_params,
            )
            
            # Normalize scores to [0, 1] range (Phase 4.5)
            normalized_points = []
            for hit in response.points:
                sim = 0.5 + (hit.score / (2.0 * self.dim))
                
                normalized_points.append(
                    models.ScoredPoint(
                        id=hit.id,
                        version=hit.version,
                        score=float(np.clip(sim, 0.0, 1.0)),
                        payload=hit.payload,
                        vector=hit.vector
                    )
                )
            return normalized_points
        except CircuitOpenError:
            logger.warning(f"Qdrant search blocked for {collection}: circuit breaker open")
            return []
        except Exception as e:
            logger.error(f"Qdrant search failed for {collection}: {e}")
            raise wrap_storage_exception("qdrant", "search", e)

    async def get_point(self, collection: str, point_id: str) -> Optional[models.Record]:
        """
        Get a single point by ID.

        Returns:
            Record if found, None if not found.
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
            return None
        except CircuitOpenError:
            logger.error(f"Qdrant get_point blocked for {point_id}: circuit breaker open")
            raise
        except Exception as e:
            logger.error(f"Qdrant get_point failed for {point_id}: {e}")
            raise wrap_storage_exception("qdrant", "get_point", e)

    async def get_collection_info(self, collection: str) -> Optional[Any]:
        """Get collection info."""
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
            raise wrap_storage_exception("qdrant", "get_collection_info", e)

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: Any = None,
        with_vectors: bool = False
    ) -> Any:
        """Scroll/Iterate over collection."""
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
            raise wrap_storage_exception("qdrant", "scroll", e)

    async def delete(self, collection: str, point_ids: List[str]):
        """Delete points by ID."""
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

    async def get_temporal_neighbors(
        self,
        collection: str,
        unix_timestamp: int,
        window: int = 2,
    ) -> List[models.Record]:
        """Get memories created within a time window around a timestamp."""
        try:
            start_ts = unix_timestamp - window
            end_ts = unix_timestamp + window

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="unix_timestamp",
                        range=models.Range(gte=start_ts, lte=end_ts),
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

            records = results[0] if results else []
            records.sort(key=lambda r: r.payload.get("unix_timestamp", 0))
            return records
        except Exception as e:
            logger.error(f"Failed to get temporal neighbors: {e}")
            raise wrap_storage_exception("qdrant", "get_temporal_neighbors", e)

    async def get_by_previous_id(
        self,
        collection: str,
        previous_id: str,
    ) -> Optional[models.Record]:
        """Get a memory that follows another (episodic chaining)."""
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
            raise wrap_storage_exception("qdrant", "get_by_previous_id", e)
