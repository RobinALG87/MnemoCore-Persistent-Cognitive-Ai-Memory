"""
Mock Qdrant Store
=================
In-memory mock implementation of QdrantStore for offline testing.

Implements all public methods from mnemocore.core.qdrant_store.QdrantStore
using Python dictionaries for storage, enabling full test isolation.
"""

from typing import List, Any, Optional, Dict
from dataclasses import dataclass, field
import asyncio
import numpy as np
from loguru import logger


@dataclass
class MockPointStruct:
    """Mock of qdrant_client.models.PointStruct"""
    id: str
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None


@dataclass
class MockScoredPoint:
    """Mock of qdrant_client.models.ScoredPoint"""
    id: str
    score: float
    version: int = 0
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None


@dataclass
class MockRecord:
    """Mock of qdrant_client.models.Record"""
    id: str
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None


class MockQdrantStore:
    """
    In-memory mock implementation of QdrantStore.

    Provides full implementation of all public methods using dictionaries:
    - ensure_collections()
    - upsert()
    - search()
    - get_point()
    - scroll()
    - delete()
    - close()

    Supports multiple collections and cosine similarity search.
    """

    def __init__(
        self,
        url: str = "mock://localhost:6333",
        api_key: Optional[str] = None,
        dimensionality: int = 1024,
        collection_hot: str = "haim_hot",
        collection_warm: str = "haim_warm",
        binary_quantization: bool = True,
        always_ram: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ):
        """Initialize mock store with configuration matching real QdrantStore."""
        self.url = url
        self.api_key = api_key
        self.dim = dimensionality
        self.collection_hot = collection_hot
        self.collection_warm = collection_warm
        self.binary_quantization = binary_quantization
        self.always_ram = always_ram
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct

        # In-memory storage: {collection_name: {point_id: MockPointStruct}}
        self._collections: Dict[str, Dict[str, MockPointStruct]] = {}
        self._closed = False

        # Mock client attribute for compatibility
        self.client = self

    async def ensure_collections(self):
        """
        Ensure HOT and WARM collections exist.

        Creates empty dictionaries for each collection if they don't exist.
        """
        if self._closed:
            raise RuntimeError("Store is closed")

        for collection_name in [self.collection_hot, self.collection_warm]:
            if collection_name not in self._collections:
                self._collections[collection_name] = {}
                logger.info(f"[MockQdrant] Created collection: {collection_name}")

    async def upsert(self, collection: str, points: List[Any]):
        """
        Async batch upsert.

        Args:
            collection: Collection name
            points: List of PointStruct objects with id, vector, and payload
        """
        if self._closed:
            raise RuntimeError("Store is closed")

        if collection not in self._collections:
            self._collections[collection] = {}

        for point in points:
            # Handle both MockPointStruct and real PointStruct
            point_id = str(point.id)
            vector = list(point.vector) if hasattr(point, 'vector') else []
            payload = dict(point.payload) if point.payload else {}

            self._collections[collection][point_id] = MockPointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )

        logger.debug(f"[MockQdrant] Upserted {len(points)} points to {collection}")

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0
    ) -> List[MockScoredPoint]:
        """
        Async semantic search using cosine similarity.

        Args:
            collection: Collection name
            query_vector: Query embedding
            limit: Maximum number of results
            score_threshold: Minimum score threshold

        Returns:
            List of MockScoredPoint sorted by score (descending)
        """
        if self._closed:
            return []

        if collection not in self._collections:
            return []

        query_arr = np.array(query_vector)
        query_norm = np.linalg.norm(query_arr)

        if query_norm == 0:
            return []

        results = []

        for point_id, point in self._collections[collection].items():
            if not point.vector:
                continue

            point_arr = np.array(point.vector)
            point_norm = np.linalg.norm(point_arr)

            if point_norm == 0:
                continue

            # Cosine similarity
            similarity = float(np.dot(query_arr, point_arr) / (query_norm * point_norm))

            if similarity >= score_threshold:
                results.append(MockScoredPoint(
                    id=point_id,
                    score=similarity,
                    payload=dict(point.payload) if point.payload else {},
                    vector=list(point.vector)
                ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:limit]

    async def get_point(self, collection: str, point_id: str) -> Optional[MockRecord]:
        """
        Get a single point by ID.

        Args:
            collection: Collection name
            point_id: Point identifier

        Returns:
            MockRecord if found, None otherwise
        """
        if self._closed:
            raise RuntimeError("Store is closed")

        if collection not in self._collections:
            return None

        point = self._collections[collection].get(str(point_id))
        if point is None:
            return None

        return MockRecord(
            id=point.id,
            payload=dict(point.payload) if point.payload else {},
            vector=list(point.vector) if point.vector else None
        )

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: Any = None,
        with_vectors: bool = False
    ) -> Any:
        """
        Scroll/Iterate over collection (for consolidation).

        Args:
            collection: Collection name
            limit: Maximum number of results
            offset: Offset for pagination (index in this mock)
            with_vectors: Whether to include vectors

        Returns:
            Tuple of (points, next_offset)
        """
        if self._closed:
            return [], None

        if collection not in self._collections:
            return [], None

        # Convert to list for indexed access
        all_points = list(self._collections[collection].values())

        # Handle offset as integer index
        start_idx = int(offset) if offset is not None else 0

        # Get slice
        points_slice = all_points[start_idx:start_idx + limit]

        # Convert to records
        records = []
        for point in points_slice:
            records.append(MockRecord(
                id=point.id,
                payload=dict(point.payload) if point.payload else {},
                vector=list(point.vector) if point.vector and with_vectors else None
            ))

        # Calculate next offset
        next_offset = start_idx + len(records) if len(records) == limit else None

        return records, next_offset

    async def delete(self, collection: str, point_ids: List[str]):
        """
        Delete points by ID.

        Args:
            collection: Collection name
            point_ids: List of point identifiers to delete
        """
        if self._closed:
            raise RuntimeError("Store is closed")

        if collection not in self._collections:
            return

        for point_id in point_ids:
            self._collections[collection].pop(str(point_id), None)

        logger.debug(f"[MockQdrant] Deleted {len(point_ids)} points from {collection}")

    async def close(self):
        """Close the store (clear all data in mock)."""
        self._closed = True
        logger.debug("[MockQdrant] Store closed")

    # --- Utility methods for testing ---

    def _get_collection_size(self, collection: str) -> int:
        """Get number of points in a collection (for testing assertions)."""
        if collection not in self._collections:
            return 0
        return len(self._collections[collection])

    def _clear_all(self):
        """Clear all collections (for test cleanup)."""
        self._collections.clear()
        self._closed = False

    def _get_point_raw(self, collection: str, point_id: str) -> Optional[MockPointStruct]:
        """Get raw point data (for testing assertions)."""
        if collection not in self._collections:
            return None
        return self._collections[collection].get(str(point_id))
