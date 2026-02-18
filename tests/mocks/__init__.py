"""
Mock Infrastructure for MnemoCore Tests
=======================================
Provides in-memory mock implementations of QdrantStore and AsyncRedisStorage
for offline testing without external service dependencies.

Usage:
    from tests.mocks import MockQdrantStore, MockAsyncRedisStorage
"""

from .mock_qdrant import MockQdrantStore
from .mock_redis import MockAsyncRedisStorage

__all__ = ["MockQdrantStore", "MockAsyncRedisStorage"]
