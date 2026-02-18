"""
Tests for Dependency Injection Migration
=========================================
Verifies that the singleton pattern has been properly removed
and replaced with dependency injection.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestAsyncRedisStorageDI:
    """Tests for AsyncRedisStorage dependency injection."""

    def test_no_get_instance_method(self):
        """AsyncRedisStorage should not have get_instance class method."""
        from src.core.async_storage import AsyncRedisStorage
        assert not hasattr(AsyncRedisStorage, 'get_instance'), \
            "AsyncRedisStorage should not have get_instance method"

    def test_constructor_accepts_parameters(self):
        """AsyncRedisStorage constructor should accept explicit parameters."""
        from src.core.async_storage import AsyncRedisStorage

        # Create with explicit parameters to verify they work
        storage = AsyncRedisStorage(
            url="redis://test:6379/0",
            stream_key="test:stream",
            max_connections=5,
            socket_timeout=10,
            password="testpass",
        )

        # Verify attributes are set
        assert storage.stream_key == "test:stream"

    def test_constructor_with_mock_client(self):
        """AsyncRedisStorage should accept a mock client for testing."""
        from src.core.async_storage import AsyncRedisStorage

        mock_client = MagicMock()
        storage = AsyncRedisStorage(client=mock_client)

        assert storage.redis_client is mock_client


class TestQdrantStoreDI:
    """Tests for QdrantStore dependency injection."""

    def test_no_get_instance_method(self):
        """QdrantStore should not have get_instance class method."""
        from src.core.qdrant_store import QdrantStore
        assert not hasattr(QdrantStore, 'get_instance'), \
            "QdrantStore should not have get_instance method"

    def test_constructor_accepts_parameters(self):
        """QdrantStore constructor should accept explicit parameters."""
        from src.core.qdrant_store import QdrantStore

        store = QdrantStore(
            url="http://test:6333",
            api_key="test-key",
            dimensionality=8192,
            collection_hot="test_hot",
            collection_warm="test_warm",
        )

        assert store.url == "http://test:6333"
        assert store.api_key == "test-key"
        assert store.dim == 8192
        assert store.collection_hot == "test_hot"
        assert store.collection_warm == "test_warm"


class TestContainer:
    """Tests for the dependency injection container."""

    def test_container_exists(self):
        """Container module should exist and be importable."""
        from src.core.container import Container, build_container
        assert Container is not None
        assert build_container is not None

    def test_build_container_creates_dependencies(self):
        """build_container should create all required dependencies."""
        from src.core.container import build_container
        from src.core.config import HAIMConfig

        # Create a minimal config
        config = HAIMConfig()

        with patch('src.core.container.AsyncRedisStorage') as mock_redis_class, \
             patch('src.core.container.QdrantStore') as mock_qdrant_class:

            mock_redis_class.return_value = MagicMock()
            mock_qdrant_class.return_value = MagicMock()

            container = build_container(config)

        assert container.config is config
        assert container.redis_storage is not None
        assert container.qdrant_store is not None

    def test_container_dataclass_fields(self):
        """Container should have expected fields."""
        from src.core.container import Container
        from src.core.config import HAIMConfig

        config = HAIMConfig()
        container = Container(config=config)

        assert hasattr(container, 'config')
        assert hasattr(container, 'redis_storage')
        assert hasattr(container, 'qdrant_store')


class TestTierManagerDI:
    """Tests for TierManager dependency injection."""

    def test_constructor_accepts_config(self):
        """TierManager constructor should accept config parameter."""
        from src.core.tier_manager import TierManager
        from src.core.config import HAIMConfig

        config = HAIMConfig()

        with patch('src.core.tier_manager.HNSW_AVAILABLE', False), \
             patch('src.core.tier_manager.FAISS_AVAILABLE', False):
            manager = TierManager(config=config)

        assert manager.config is config

    def test_constructor_accepts_qdrant_store(self):
        """TierManager constructor should accept qdrant_store parameter."""
        from src.core.tier_manager import TierManager
        from src.core.config import HAIMConfig

        config = HAIMConfig()
        mock_qdrant = MagicMock()

        with patch('src.core.tier_manager.HNSW_AVAILABLE', False), \
             patch('src.core.tier_manager.FAISS_AVAILABLE', False):
            manager = TierManager(config=config, qdrant_store=mock_qdrant)

        assert manager.qdrant is mock_qdrant
        assert manager.use_qdrant is True


class TestHAIMEngineDI:
    """Tests for HAIMEngine dependency injection."""

    def test_constructor_accepts_config(self):
        """HAIMEngine constructor should accept config parameter."""
        from src.core.engine import HAIMEngine
        from src.core.config import HAIMConfig

        config = HAIMConfig()

        # Patch at tier_manager level since that's where HNSW/FAISS is used
        with patch('src.core.tier_manager.HNSW_AVAILABLE', False), \
             patch('src.core.tier_manager.FAISS_AVAILABLE', False):
            engine = HAIMEngine(config=config)

        assert engine.config is config

    def test_constructor_accepts_tier_manager(self):
        """HAIMEngine constructor should accept tier_manager parameter."""
        from src.core.engine import HAIMEngine
        from src.core.config import HAIMConfig
        from src.core.tier_manager import TierManager

        config = HAIMConfig()

        with patch('src.core.tier_manager.HNSW_AVAILABLE', False), \
             patch('src.core.tier_manager.FAISS_AVAILABLE', False):
            tier_manager = TierManager(config=config)
            engine = HAIMEngine(config=config, tier_manager=tier_manager)

        assert engine.tier_manager is tier_manager


class TestConsolidationWorkerDI:
    """Tests for ConsolidationWorker dependency injection."""

    def test_constructor_accepts_storage(self):
        """ConsolidationWorker constructor should accept storage parameter."""
        from src.core.consolidation_worker import ConsolidationWorker

        mock_storage = MagicMock()
        mock_tier_manager = MagicMock()

        worker = ConsolidationWorker(
            storage=mock_storage,
            tier_manager=mock_tier_manager,
        )

        assert worker.storage is mock_storage
        assert worker.tier_manager is mock_tier_manager


class TestNoSingletonPattern:
    """Tests to ensure singleton pattern is fully removed."""

    def test_no_singleton_instances(self):
        """Classes should not have _instance class attribute for singletons."""
        from src.core.async_storage import AsyncRedisStorage
        from src.core.qdrant_store import QdrantStore

        # _instance is the typical singleton storage attribute
        assert not hasattr(AsyncRedisStorage, '_instance') or \
               AsyncRedisStorage._instance is None or \
               '_instance' not in AsyncRedisStorage.__dict__

        # Note: QdrantStore might have _instance from object base,
        # but shouldn't have it defined explicitly for singleton use
        if hasattr(QdrantStore, '_instance'):
            # Check it's not being used as singleton storage
            assert '_instance' not in QdrantStore.__dict__ or \
                   QdrantStore.__dict__['_instance'] is None

    def test_multiple_instances_independent(self):
        """Creating multiple instances should work independently."""
        from src.core.async_storage import AsyncRedisStorage

        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        storage1 = AsyncRedisStorage(client=mock_client1)
        storage2 = AsyncRedisStorage(client=mock_client2)

        # Each should have its own client
        assert storage1.redis_client is mock_client1
        assert storage2.redis_client is mock_client2
        assert storage1 is not storage2

