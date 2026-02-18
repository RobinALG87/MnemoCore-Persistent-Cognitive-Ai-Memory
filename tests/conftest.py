import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# =============================================================================
# Mock Infrastructure Fixtures (Phase 3.5 - Offline Testing Support)
# =============================================================================

@pytest.fixture
def qdrant_store():
    """
    Fixture providing a MockQdrantStore instance for offline testing.

    This mock provides full in-memory implementation of QdrantStore
    without requiring a running Qdrant server.

    Usage:
        async def test_search(qdrant_store):
            await qdrant_store.ensure_collections()
            # ... test code
    """
    from tests.mocks import MockQdrantStore

    store = MockQdrantStore(
        url="mock://localhost:6333",
        dimensionality=1024,
        collection_hot="haim_hot",
        collection_warm="haim_warm"
    )
    return store


@pytest.fixture
def redis_storage():
    """
    Fixture providing a MockAsyncRedisStorage instance for offline testing.

    This mock provides full in-memory implementation of AsyncRedisStorage
    without requiring a running Redis server.

    Usage:
        async def test_storage(redis_storage):
            await redis_storage.store_memory("node1", {"data": "test"})
            result = await redis_storage.retrieve_memory("node1")
    """
    from tests.mocks import MockAsyncRedisStorage

    storage = MockAsyncRedisStorage(
        url="redis://localhost:6379/0",
        stream_key="haim:subconscious"
    )
    return storage


@pytest.fixture
def engine(qdrant_store, redis_storage):
    """
    Fixture providing a mock cognitive engine with mocked storage backends.

    Creates a complete mock engine configuration suitable for unit tests
    that need to interact with the cognitive memory system.

    Usage:
        async def test_engine(engine):
            # Engine has mocked qdrant_store and redis_storage
            await engine.qdrant_store.ensure_collections()
    """
    from dataclasses import dataclass

    @dataclass
    class MockEngine:
        qdrant_store: object
        redis_storage: object
        config: dict

        async def initialize(self):
            await self.qdrant_store.ensure_collections()
            return True

        async def shutdown(self):
            await self.qdrant_store.close()
            await self.redis_storage.close()

    mock_config = {
        "qdrant_url": "mock://localhost:6333",
        "redis_url": "redis://localhost:6379/0",
        "dimensionality": 1024,
    }

    return MockEngine(
        qdrant_store=qdrant_store,
        redis_storage=redis_storage,
        config=mock_config
    )


# =============================================================================
# Legacy Mock Fixtures (for backward compatibility)
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def mock_hardware_dependencies():
    """Globally mock Qdrant and Redis to prevent hangs during testing."""
    # Ensure modules are imported so patch can find them in mnemocore.core
    import mnemocore.core.async_storage
    import mnemocore.core.qdrant_store

    # 1. Mock Redis client for AsyncRedisStorage
    mock_redis_client = MagicMock()
    mock_redis_client.ping = AsyncMock(return_value=True)
    mock_redis_client.get = AsyncMock(return_value=None)
    mock_redis_client.set = AsyncMock(return_value=True)
    mock_redis_client.setex = AsyncMock(return_value=True)
    mock_redis_client.delete = AsyncMock(return_value=1)
    mock_redis_client.mget = AsyncMock(return_value=[])
    mock_redis_client.zadd = AsyncMock(return_value=1)
    mock_redis_client.zrange = AsyncMock(return_value=[])
    mock_redis_client.zrem = AsyncMock(return_value=1)
    mock_redis_client.xadd = AsyncMock(return_value="1234567890-0")
    mock_redis_client.xread = AsyncMock(return_value=[])
    mock_redis_client.xreadgroup = AsyncMock(return_value=[])
    mock_redis_client.xgroup_create = AsyncMock(return_value=True)
    mock_redis_client.xack = AsyncMock(return_value=True)

    # Pipeline mock
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.incr = MagicMock()
    mock_pipeline.expire = MagicMock()
    mock_pipeline.execute = AsyncMock(return_value=[1, True])
    mock_redis_client.pipeline.return_value = mock_pipeline

    # Create a mock AsyncRedisStorage instance
    mock_redis_storage = MagicMock(spec=mnemocore.core.async_storage.AsyncRedisStorage)
    mock_redis_storage.redis_client = mock_redis_client
    mock_redis_storage.check_health = AsyncMock(return_value=True)
    mock_redis_storage.store_memory = AsyncMock(return_value=None)
    mock_redis_storage.retrieve_memory = AsyncMock(return_value=None)
    mock_redis_storage.batch_retrieve = AsyncMock(return_value=[])
    mock_redis_storage.delete_memory = AsyncMock(return_value=None)
    mock_redis_storage.get_eviction_candidates = AsyncMock(return_value=[])
    mock_redis_storage.update_ltp = AsyncMock(return_value=None)
    mock_redis_storage.publish_event = AsyncMock(return_value=None)
    mock_redis_storage.close = AsyncMock(return_value=None)

    # 2. Mock Qdrant client
    mock_qdrant_client = MagicMock()
    mock_qdrant_client.collection_exists = AsyncMock(return_value=False)
    mock_qdrant_client.create_collection = AsyncMock(return_value=None)
    mock_qdrant_client.upsert = AsyncMock(return_value=None)
    mock_qdrant_client.search = AsyncMock(return_value=[])
    mock_qdrant_client.retrieve = AsyncMock(return_value=[])
    mock_qdrant_client.scroll = AsyncMock(return_value=([], None))
    mock_qdrant_client.delete = AsyncMock(return_value=None)
    mock_qdrant_client.get_collection = AsyncMock()
    mock_qdrant_client.close = AsyncMock(return_value=None)

    # Create a mock QdrantStore instance
    mock_qdrant_instance = MagicMock(spec=mnemocore.core.qdrant_store.QdrantStore)
    mock_qdrant_instance.client = mock_qdrant_client
    mock_qdrant_instance.ensure_collections = AsyncMock(return_value=None)
    mock_qdrant_instance.upsert = AsyncMock(return_value=None)
    mock_qdrant_instance.search = AsyncMock(return_value=[])
    mock_qdrant_instance.get_point = AsyncMock(return_value=None)
    mock_qdrant_instance.scroll = AsyncMock(return_value=([], None))
    mock_qdrant_instance.delete = AsyncMock(return_value=None)
    mock_qdrant_instance.close = AsyncMock(return_value=None)

    # Patch the Container to return mocked instances
    from mnemocore.core import container as container_module

    original_build_container = container_module.build_container

    def mock_build_container(config):
        container = MagicMock()
        container.config = config
        container.redis_storage = mock_redis_storage
        container.qdrant_store = mock_qdrant_instance
        return container

    container_patch = patch.object(container_module, 'build_container', side_effect=mock_build_container)
    container_patch.start()

    # Patch _initialize_from_pool instead of __init__ to allow constructor to run
    redis_init_patch = patch.object(
        mnemocore.core.async_storage.AsyncRedisStorage,
        '_initialize_from_pool',
        return_value=None
    )
    redis_init_patch.start()

    # Patch AsyncQdrantClient instead of __init__
    qdrant_client_patch = patch('mnemocore.core.qdrant_store.AsyncQdrantClient')
    qdrant_client_patch.start()

    yield (mock_qdrant_instance, mock_redis_storage)

    # Stop all patches
    container_patch.stop()
    redis_init_patch.stop()
    qdrant_client_patch.stop()


@pytest.fixture(autouse=True)
def clean_config():
    """Reset config state between tests."""
    from mnemocore.core.config import reset_config
    reset_config()
    yield
    reset_config()


@pytest.fixture
def mock_container():
    """Create a mock container for testing."""
    from mnemocore.core.config import get_config

    config = get_config()

    mock_redis_client = MagicMock()
    mock_redis_client.ping = AsyncMock(return_value=True)
    mock_redis_client.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_redis_client.pipeline.return_value)
    mock_redis_client.pipeline.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_redis_client.pipeline.return_value.execute = AsyncMock(return_value=[1, True])

    mock_redis_storage = MagicMock()
    mock_redis_storage.redis_client = mock_redis_client
    mock_redis_storage.check_health = AsyncMock(return_value=True)
    mock_redis_storage.publish_event = AsyncMock(return_value=None)
    mock_redis_storage.store_memory = AsyncMock(return_value=None)
    mock_redis_storage.retrieve_memory = AsyncMock(return_value=None)
    mock_redis_storage.delete_memory = AsyncMock(return_value=None)
    mock_redis_storage.close = AsyncMock(return_value=None)

    mock_qdrant = MagicMock()
    mock_qdrant.ensure_collections = AsyncMock(return_value=None)
    mock_qdrant.upsert = AsyncMock(return_value=None)
    mock_qdrant.search = AsyncMock(return_value=[])
    mock_qdrant.get_point = AsyncMock(return_value=None)
    mock_qdrant.scroll = AsyncMock(return_value=([], None))
    mock_qdrant.delete = AsyncMock(return_value=None)
    mock_qdrant.close = AsyncMock(return_value=None)

    container = MagicMock()
    container.config = config
    container.redis_storage = mock_redis_storage
    container.qdrant_store = mock_qdrant

    return container
