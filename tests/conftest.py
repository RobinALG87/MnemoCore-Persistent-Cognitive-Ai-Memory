import sys
import os
import asyncio
from pathlib import Path
from typing import Optional, Generator, AsyncGenerator

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# Ensure local src/ package imports work without editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """
    Configure custom pytest markers.

    This function is called by pytest at startup to register custom markers.
    """
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires Docker or external services)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running (use pytest -m 'not slow' to skip)"
    )
    config.addinivalue_line(
        "markers",
        "requires_redis: mark test as requiring a running Redis instance"
    )
    config.addinivalue_line(
        "markers",
        "requires_qdrant: mark test as requiring a running Qdrant instance"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle markers automatically.

    - Skip integration tests unless --run-integration flag is passed
    - Skip slow tests unless --run-slow flag is passed
    - Skip Docker-requiring tests if services are not available
    """
    # Check for CLI flags
    run_integration = config.getoption("--run-integration", default=False)
    run_slow = config.getoption("--run-slow", default=False)

    # Check for Docker services availability
    redis_available = _check_redis_available()
    qdrant_available = _check_qdrant_available()

    skip_integration = pytest.mark.skip(
        reason="Integration test skipped. Use --run-integration to run."
    )
    skip_slow = pytest.mark.skip(
        reason="Slow test skipped. Use --run-slow to run."
    )
    skip_redis = pytest.mark.skip(
        reason="Redis not available. Start Redis with: docker run -p 6379:6379 redis"
    )
    skip_qdrant = pytest.mark.skip(
        reason="Qdrant not available. Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant"
    )

    for item in items:
        # Handle integration marker
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)

        # Handle slow marker
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)

        # Handle service-specific markers
        if "requires_redis" in item.keywords and not redis_available:
            item.add_marker(skip_redis)

        if "requires_qdrant" in item.keywords and not qdrant_available:
            item.add_marker(skip_qdrant)


def _check_redis_available() -> bool:
    """Check if Redis is available for integration tests."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, socket_timeout=2)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


def _check_qdrant_available() -> bool:
    """Check if Qdrant is available for integration tests."""
    try:
        import requests
        response = requests.get("http://localhost:6333/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires Docker services)"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


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


# =============================================================================
# Integration Test Fixtures (Task 16.5)
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for async tests.

    This fixture ensures a consistent event loop is used across all async tests.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_test_dir(tmp_path) -> Path:
    """
    Create a temporary directory for test data.

    This fixture provides a clean temporary directory for each test
    that can be used for file-based operations.
    """
    test_dir = tmp_path / "mnemocore_test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
async def integration_engine(temp_test_dir):
    """
    Create a real HAIMEngine for integration testing without mocking.

    This fixture creates a fully functional engine instance that uses
    local file storage instead of external services. Suitable for
    testing actual engine behavior without Docker dependencies.

    Note: This engine does NOT use Qdrant or Redis. All storage is
    local file-based for offline testing.

    Usage:
        @pytest.mark.integration
        async def test_real_engine(integration_engine):
            await integration_engine.store("test content")
            results = await integration_engine.query("test")
    """
    from mnemocore.core.engine import HAIMEngine
    from mnemocore.core.config import get_config

    config = get_config()

    with patch("mnemocore.core.engine.AsyncQdrantClient"):
        engine = HAIMEngine(
            dimension=1024,  # Smaller for faster tests
            config=config,
        )

        # Disable external services
        engine.tier_manager.use_qdrant = False
        engine.tier_manager.warm_path = temp_test_dir / "warm"
        engine.tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

        await engine.initialize()

        yield engine

        await engine.close()


@pytest.fixture
async def real_qdrant_store():
    """
    Create a real QdrantStore for integration tests.

    This fixture requires a running Qdrant instance (Docker).
    Tests using this fixture should be marked with @pytest.mark.requires_qdrant

    Skips automatically if Qdrant is not available.

    Usage:
        @pytest.mark.integration
        @pytest.mark.requires_qdrant
        async def test_qdrant(real_qdrant_store):
            await real_qdrant_store.ensure_collections()
    """
    pytest.importorskip("qdrant_client")

    if not _check_qdrant_available():
        pytest.skip("Qdrant not available at localhost:6333")

    from mnemocore.core.qdrant_store import QdrantStore
    from mnemocore.core.config import get_config

    config = get_config()
    store = QdrantStore(
        url="http://localhost:6333",
        dimensionality=config.dimensionality,
        collection_hot=f"test_hot_{os.getpid()}",
        collection_warm=f"test_warm_{os.getpid()}",
    )

    await store.ensure_collections()

    yield store

    # Cleanup: Delete test collections
    try:
        await store.client.delete_collection(store.collection_hot)
        await store.client.delete_collection(store.collection_warm)
    except Exception:
        pass

    await store.close()


@pytest.fixture
async def real_redis_storage():
    """
    Create a real AsyncRedisStorage for integration tests.

    This fixture requires a running Redis instance (Docker).
    Tests using this fixture should be marked with @pytest.mark.requires_redis

    Skips automatically if Redis is not available.

    Usage:
        @pytest.mark.integration
        @pytest.mark.requires_redis
        async def test_redis(real_redis_storage):
            await real_redis_storage.store_memory("key", {"data": "value"})
    """
    pytest.importorskip("redis.asyncio")

    if not _check_redis_available():
        pytest.skip("Redis not available at localhost:6379")

    from mnemocore.core.async_storage import AsyncRedisStorage

    storage = AsyncRedisStorage(
        url="redis://localhost:6379/15",  # Use DB 15 for tests
        stream_key="test:subconscious"
    )

    await storage._initialize_from_pool()

    yield storage

    # Cleanup: Flush test database
    try:
        await storage.redis_client.flushdb()
    except Exception:
        pass

    await storage.close()


@pytest.fixture
def isolated_event_bus():
    """
    Create an isolated EventBus for testing without singleton interference.

    This fixture ensures tests don't interfere with each other by providing
    a fresh EventBus instance that is properly cleaned up.

    Usage:
        @pytest.mark.asyncio
        async def test_events(isolated_event_bus):
            await isolated_event_bus.start()
            # ... test code
            await isolated_event_bus.stop()
    """
    from mnemocore.events.event_bus import EventBus, reset_event_bus

    # Reset singleton state
    reset_event_bus()

    bus = EventBus(max_queue_size=100, delivery_timeout=5.0)

    yield bus

    # Cleanup
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(bus.shutdown())
        else:
            loop.run_until_complete(bus.shutdown())
    except Exception:
        pass

    reset_event_bus()


@pytest.fixture
def sample_memory_node():
    """
    Create a sample MemoryNode for testing.

    Provides a pre-configured MemoryNode with sensible defaults.

    Usage:
        def test_node(sample_memory_node):
            assert sample_memory_node.content == "Test memory content"
    """
    from mnemocore.core.node import MemoryNode
    from mnemocore.core.binary_hdv import BinaryHDV
    import numpy as np

    dimension = 1024
    packed = np.random.randint(0, 256, size=dimension // 8, dtype=np.uint8)
    hdv = BinaryHDV(packed, dimension)

    node = MemoryNode(
        id=f"test_node_{os.getpid()}_{id(object)}",
        content="Test memory content for integration testing",
        hdv=hdv,
        created_at=datetime.now(timezone.utc),
        metadata={
            "test": True,
            "source": "conftest_fixture",
        },
        ltp_strength=0.5,
        tier="hot",
    )

    return node


@pytest.fixture
def sample_episode():
    """
    Create a sample Episode for testing episodic memory.

    Usage:
        def test_episode(sample_episode):
            assert sample_episode.goal == "Test goal"
    """
    from mnemocore.core.memory_model import Episode, EpisodeEvent

    episode = Episode(
        id=f"test_ep_{os.getpid()}",
        agent_id="test_agent",
        goal="Test episode goal for integration testing",
        created_at=datetime.now(timezone.utc),
        events=[
            EpisodeEvent(
                kind="observation",
                content="Test observation",
                timestamp=datetime.now(timezone.utc),
            ),
            EpisodeEvent(
                kind="action",
                content="Test action",
                timestamp=datetime.now(timezone.utc),
            ),
        ],
        outcome="success",
        reward=1.0,
    )

    return episode


# =============================================================================
# Session-Scoped Fixtures for Performance
# =============================================================================

@pytest.fixture(scope="session")
def binary_encoder():
    """
    Create a session-scoped TextEncoder for binary HDV operations.

    Sharing the encoder across tests improves performance by avoiding
    repeated initialization.
    """
    from mnemocore.core.binary_hdv import TextEncoder

    encoder = TextEncoder(dimension=1024)
    return encoder


@pytest.fixture(scope="session")
def sample_hdv(binary_encoder):
    """
    Create a sample BinaryHDV for testing.

    Uses the session-scoped encoder for consistency.
    """
    return binary_encoder.encode("sample test vector for integration testing")


# =============================================================================
# Helper Imports (to avoid missing imports in fixtures)
# =============================================================================

from datetime import datetime, timezone
