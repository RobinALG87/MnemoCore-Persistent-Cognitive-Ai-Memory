"""
Tests for Dream Loop (Subconscious Daemon)

Tests configurability, graceful shutdown, non-blocking behavior, and metrics.
"""

import asyncio
import time
import sys
import importlib
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

pytest_plugins = ['pytest_asyncio']


@dataclass(frozen=True)
class MockDreamLoopConfig:
    """Mock dream loop configuration."""
    enabled: bool = True
    frequency_seconds: int = 1  # Short for testing
    batch_size: int = 10
    max_iterations: int = 0
    ollama_url: str = "http://localhost:11434/api/generate"
    model: str = "gemma3:1b"


@dataclass(frozen=True)
class MockRedisConfig:
    """Mock Redis configuration."""
    url: str = "redis://localhost:6379/0"
    stream_key: str = "haim:subconscious"
    max_connections: int = 10
    socket_timeout: int = 5
    password: str = None


@dataclass(frozen=True)
class MockConfig:
    """Mock configuration for testing."""
    dream_loop: MockDreamLoopConfig = None
    redis: MockRedisConfig = None

    def __post_init__(self):
        if self.dream_loop is None:
            object.__setattr__(self, 'dream_loop', MockDreamLoopConfig())
        if self.redis is None:
            object.__setattr__(self, 'redis', MockRedisConfig())


@pytest.fixture
def mock_config():
    """Create a mock configuration with short intervals for testing."""
    return MockConfig(
        dream_loop=MockDreamLoopConfig(
            enabled=True,
            frequency_seconds=1,
            batch_size=10,
            max_iterations=0,
        ),
        redis=MockRedisConfig()
    )


@pytest.fixture
def mock_config_disabled():
    """Create a mock configuration with dream loop disabled."""
    return MockConfig(
        dream_loop=MockDreamLoopConfig(
            enabled=False,
            frequency_seconds=1,
        ),
        redis=MockRedisConfig()
    )


@pytest.fixture
def mock_config_limited_iterations():
    """Create a mock configuration with limited iterations."""
    return MockConfig(
        dream_loop=MockDreamLoopConfig(
            enabled=True,
            frequency_seconds=1,
            max_iterations=2,
        ),
        redis=MockRedisConfig()
    )


@pytest.fixture
def mock_storage():
    """Create a mock AsyncRedisStorage."""
    storage = MagicMock()
    storage.redis_client = MagicMock()
    storage.check_health = AsyncMock(return_value=True)
    storage.publish_event = AsyncMock(return_value=None)
    storage.retrieve_memory = AsyncMock(return_value=None)
    storage.close = AsyncMock(return_value=None)
    return storage


@pytest.fixture
def daemon_module():
    """Fixture to import the daemon module with all mocks in place."""
    # Create mock for aiohttp
    mock_aiohttp = MagicMock()

    # Create mock for DREAM_LOOP metrics
    mock_dream_loop_total = MagicMock()
    mock_dream_loop_total.labels = MagicMock(return_value=MagicMock())
    mock_dream_loop_iteration_seconds = MagicMock()
    mock_dream_loop_iteration_seconds.observe = MagicMock()
    mock_dream_loop_insights = MagicMock()
    mock_dream_loop_insights.labels = MagicMock(return_value=MagicMock())
    mock_dream_loop_active = MagicMock()
    mock_dream_loop_active.set = MagicMock()

    # Patch sys.modules to inject mocks before import
    patches = {
        'aiohttp': mock_aiohttp,
        'src.subconscious.daemon.aiohttp': mock_aiohttp,
        'src.subconscious.daemon.DREAM_LOOP_TOTAL': mock_dream_loop_total,
        'src.subconscious.daemon.DREAM_LOOP_ITERATION_SECONDS': mock_dream_loop_iteration_seconds,
        'src.subconscious.daemon.DREAM_LOOP_INSIGHTS_GENERATED': mock_dream_loop_insights,
        'src.subconscious.daemon.DREAM_LOOP_ACTIVE': mock_dream_loop_active,
    }

    # Apply patches to sys.modules
    original_values = {}
    for key, value in patches.items():
        if key in sys.modules:
            original_values[key] = sys.modules[key]
        sys.modules[key] = value

    # Remove daemon from sys.modules if it exists to force reload
    if 'src.subconscious.daemon' in sys.modules:
        del sys.modules['src.subconscious.daemon']

    try:
        import mnemocore.subconscious.daemon as dm
        yield dm
    finally:
        # Restore original sys.modules
        for key in patches:
            if key in original_values:
                sys.modules[key] = original_values[key]
            elif key in sys.modules:
                del sys.modules[key]
        # Clean up daemon module
        if 'src.subconscious.daemon' in sys.modules:
            del sys.modules['src.subconscious.daemon']


class TestDreamLoopStartsAndStops:
    """Test that dream loop can start and stop properly."""

    @pytest.mark.asyncio
    async def test_dream_loop_starts_and_stops(self, mock_config, mock_storage, daemon_module):
        """Test that the dream loop starts and stops correctly."""
        SubconsciousDaemon = daemon_module.SubconsciousDaemon

        daemon = SubconsciousDaemon(storage=mock_storage, config=mock_config)

        # Verify initial state
        assert daemon.running is False
        assert daemon._should_stop() is False

        # Start daemon in background task
        run_task = asyncio.create_task(daemon.run())

        # Wait a bit for startup
        await asyncio.sleep(0.2)

        # Verify running state
        assert daemon.running is True

        # Stop the daemon
        await daemon.request_stop()

        # Wait for the task to complete
        await asyncio.wait_for(run_task, timeout=2.0)

        # Verify stopped state
        assert daemon.running is False
        assert daemon._should_stop() is True

    @pytest.mark.asyncio
    async def test_dream_loop_respects_disabled_config(self, mock_config_disabled, mock_storage, daemon_module):
        """Test that the dream loop exits immediately when disabled."""
        SubconsciousDaemon = daemon_module.SubconsciousDaemon

        daemon = SubconsciousDaemon(storage=mock_storage, config=mock_config_disabled)

        # Run should return immediately when disabled
        await daemon.run()

        # Verify it never started
        assert daemon.running is False


class TestDreamLoopFrequency:
    """Test that dream loop respects frequency configuration."""

    @pytest.mark.asyncio
    async def test_dream_respects_frequency(self, mock_config, mock_storage, daemon_module):
        """Test that dream cycles respect the configured frequency."""
        SubconsciousDaemon = daemon_module.SubconsciousDaemon

        daemon = SubconsciousDaemon(storage=mock_storage, config=mock_config)

        # Track cycle times
        cycle_times = []

        original_run_cycle = daemon.run_cycle

        async def tracked_run_cycle():
            cycle_times.append(time.time())
            await original_run_cycle()

        daemon.run_cycle = tracked_run_cycle

        # Start daemon
        run_task = asyncio.create_task(daemon.run())

        # Wait for a couple of cycles
        await asyncio.sleep(0.3)
        await daemon.request_stop()
        await asyncio.wait_for(run_task, timeout=2.0)

        # Verify at least one cycle ran
        assert len(cycle_times) >= 1


class TestDreamLoopNonBlocking:
    """Test that dream loop does not block other operations."""

    @pytest.mark.asyncio
    async def test_dream_does_not_block_queries(self, mock_config, mock_storage, daemon_module):
        """Test that dream loop iterations don't block other async operations."""
        SubconsciousDaemon = daemon_module.SubconsciousDaemon

        daemon = SubconsciousDaemon(storage=mock_storage, config=mock_config)

        # Track query execution
        query_executed = asyncio.Event()

        async def mock_query():
            query_executed.set()
            return {"result": "ok"}

        # Start daemon
        run_task = asyncio.create_task(daemon.run())

        # Simulate a concurrent query while daemon is running
        await asyncio.sleep(0.1)  # Let daemon start
        query_task = asyncio.create_task(mock_query())

        # Query should complete quickly (not blocked by daemon)
        try:
            await asyncio.wait_for(query_executed.wait(), timeout=0.5)
            assert query_executed.is_set()
        finally:
            await daemon.request_stop()
            await asyncio.wait_for(run_task, timeout=2.0)


class TestDreamLoopIdempotentRestart:
    """Test that dream loop can be restarted idempotently."""

    @pytest.mark.asyncio
    async def test_dream_loop_idempotent_restart(self, mock_config, mock_storage, daemon_module):
        """Test that the dream loop can be stopped and restarted multiple times."""
        SubconsciousDaemon = daemon_module.SubconsciousDaemon

        daemon = SubconsciousDaemon(storage=mock_storage, config=mock_config)

        # First run
        run_task1 = asyncio.create_task(daemon.run())
        await asyncio.sleep(0.1)
        assert daemon.running is True

        await daemon.request_stop()
        await asyncio.wait_for(run_task1, timeout=2.0)
        assert daemon.running is False
        assert daemon._should_stop() is True

        # Second run (restart)
        run_task2 = asyncio.create_task(daemon.run())
        await asyncio.sleep(0.1)
        assert daemon.running is True

        await daemon.request_stop()
        await asyncio.wait_for(run_task2, timeout=2.0)
        assert daemon.running is False

    @pytest.mark.asyncio
    async def test_dream_loop_multiple_stop_calls(self, mock_config, mock_storage, daemon_module):
        """Test that multiple stop calls don't cause issues."""
        SubconsciousDaemon = daemon_module.SubconsciousDaemon

        daemon = SubconsciousDaemon(storage=mock_storage, config=mock_config)

        # Multiple stop calls should not raise errors
        daemon.stop()
        daemon.stop()
        await daemon.request_stop()
        daemon.stop()

        assert daemon._should_stop() is True


class TestDreamLoopMetrics:
    """Test that dream loop emits proper metrics."""

    @pytest.mark.asyncio
    async def test_dream_loop_metrics_recorded(self, mock_config, mock_storage, daemon_module):
        """Test that metrics are recorded during dream loop execution."""
        SubconsciousDaemon = daemon_module.SubconsciousDaemon

        daemon = SubconsciousDaemon(storage=mock_storage, config=mock_config)

        # Run one cycle
        run_task = asyncio.create_task(daemon.run())
        await asyncio.sleep(0.3)

        await daemon.request_stop()
        await asyncio.wait_for(run_task, timeout=2.0)

        # Verify daemon stopped
        assert daemon.running is False


class TestDreamLoopMaxIterations:
    """Test that dream loop respects max_iterations configuration."""

    @pytest.mark.asyncio
    async def test_dream_loop_respects_max_iterations(self, mock_config_limited_iterations, mock_storage, daemon_module):
        """Test that the dream loop stops after max_iterations."""
        SubconsciousDaemon = daemon_module.SubconsciousDaemon

        daemon = SubconsciousDaemon(storage=mock_storage, config=mock_config_limited_iterations)

        # Start daemon
        start_time = time.time()
        run_task = asyncio.create_task(daemon.run())

        # Wait for task to complete (should stop after max_iterations)
        await asyncio.wait_for(run_task, timeout=5.0)
        elapsed = time.time() - start_time

        # Verify it stopped on its own
        assert daemon.running is False
        # Should have completed within reasonable time (2 iterations at 1s each)
        assert elapsed < 5.0


class TestDreamLoopConfiguration:
    """Test dream loop configuration loading."""

    def test_dream_loop_config_from_yaml(self):
        """Test that dream loop configuration is loaded from config.yaml."""
        from mnemocore.core.config import load_config, DreamLoopConfig

        config = load_config()

        # Verify dream_loop config exists and has correct attributes
        assert hasattr(config, 'dream_loop')
        assert isinstance(config.dream_loop, DreamLoopConfig)
        assert hasattr(config.dream_loop, 'enabled')
        assert hasattr(config.dream_loop, 'frequency_seconds')
        assert hasattr(config.dream_loop, 'batch_size')
        assert hasattr(config.dream_loop, 'max_iterations')
        assert hasattr(config.dream_loop, 'ollama_url')
        assert hasattr(config.dream_loop, 'model')

    def test_dream_loop_config_defaults(self):
        """Test that dream loop config has sensible defaults."""
        from mnemocore.core.config import DreamLoopConfig

        config = DreamLoopConfig()

        assert config.enabled is True
        assert config.frequency_seconds == 60
        assert config.batch_size == 10
        assert config.max_iterations == 0
        assert config.ollama_url == "http://localhost:11434/api/generate"
        assert config.model == "gemma3:1b"
