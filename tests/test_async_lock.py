"""
Tests for async lock initialization patterns.

This module verifies that asyncio.Lock, asyncio.Semaphore, and asyncio.Event
are properly initialized in async initialize() methods rather than in __init__.

This prevents RuntimeError when objects are instantiated outside of an
async context (e.g., during import or synchronous instantiation).
"""

import asyncio
import os
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch


class TestHAIMEngineAsyncLock:
    """Tests for HAIMEngine async lock initialization."""

    def test_engine_sync_instantiation_no_runtime_error(self, tmp_path):
        """
        Verify that HAIMEngine can be instantiated synchronously without
        raising RuntimeError about no running event loop.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.engine import HAIMEngine

            # This should NOT raise RuntimeError
            engine = HAIMEngine(dimension=1024)

            # Locks are created eagerly in __init__ (Python 3.10+ allows this safely)
            assert isinstance(engine.synapse_lock, asyncio.Lock)
            assert isinstance(engine._write_lock, asyncio.Lock)
            assert isinstance(engine._dream_sem, asyncio.Semaphore)
            assert engine._initialized is False
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

    @pytest.mark.asyncio
    async def test_engine_async_initialization(self, tmp_path):
        """
        Verify that HAIMEngine.initialize() properly initializes all
        asyncio primitives with a running event loop.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.engine import HAIMEngine
            from src.core.tier_manager import TierManager

            # Create a TierManager with use_qdrant=False to avoid connection issues
            tier_manager = TierManager()
            tier_manager.use_qdrant = False
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            engine = HAIMEngine(dimension=1024, tier_manager=tier_manager)
            await engine.initialize()

            # Locks should now be initialized
            assert engine.synapse_lock is not None
            assert isinstance(engine.synapse_lock, asyncio.Lock)
            assert engine._write_lock is not None
            assert isinstance(engine._write_lock, asyncio.Lock)
            assert engine._dream_sem is not None
            assert isinstance(engine._dream_sem, asyncio.Semaphore)
            assert engine._initialized is True
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

    @pytest.mark.asyncio
    async def test_engine_initialize_is_idempotent(self, tmp_path):
        """
        Verify that calling initialize() multiple times is safe and
        does not recreate locks.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.engine import HAIMEngine
            from src.core.tier_manager import TierManager

            # Create a TierManager with use_qdrant=False to avoid connection issues
            tier_manager = TierManager()
            tier_manager.use_qdrant = False
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            engine = HAIMEngine(dimension=1024, tier_manager=tier_manager)
            await engine.initialize()

            # Capture lock references
            first_lock = engine.synapse_lock
            first_write_lock = engine._write_lock
            first_sem = engine._dream_sem

            # Call initialize again
            await engine.initialize()

            # Should be the same objects
            assert engine.synapse_lock is first_lock
            assert engine._write_lock is first_write_lock
            assert engine._dream_sem is first_sem
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

    @pytest.mark.asyncio
    async def test_engine_locks_functional(self, tmp_path):
        """
        Verify that the initialized locks can actually be used.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.engine import HAIMEngine
            from src.core.tier_manager import TierManager

            # Create a TierManager with use_qdrant=False to avoid connection issues
            tier_manager = TierManager()
            tier_manager.use_qdrant = False
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            engine = HAIMEngine(dimension=1024, tier_manager=tier_manager)
            await engine.initialize()

            # Test that locks work correctly
            async with engine.synapse_lock:
                pass  # Lock acquired and released

            async with engine._write_lock:
                pass  # Lock acquired and released

            async with engine._dream_sem:
                pass  # Semaphore acquired and released
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()


class TestTierManagerAsyncLock:
    """Tests for TierManager async lock initialization."""

    def test_tier_manager_sync_instantiation_no_runtime_error(self, tmp_path):
        """
        Verify that TierManager can be instantiated synchronously without
        raising RuntimeError about no running event loop.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.tier_manager import TierManager

            # Mock QdrantClient to raise error, forcing fallback to file system
            with patch("qdrant_client.QdrantClient", side_effect=Exception("Qdrant Mock Fail")):
                # This should NOT raise RuntimeError
                tier_manager = TierManager()

                # Lock is created eagerly in __init__ (Python 3.10+ allows this safely)
                assert isinstance(tier_manager.lock, asyncio.Lock)
                assert tier_manager._initialized is False
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

    @pytest.mark.asyncio
    async def test_tier_manager_async_initialization(self, tmp_path):
        """
        Verify that TierManager.initialize() properly initializes the
        asyncio.Lock with a running event loop.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.tier_manager import TierManager

            tier_manager = TierManager()
            tier_manager.use_qdrant = False  # Force file system fallback
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            await tier_manager.initialize()

            # Lock should now be initialized
            assert tier_manager.lock is not None
            assert isinstance(tier_manager.lock, asyncio.Lock)
            assert tier_manager._initialized is True
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

    @pytest.mark.asyncio
    async def test_tier_manager_initialize_is_idempotent(self, tmp_path):
        """
        Verify that calling initialize() multiple times is safe and
        does not recreate locks.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.tier_manager import TierManager

            tier_manager = TierManager()
            tier_manager.use_qdrant = False  # Force file system fallback
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            await tier_manager.initialize()

            # Capture lock reference
            first_lock = tier_manager.lock

            # Call initialize again
            await tier_manager.initialize()

            # Should be the same object
            assert tier_manager.lock is first_lock
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

    @pytest.mark.asyncio
    async def test_tier_manager_lock_functional(self, tmp_path):
        """
        Verify that the initialized lock can actually be used.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.tier_manager import TierManager

            tier_manager = TierManager()
            tier_manager.use_qdrant = False  # Force file system fallback
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            await tier_manager.initialize()

            # Test that lock works correctly
            async with tier_manager.lock:
                pass  # Lock acquired and released
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()


class TestAsyncLockPatternIntegration:
    """Integration tests for async lock patterns across the codebase."""

    @pytest.mark.asyncio
    async def test_full_engine_workflow(self, tmp_path):
        """
        Test a complete workflow: instantiate engine, initialize, use locks.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.engine import HAIMEngine
            from src.core.tier_manager import TierManager

            # Create a TierManager with use_qdrant=False to avoid connection issues
            tier_manager = TierManager()
            tier_manager.use_qdrant = False
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            # Synchronous instantiation (safe)
            engine = HAIMEngine(dimension=1024, tier_manager=tier_manager)

            # Async initialization
            await engine.initialize()

            # Verify we can use engine operations that depend on locks
            # Using synapse_lock via bind_memories
            await engine.bind_memories("test_id_a", "test_id_b", success=True)

            # Verify locks are functional after use
            assert engine.synapse_lock.locked() is False
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

    @pytest.mark.asyncio
    async def test_concurrent_initialize_calls(self, tmp_path):
        """
        Test that concurrent initialize() calls are safe due to idempotency.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.engine import HAIMEngine
            from src.core.tier_manager import TierManager

            # Create a TierManager with use_qdrant=False to avoid connection issues
            tier_manager = TierManager()
            tier_manager.use_qdrant = False
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            engine = HAIMEngine(dimension=1024, tier_manager=tier_manager)

            # Run multiple initialize calls concurrently
            await asyncio.gather(
                engine.initialize(),
                engine.initialize(),
                engine.initialize(),
            )

            # Should only be initialized once
            assert engine._initialized is True
            assert engine.synapse_lock is not None
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

    @pytest.mark.asyncio
    async def test_tier_manager_concurrent_access(self, tmp_path):
        """
        Test that TierManager lock protects concurrent access properly.
        """
        # Set up a temporary data directory
        os.environ["HAIM_DATA_DIR"] = str(tmp_path / "data")
        os.environ["HAIM_WARM_MMAP_DIR"] = str(tmp_path / "warm")
        os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(tmp_path / "cold")

        try:
            from src.core.config import reset_config
            reset_config()

            from src.core.tier_manager import TierManager
            from src.core.node import MemoryNode
            from src.core.binary_hdv import BinaryHDV

            tier_manager = TierManager()
            tier_manager.use_qdrant = False  # Force file system fallback
            if not tier_manager.warm_path:
                tier_manager.warm_path = Path(tmp_path / "warm")
                tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

            await tier_manager.initialize()

            # Create test nodes
            nodes = []
            for i in range(10):
                hdv = BinaryHDV.random(1024)
                node = MemoryNode(
                    id=f"test_node_{i}",
                    hdv=hdv,
                    content=f"Test content {i}",
                    metadata={}
                )
                nodes.append(node)

            # Add nodes concurrently
            async def add_node(node):
                await tier_manager.add_memory(node)

            await asyncio.gather(*[add_node(n) for n in nodes])

            # All nodes should be in hot tier
            assert len(tier_manager.hot) >= 10
        finally:
            # Cleanup
            for key in ["HAIM_DATA_DIR", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
                os.environ.pop(key, None)
            from src.core.config import reset_config
            reset_config()

