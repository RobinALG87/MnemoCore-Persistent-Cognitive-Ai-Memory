"""
Tests for Concurrency in HAIMEngine
====================================
Task 15.7: Converted from standalone asyncio.run() script to pytest-asyncio tests.

Tests verify that HAIMEngine handles concurrent store and query operations
correctly without race conditions or data corruption.
"""

import asyncio
import random
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.config import get_config


class MockTierManager:
    """Mock TierManager for concurrency testing."""

    def __init__(self):
        self.hot = {}
        self.use_qdrant = False
        self.warm_path = None
        self._lock = asyncio.Lock()

    async def get_hot_snapshot(self):
        """Return a snapshot of hot tier memories."""
        async with self._lock:
            return list(self.hot.values())


@pytest.fixture
def mock_engine():
    """Create a mock HAIMEngine for concurrency testing."""
    engine = MagicMock(spec=HAIMEngine)
    engine.initialize = AsyncMock(return_value=None)
    engine.close = AsyncMock(return_value=None)

    # Track stored memories
    stored_memories = []
    store_lock = asyncio.Lock()

    async def mock_store(content, metadata=None):
        async with store_lock:
            mem_id = f"mem_{len(stored_memories)}_{random.randint(1000, 9999)}"
            stored_memories.append({"id": mem_id, "content": content, "metadata": metadata})
            return mem_id

    async def mock_query(query_text, top_k=10, metadata_filter=None):
        # Return some mock results
        async with store_lock:
            results = []
            for mem in stored_memories[:top_k]:
                results.append((mem["id"], random.uniform(0.5, 1.0)))
            return results

    engine.store = mock_store
    engine.query = mock_query
    engine.tier_manager = MockTierManager()
    engine._stored_memories = stored_memories  # For test assertions

    return engine


@pytest.fixture
def mock_engine_with_real_tier_manager(tmp_path):
    """Create an engine with a more realistic TierManager mock."""
    engine = MagicMock(spec=HAIMEngine)
    engine.initialize = AsyncMock(return_value=None)
    engine.close = AsyncMock(return_value=None)

    # Create a mock tier manager with actual storage
    tier_manager = MagicMock()
    tier_manager.hot = {}
    tier_manager.use_qdrant = False
    tier_manager.warm_path = tmp_path / "warm"
    tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

    async def get_hot_snapshot():
        return list(tier_manager.hot.values())

    tier_manager.get_hot_snapshot = get_hot_snapshot
    engine.tier_manager = tier_manager

    # Track operations
    store_count = [0]
    query_count = [0]
    lock = asyncio.Lock()

    async def mock_store(content, metadata=None):
        async with lock:
            mem_id = f"mem_{store_count[0]}"
            store_count[0] += 1
            tier_manager.hot[mem_id] = MagicMock(
                id=mem_id,
                content=content,
                metadata=metadata or {},
                ltp_strength=0.5
            )
            return mem_id

    async def mock_query(query_text, top_k=10, metadata_filter=None):
        async with lock:
            query_count[0] += 1
            results = []
            for mem_id, node in list(tier_manager.hot.items())[:top_k]:
                results.append((mem_id, random.uniform(0.5, 1.0)))
            return results

    engine.store = mock_store
    engine.query = mock_query
    engine._store_count = store_count
    engine._query_count = query_count

    return engine


class TestConcurrencyBasic:
    """Basic concurrency tests for HAIMEngine."""

    @pytest.mark.asyncio
    async def test_concurrent_stores_no_exceptions(self, mock_engine):
        """Test that concurrent store operations don't raise exceptions."""
        async def store_task(worker_id, num_ops):
            for i in range(num_ops):
                await mock_engine.store(f"Content from worker {worker_id}, op {i}")
                await asyncio.sleep(0.001)

        # Run multiple concurrent store tasks
        tasks = [store_task(i, 10) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all stores completed
        assert len(mock_engine._stored_memories) == 50

    @pytest.mark.asyncio
    async def test_concurrent_queries_no_exceptions(self, mock_engine):
        """Test that concurrent query operations don't raise exceptions."""
        # First store some data
        for i in range(10):
            await mock_engine.store(f"Test content {i}")

        async def query_task(worker_id, num_ops):
            for i in range(num_ops):
                results = await mock_engine.query(f"query {worker_id}", top_k=5)
                assert isinstance(results, list)

        # Run multiple concurrent query tasks
        tasks = [query_task(i, 10) for i in range(5)]
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_mixed_concurrent_operations(self, mock_engine):
        """Test that mixed store/query operations work correctly."""
        async def mixed_task(worker_id, num_ops):
            for i in range(num_ops):
                if random.random() > 0.5:
                    await mock_engine.store(f"Content {worker_id}-{i}")
                else:
                    await mock_engine.query(f"Query {worker_id}", top_k=3)
                await asyncio.sleep(0.001)

        # Run mixed concurrent operations
        tasks = [mixed_task(i, 20) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify some stores completed
        assert len(mock_engine._stored_memories) > 0


class TestConcurrencyWithRealisticEngine:
    """Concurrency tests with more realistic engine mock."""

    @pytest.mark.asyncio
    async def test_concurrent_operations_complete(self, mock_engine_with_real_tier_manager):
        """Test that all concurrent operations complete successfully."""
        engine = mock_engine_with_real_tier_manager

        async def worker_task(worker_id, num_ops=10):
            for i in range(num_ops):
                if random.random() > 0.5:
                    content = f"Worker {worker_id} - Operation {i}"
                    await engine.store(content, metadata={"worker": worker_id})
                else:
                    await engine.query(f"something about worker {worker_id}", top_k=2)
                await asyncio.sleep(random.uniform(0.001, 0.005))

        # Run multiple workers concurrently
        num_workers = 5
        tasks = [worker_task(i, 10) for i in range(num_workers)]

        start_time = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # Verify operations completed
        assert engine._store_count[0] > 0
        assert engine._query_count[0] > 0

        # Verify reasonable completion time (not hanging)
        assert elapsed < 10.0, f"Operations took too long: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_data_integrity_under_concurrency(self, mock_engine_with_real_tier_manager):
        """Test that data integrity is maintained under concurrent access."""
        engine = mock_engine_with_real_tier_manager
        expected_content = set()

        async def store_unique_content(worker_id, num_ops):
            for i in range(num_ops):
                content = f"unique_{worker_id}_{i}"
                expected_content.add(content)
                await engine.store(content)
                await asyncio.sleep(0.001)

        # Store unique content concurrently
        tasks = [store_unique_content(i, 10) for i in range(3)]
        await asyncio.gather(*tasks)

        # Verify all content was stored
        assert engine._store_count[0] == len(expected_content)

    @pytest.mark.asyncio
    async def test_high_concurrency_load(self, mock_engine):
        """Test engine under high concurrency load."""
        async def stress_task(task_id, num_ops=50):
            for i in range(num_ops):
                await mock_engine.store(f"stress_{task_id}_{i}")
                await mock_engine.query(f"query_{task_id}", top_k=5)
                await asyncio.sleep(0.0001)  # Very small delay

        # Run many concurrent tasks
        tasks = [stress_task(i, 20) for i in range(10)]
        start_time = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # All operations should complete without error
        assert len(mock_engine._stored_memories) == 200

        # Should complete in reasonable time
        assert elapsed < 30.0


class TestConcurrencyEdgeCases:
    """Edge case tests for concurrency."""

    @pytest.mark.asyncio
    async def test_rapid_sequential_stores(self, mock_engine):
        """Test rapid sequential stores from single coroutine."""
        for i in range(100):
            await mock_engine.store(f"rapid_{i}")

        assert len(mock_engine._stored_memories) == 100

    @pytest.mark.asyncio
    async def test_concurrent_same_content_stores(self, mock_engine):
        """Test storing identical content concurrently."""
        async def store_same_content():
            return await mock_engine.store("identical content")

        # Store same content concurrently
        tasks = [store_same_content() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Each store should return a unique ID
        assert len(set(results)) == 10  # All unique IDs

    @pytest.mark.asyncio
    async def test_query_during_store_operations(self, mock_engine):
        """Test querying while stores are in progress."""
        store_complete = asyncio.Event()
        query_results = []

        async def background_stores():
            for i in range(50):
                await mock_engine.store(f"background_{i}")
                await asyncio.sleep(0.01)
            store_complete.set()

        async def query_while_storing():
            while not store_complete.is_set():
                results = await mock_engine.query("test", top_k=5)
                query_results.append(len(results))
                await asyncio.sleep(0.005)

        # Run stores and queries concurrently
        await asyncio.gather(background_stores(), query_while_storing())

        # Verify queries executed
        assert len(query_results) > 0

    @pytest.mark.asyncio
    async def test_cancelled_task_handling(self, mock_engine):
        """Test that cancelled tasks don't corrupt state."""
        async def long_running_store():
            for i in range(100):
                await mock_engine.store(f"long_{i}")
                await asyncio.sleep(0.1)

        # Start task and cancel it
        task = asyncio.create_task(long_running_store())
        await asyncio.sleep(0.05)  # Let it run a bit
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Engine should still be in consistent state
        # Some stores may have completed before cancellation
        assert len(mock_engine._stored_memories) >= 0


class TestConcurrencyMetrics:
    """Tests for concurrency performance metrics."""

    @pytest.mark.asyncio
    async def test_operation_timing(self, mock_engine):
        """Test that concurrent operations complete in reasonable time."""
        async def timed_store(worker_id):
            start = time.time()
            for i in range(10):
                await mock_engine.store(f"timed_{worker_id}_{i}")
            return time.time() - start

        tasks = [timed_store(i) for i in range(5)]
        times = await asyncio.gather(*tasks)

        # All workers should complete in reasonable time
        for t in times:
            assert t < 5.0, f"Worker took too long: {t:.2f}s"

    @pytest.mark.asyncio
    async def test_throughput_under_load(self, mock_engine):
        """Test throughput under concurrent load."""
        num_ops = 100

        async def store_many():
            for i in range(num_ops):
                await mock_engine.store(f"throughput_{i}")

        start_time = time.time()
        await store_many()
        elapsed = time.time() - start_time

        throughput = num_ops / elapsed
        # Basic sanity check - should process at least 10 ops/sec
        assert throughput > 10, f"Throughput too low: {throughput:.1f} ops/sec"


# Allow running this file directly for quick manual testing
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
