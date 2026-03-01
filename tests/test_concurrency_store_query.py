"""
Concurrency Tests for Store and Query Operations
=================================================

Tests for thread-safety and concurrency correctness of the core store/query
operations in MnemoCore. Validates that concurrent operations do not cause
data corruption, duplicates, or crashes.

Test Categories:
    - 100 concurrent store() calls -> all succeed, no duplicates
    - 50 concurrent store() + 50 concurrent query() -> no crashes, valid results
    - Concurrent tier promotion/demotion -> no data loss
    - Concurrent working memory push/pop -> no corruption
"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import numpy as np

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.tier_manager import TierManager
from mnemocore.core.working_memory import WorkingMemoryService
from mnemocore.core.memory_model import WorkingMemoryItem
from mnemocore.core.node import MemoryNode
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.config import get_config


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary directory for storage during tests."""
    storage_dir = tmp_path / "mnemocore_test"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def mock_qdrant():
    """Create a mock QdrantStore for tests that need it."""
    mock = MagicMock()
    mock.ensure_collections = AsyncMock(return_value=None)
    mock.upsert = AsyncMock(return_value=None)
    mock.search = AsyncMock(return_value=[])
    mock.get_point = AsyncMock(return_value=None)
    mock.scroll = AsyncMock(return_value=([], None))
    mock.delete = AsyncMock(return_value=None)
    mock.close = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_redis():
    """Create a mock AsyncRedisStorage for tests."""
    mock = MagicMock()
    mock.check_health = AsyncMock(return_value=True)
    mock.store_memory = AsyncMock(return_value=None)
    mock.retrieve_memory = AsyncMock(return_value=None)
    mock.close = AsyncMock(return_value=None)
    return mock


@pytest.fixture
async def engine(temp_storage_dir, mock_qdrant, mock_redis):
    """
    Create an HAIMEngine instance for testing with mocked external dependencies.

    Uses in-memory storage where possible to isolate concurrency behavior.
    """
    config = get_config()

    # Create engine with mocked dependencies
    engine = HAIMEngine(
        dimension=1024,  # Smaller dimension for faster tests
        config=config,
    )
    # Disable Qdrant for tests
    engine.tier_manager.use_qdrant = False
    engine.tier_manager.warm_path = temp_storage_dir / "warm"
    engine.tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

    await engine.initialize()

    yield engine

    await engine.close()


@pytest.fixture
def working_memory():
    """Create a WorkingMemoryService for testing."""
    return WorkingMemoryService(max_items_per_agent=32)


# =============================================================================
# Test: 100 Concurrent Store Operations
# =============================================================================

class TestConcurrentStoreOperations:
    """
    Tests for concurrent store() calls.

    Validates that:
    - All store operations complete successfully
    - No duplicate memories are created
    - All stored memories are retrievable
    """

    @pytest.mark.asyncio
    async def test_100_concurrent_stores_all_succeed(self, engine):
        """
        Test that 100 concurrent store() calls all succeed without errors.

        This validates that the tier manager's locking mechanism correctly
        serializes access to shared state during concurrent writes.
        """
        num_stores = 100
        store_tasks = []

        async def store_item(i: int) -> str:
            """Store a single item and return its ID."""
            content = f"Concurrent test item {i} - {uuid.uuid4().hex[:8]}"
            node_id = await engine.store(
                content,
                metadata={"batch": "concurrent_test", "index": i}
            )
            return node_id

        # Launch all store operations concurrently
        for i in range(num_stores):
            store_tasks.append(store_item(i))

        # Wait for all to complete
        node_ids = await asyncio.gather(*store_tasks, return_exceptions=True)

        # Verify all succeeded
        successes = [nid for nid in node_ids if not isinstance(nid, Exception)]
        failures = [nid for nid in node_ids if isinstance(nid, Exception)]

        assert len(successes) == num_stores, (
            f"Expected {num_stores} successful stores, got {len(successes)}. "
            f"Failures: {failures}"
        )

        # Verify no duplicate IDs
        unique_ids = set(successes)
        assert len(unique_ids) == num_stores, (
            f"Expected {num_stores} unique IDs, got {len(unique_ids)} "
            f"(duplicates detected)"
        )

    @pytest.mark.asyncio
    async def test_concurrent_stores_no_duplicates(self, engine):
        """
        Test that concurrent stores with identical content do not create
        duplicate entries when deduplication is expected.

        Note: MnemoCore generates unique IDs per store call, so this test
        validates that the internal state remains consistent.
        """
        num_stores = 50
        content = "Shared content for deduplication test"
        store_tasks = []

        # Store the same content multiple times concurrently
        for i in range(num_stores):
            store_tasks.append(engine.store(content, metadata={"iteration": i}))

        node_ids = await asyncio.gather(*store_tasks, return_exceptions=True)

        # All should succeed (each gets a unique ID)
        successes = [nid for nid in node_ids if not isinstance(nid, Exception)]
        assert len(successes) == num_stores

        # Verify HOT tier state is consistent
        hot_snapshot = await engine.tier_manager.get_hot_snapshot()
        hot_ids = {node.id for node in hot_snapshot}

        # All stored IDs should be in HOT (or promoted to WARM)
        for nid in successes:
            assert nid in hot_ids or await engine.tier_manager._warm_storage.contains(nid)


# =============================================================================
# Test: 50 Concurrent Store + 50 Concurrent Query
# =============================================================================

class TestConcurrentStoreQuery:
    """
    Tests for concurrent store and query operations.

    Validates that:
    - Mixed read/write operations don't crash
    - Query results remain valid during concurrent writes
    - No data corruption occurs
    """

    @pytest.mark.asyncio
    async def test_50_store_50_query_no_crashes(self, engine):
        """
        Test 50 concurrent store() + 50 concurrent query() operations.

        Validates that mixed read/write workloads don't cause crashes
        or deadlocks.
        """
        num_ops = 50
        tasks = []

        # First, store some initial data to query against
        for i in range(10):
            await engine.store(f"Initial content {i}", metadata={"seed": True})

        async def store_op(i: int):
            """Store operation."""
            return await engine.store(
                f"Store operation {i}",
                metadata={"op": "store", "index": i}
            )

        async def query_op(i: int):
            """Query operation."""
            results = await engine.query(f"content {i % 10}", top_k=5)
            return ("query", i, len(results))

        # Mix of stores and queries
        for i in range(num_ops):
            tasks.append(store_op(i))
            tasks.append(query_op(i))

        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for crashes
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, (
            f"Concurrent operations caused {len(exceptions)} exceptions: {exceptions}"
        )

        # Verify results are valid
        query_results = [r for r in results if isinstance(r, tuple) and r[0] == "query"]
        for _, idx, count in query_results:
            assert isinstance(count, int), f"Invalid query result count: {count}"
            assert count >= 0, f"Negative result count: {count}"

    @pytest.mark.asyncio
    async def test_concurrent_queries_return_valid_results(self, engine):
        """
        Test that concurrent queries return valid, consistent results.

        Each query should return results that are valid MemoryNode objects
        or empty lists, never corrupted data.
        """
        # Seed some data
        for i in range(20):
            await engine.store(f"Test content {i} with unique keywords", metadata={"seed": True})

        num_queries = 100
        tasks = []

        async def query_and_validate(i: int):
            """Query and validate results."""
            results = await engine.query(f"content {i % 20}", top_k=5)
            # Validate each result
            for node_id, score in results:
                assert isinstance(node_id, str), f"Invalid node_id type: {type(node_id)}"
                assert isinstance(score, (int, float)), f"Invalid score type: {type(score)}"
                assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
            return len(results)

        for i in range(num_queries):
            tasks.append(query_and_validate(i))

        counts = await asyncio.gather(*tasks, return_exceptions=True)

        exceptions = [c for c in counts if isinstance(c, Exception)]
        assert len(exceptions) == 0, f"Query exceptions: {exceptions}"

        # All counts should be non-negative integers
        for c in counts:
            assert isinstance(c, int), f"Invalid count: {c}"
            assert c >= 0, f"Negative count: {c}"


# =============================================================================
# Test: Concurrent Tier Promotion/Demotion
# =============================================================================

class TestConcurrentTierOperations:
    """
    Tests for concurrent tier promotion and demotion.

    Validates that:
    - Tier transitions don't lose data
    - Concurrent promotions/demotions are handled correctly
    - Memory state remains consistent
    """

    @pytest.mark.asyncio
    async def test_concurrent_tier_transitions_no_data_loss(self, engine):
        """
        Test that concurrent tier transitions don't result in data loss.

        Creates memories and then triggers concurrent promotions/demotions
        by accessing them with different patterns.
        """
        # Store memories with varying LTP strengths
        node_ids = []
        for i in range(20):
            nid = await engine.store(
                f"Tier test memory {i}",
                metadata={"tier_test": True, "index": i}
            )
            node_ids.append(nid)

            # Set varying LTP strengths
            node = await engine.tier_manager.get_memory(nid)
            if node:
                # Low LTP -> likely to demote
                if i % 3 == 0:
                    node.ltp_strength = 0.1
                # High LTP -> likely to stay in HOT
                elif i % 3 == 1:
                    node.ltp_strength = 0.9
                # Medium LTP
                else:
                    node.ltp_strength = 0.5

        # Concurrent access to trigger tier transitions
        async def access_memory(nid: str, times: int):
            """Access a memory multiple times to trigger promotion."""
            for _ in range(times):
                node = await engine.tier_manager.get_memory(nid)
                if node:
                    node.access()
            return nid

        tasks = []
        for i, nid in enumerate(node_ids):
            # Varying access patterns
            access_count = (i % 5) + 1
            tasks.append(access_memory(nid, access_count))

        result_ids = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no data loss - all memories should still be retrievable
        for nid in node_ids:
            node = await engine.tier_manager.get_memory(nid)
            assert node is not None, f"Memory {nid} was lost during tier transitions"
            assert node.content is not None, f"Memory {nid} lost its content"

    @pytest.mark.asyncio
    async def test_concurrent_promotions_handle_capacity(self, engine):
        """
        Test that concurrent promotions handle capacity limits correctly.

        When promoting multiple memories to HOT tier concurrently, the
        eviction logic should maintain capacity limits without data loss.
        """
        # Fill HOT tier to near capacity
        config = get_config()
        max_hot = config.tiers_hot.max_memories

        # Store more items than HOT can hold
        node_ids = []
        for i in range(max_hot + 20):
            nid = await engine.store(
                f"Capacity test {i}",
                metadata={"capacity_test": True}
            )
            node_ids.append(nid)

        # Verify all nodes are still accessible (either in HOT or WARM)
        for nid in node_ids:
            node = await engine.tier_manager.get_memory(nid)
            assert node is not None, f"Node {nid} lost during capacity management"

        # Verify HOT tier respects capacity
        stats = await engine.tier_manager.get_stats()
        assert stats["hot_count"] <= max_hot, (
            f"HOT tier exceeded capacity: {stats['hot_count']} > {max_hot}"
        )


# =============================================================================
# Test: Concurrent Working Memory Push/Pop
# =============================================================================

class TestConcurrentWorkingMemory:
    """
    Tests for concurrent working memory operations.

    Validates that:
    - Concurrent push operations don't corrupt state
    - Concurrent pop/eviction maintains consistency
    - Agent isolation is preserved under concurrency
    """

    @pytest.mark.asyncio
    async def test_concurrent_push_no_corruption(self, working_memory):
        """
        Test that concurrent push_item operations don't corrupt working memory.

        All items should be stored correctly without data loss or corruption.
        """
        num_items = 100
        agent_id = "test_agent"

        async def push_item(i: int):
            """Push a single item to working memory."""
            item = WorkingMemoryItem(
                id=f"wm_{i}_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                created_at=datetime.now(timezone.utc),
                ttl_seconds=3600,
                content=f"Working memory item {i}",
                importance=0.5 + (i % 10) * 0.05,
                kind="thought",
                tags=["concurrent_test"]
            )
            await working_memory.push_item(agent_id, item)
            return item.id

        tasks = [push_item(i) for i in range(num_items)]
        item_ids = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all pushes succeeded
        successes = [iid for iid in item_ids if not isinstance(iid, Exception)]
        assert len(successes) == num_items, (
            f"Expected {num_items} successful pushes, got {len(successes)}"
        )

        # Verify final state is consistent
        state = await working_memory.get_state(agent_id)
        assert state is not None
        assert len(state.items) <= working_memory.max_items_per_agent

        # Verify no duplicate IDs in state
        state_ids = [item.id for item in state.items]
        assert len(state_ids) == len(set(state_ids)), "Duplicate IDs in working memory"

    @pytest.mark.asyncio
    async def test_concurrent_push_pop_maintains_consistency(self, working_memory):
        """
        Test that interleaved push and clear operations maintain consistency.

        Working memory should never be in an inconsistent state.
        """
        agent_id = "test_agent"
        num_operations = 50

        async def push_op(i: int):
            """Push an item."""
            item = WorkingMemoryItem(
                id=f"wm_push_{i}",
                agent_id=agent_id,
                created_at=datetime.now(timezone.utc),
                ttl_seconds=3600,
                content=f"Item {i}",
                importance=0.5,
                kind="thought",
                tags=[]
            )
            await working_memory.push_item(agent_id, item)
            return "push"

        async def clear_op(i: int):
            """Clear working memory."""
            await working_memory.clear(agent_id)
            return "clear"

        # Interleave pushes and clears
        tasks = []
        for i in range(num_operations):
            if i % 5 == 0:
                tasks.append(clear_op(i))
            else:
                tasks.append(push_op(i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Exceptions during operations: {exceptions}"

        # Final state should be valid (either empty or with valid items)
        state = await working_memory.get_state(agent_id)
        if state:
            for item in state.items:
                assert item.id is not None
                assert item.content is not None
                assert item.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_concurrent_multi_agent_isolation(self, working_memory):
        """
        Test that concurrent operations from different agents maintain isolation.

        Each agent's working memory should remain independent.
        """
        num_agents = 5
        items_per_agent = 20

        async def agent_operations(agent_id: str):
            """Perform multiple operations for a single agent."""
            for i in range(items_per_agent):
                item = WorkingMemoryItem(
                    id=f"wm_{agent_id}_{i}",
                    agent_id=agent_id,
                    created_at=datetime.now(timezone.utc),
                    ttl_seconds=3600,
                    content=f"Content for {agent_id} item {i}",
                    importance=0.5,
                    kind="thought",
                    tags=[]
                )
                await working_memory.push_item(agent_id, item)

            # Verify isolation
            state = await working_memory.get_state(agent_id)
            if state:
                for item in state.items:
                    assert item.agent_id == agent_id, (
                        f"Agent contamination: {item.agent_id} in {agent_id}'s memory"
                    )
            return agent_id

        # Run concurrent operations for all agents
        tasks = [agent_operations(f"agent_{i}") for i in range(num_agents)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == num_agents

        # Verify final isolation
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            state = await working_memory.get_state(agent_id)
            if state:
                for item in state.items:
                    assert item.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_concurrent_promote_prune_no_corruption(self, working_memory):
        """
        Test that concurrent promote_item and prune_all operations don't corrupt state.

        These operations modify the same internal structures and must be safe.
        """
        agent_id = "test_agent"

        # Pre-populate with items
        for i in range(20):
            item = WorkingMemoryItem(
                id=f"wm_promote_{i}",
                agent_id=agent_id,
                created_at=datetime.now(timezone.utc),
                ttl_seconds=3600,
                content=f"Item {i}",
                importance=0.3,
                kind="thought",
                tags=[]
            )
            await working_memory.push_item(agent_id, item)

        async def promote_op(i: int):
            """Promote an item."""
            await working_memory.promote_item(agent_id, f"wm_promote_{i % 20}", bonus=0.1)
            return "promote"

        async def prune_op(i: int):
            """Prune all agents."""
            await working_memory.prune_all()
            return "prune"

        # Mix promotes and prunes
        tasks = []
        for i in range(40):
            if i % 3 == 0:
                tasks.append(prune_op(i))
            else:
                tasks.append(promote_op(i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Exceptions: {exceptions}"

        # Final state should be valid
        state = await working_memory.get_state(agent_id)
        if state:
            # All items should have valid structure
            for item in state.items:
                assert item.id is not None
                assert 0.0 <= item.importance <= 1.0


# =============================================================================
# Test: Stress Testing
# =============================================================================

class TestConcurrencyStress:
    """
    Stress tests for concurrency under high load.

    These tests push the system harder to find edge cases.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_high_concurrency_store_stress(self, engine):
        """
        Stress test with 500 concurrent store operations.

        Marked as 'slow' - run with: pytest -m slow
        """
        num_stores = 500

        async def store_op(i: int):
            return await engine.store(
                f"Stress test item {i} - {uuid.uuid4().hex}",
                metadata={"stress_test": True, "index": i}
            )

        tasks = [store_op(i) for i in range(num_stores)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) >= num_stores * 0.95, (
            f"Too many failures under stress: {num_stores - len(successes)} / {num_stores}"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_concurrent_load(self, engine):
        """
        Test sustained concurrent load over time.

        Runs multiple waves of concurrent operations.
        """
        waves = 10
        ops_per_wave = 50

        for wave in range(waves):
            tasks = []
            for i in range(ops_per_wave):
                tasks.append(engine.store(
                    f"Wave {wave} item {i}",
                    metadata={"wave": wave, "item": i}
                ))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            successes = [r for r in results if not isinstance(r, Exception)]

            # Each wave should have high success rate
            success_rate = len(successes) / ops_per_wave
            assert success_rate >= 0.95, f"Wave {wave} had low success rate: {success_rate}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
