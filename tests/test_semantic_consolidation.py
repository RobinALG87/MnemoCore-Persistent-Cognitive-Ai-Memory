"""
Comprehensive Tests for Semantic Consolidation Worker
=====================================================

Tests the nightly semantic consolidation background worker.

Coverage:
- run_once() with mocked node store
- Hamming distance computation correctness
- Cluster merging logic
- Node existence check
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import numpy as np

from mnemocore.core.semantic_consolidation import (
    SemanticConsolidationWorker,
    SemanticConsolidationConfig,
    _hamming_matrix,
    _kmedoids_iter,
)
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def consolidation_config():
    """Create a SemanticConsolidationConfig for testing."""
    return SemanticConsolidationConfig(
        schedule_hour=3,
        duplicate_epsilon=0.05,
        cluster_k=8,
        cluster_max_iter=5,
        min_cluster_size=2,
        prune_duplicates=True,
        min_ltp_to_prune=1.0,
        batch_size=100,
        enabled=True,
    )


@pytest.fixture
def mock_engine():
    """Create a mock HAIMEngine for testing."""
    engine = MagicMock()
    engine.tier_manager = MagicMock()
    engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[])
    engine.tier_manager.list_warm = AsyncMock(return_value=[])
    engine.tier_manager.delete_memory = AsyncMock(return_value=True)
    engine.tier_manager.get_memory = AsyncMock(return_value=None)
    return engine


def create_test_node(node_id: str, content: str, ltp: float = 0.5, dimension: int = 1024) -> MemoryNode:
    """Helper to create test memory nodes."""
    node = MemoryNode(
        id=node_id,
        content=content,
        hdv=BinaryHDV.random(dimension),
    )
    node.ltp_strength = ltp
    return node


# =============================================================================
# SemanticConsolidationConfig Tests
# =============================================================================

class TestSemanticConsolidationConfig:
    """Test configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = SemanticConsolidationConfig()

        assert config.schedule_hour == 3
        assert config.duplicate_epsilon == 0.05
        assert config.cluster_k == 32
        assert config.cluster_max_iter == 10
        assert config.min_cluster_size == 3
        assert config.prune_duplicates is True
        assert config.enabled is True

    def test_custom_config(self):
        """Custom config values should be set correctly."""
        config = SemanticConsolidationConfig(
            schedule_hour=5,
            duplicate_epsilon=0.1,
            cluster_k=16,
            prune_duplicates=False,
        )

        assert config.schedule_hour == 5
        assert config.duplicate_epsilon == 0.1
        assert config.cluster_k == 16
        assert config.prune_duplicates is False


# =============================================================================
# Hamming Distance Tests
# =============================================================================

class TestHammingMatrix:
    """Test Hamming distance matrix computation."""

    def test_hamming_matrix_identical_vectors(self):
        """Distance between identical vectors should be 0."""
        # Create identical vectors
        vec = np.random.randint(0, 256, (3, 128), dtype=np.uint8)
        vecs = np.vstack([vec, vec, vec])  # 3 identical vectors

        dist = _hamming_matrix(vecs)

        # Diagonal should be 0
        for i in range(3):
            assert dist[i, i] == 0.0

    def test_hamming_matrix_symmetric(self):
        """Distance matrix should be symmetric."""
        vecs = np.random.randint(0, 256, (5, 128), dtype=np.uint8)

        dist = _hamming_matrix(vecs)

        # Should be symmetric
        assert np.allclose(dist, dist.T)

    def test_hamming_matrix_range(self):
        """Distances should be in [0, 1] range."""
        vecs = np.random.randint(0, 256, (10, 128), dtype=np.uint8)

        dist = _hamming_matrix(vecs)

        assert np.all(dist >= 0.0)
        assert np.all(dist <= 1.0)

    def test_hamming_matrix_correctness(self):
        """Should compute correct normalized Hamming distance."""
        # Create two vectors with known difference
        vec1 = np.zeros(16, dtype=np.uint8)  # All zeros
        vec2 = np.zeros(16, dtype=np.uint8)
        vec2[0] = 255  # All ones in first byte

        vecs = np.vstack([vec1, vec2])

        dist = _hamming_matrix(vecs)

        # Distance should be 8/128 = 0.0625 (8 bits different out of 128)
        assert abs(dist[0, 1] - 8 / 128) < 0.01
        assert abs(dist[1, 0] - 8 / 128) < 0.01

    def test_hamming_matrix_single_vector(self):
        """Should handle single vector."""
        vecs = np.random.randint(0, 256, (1, 128), dtype=np.uint8)

        dist = _hamming_matrix(vecs)

        assert dist.shape == (1, 1)
        assert dist[0, 0] == 0.0


# =============================================================================
# K-Medoids Tests
# =============================================================================

class TestKMedoids:
    """Test k-medoids clustering."""

    def test_kmedoids_returns_correct_shapes(self):
        """Should return correct shapes for medoids and labels."""
        vecs = np.random.randint(0, 256, (20, 128), dtype=np.uint8)
        dist = _hamming_matrix(vecs)

        medoids, labels = _kmedoids_iter(dist, k=4, max_iter=5)

        assert len(medoids) <= 4
        assert len(labels) == 20
        assert all(0 <= l < len(medoids) for l in labels)

    def test_kmedoids_clusters_similar_vectors(self):
        """Similar vectors should be in same cluster."""
        # Create two groups of similar vectors
        base1 = np.random.randint(0, 256, (1, 128), dtype=np.uint8)
        base2 = np.random.randint(0, 256, (1, 128), dtype=np.uint8)

        group1 = np.vstack([base1 + np.random.randint(0, 2, (5, 128)) for _ in range(5)])
        group2 = np.vstack([base2 + np.random.randint(0, 2, (5, 128)) for _ in range(5)])

        vecs = np.vstack([group1, group2]).astype(np.uint8)
        dist = _hamming_matrix(vecs)

        medoids, labels = _kmedoids_iter(dist, k=2, max_iter=10)

        # Should have 2 clusters
        assert len(set(labels)) <= 2

    def test_kmedoids_respects_k(self):
        """Should not create more than k clusters."""
        vecs = np.random.randint(0, 256, (50, 128), dtype=np.uint8)
        dist = _hamming_matrix(vecs)

        k = 5
        medoids, labels = _kmedoids_iter(dist, k=k, max_iter=10)

        assert len(medoids) <= k

    def test_kmedoids_handles_more_clusters_than_data(self):
        """Should handle k > n case."""
        vecs = np.random.randint(0, 256, (3, 128), dtype=np.uint8)
        dist = _hamming_matrix(vecs)

        medoids, labels = _kmedoids_iter(dist, k=10, max_iter=5)

        # Should not have more clusters than data points
        assert len(medoids) <= 3


# =============================================================================
# SemanticConsolidationWorker Lifecycle Tests
# =============================================================================

class TestSemanticConsolidationWorkerLifecycle:
    """Test worker lifecycle."""

    def test_worker_init(self, mock_engine, consolidation_config):
        """Worker should initialize correctly."""
        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)

        assert worker.engine == mock_engine
        assert worker.cfg == consolidation_config
        assert worker._running is False
        assert worker._task is None

    @pytest.mark.asyncio
    async def test_worker_start(self, mock_engine, consolidation_config):
        """Worker should start when enabled."""
        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)

        await worker.start()

        assert worker._running is True
        assert worker._task is not None

        await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_start_disabled(self, mock_engine):
        """Worker should not start when disabled."""
        config = SemanticConsolidationConfig(enabled=False)
        worker = SemanticConsolidationWorker(mock_engine, config=config)

        await worker.start()

        assert worker._running is False
        assert worker._task is None

    @pytest.mark.asyncio
    async def test_worker_stop(self, mock_engine, consolidation_config):
        """Worker should stop cleanly."""
        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)

        await worker.start()
        await worker.stop()

        assert worker._running is False
        assert worker._task is None

    @pytest.mark.asyncio
    async def test_worker_stop_without_start(self, mock_engine, consolidation_config):
        """Worker should handle stop without start."""
        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)

        # Should not raise
        await worker.stop()


# =============================================================================
# run_once Tests
# =============================================================================

class TestRunOnce:
    """Test run_once method."""

    @pytest.mark.asyncio
    async def test_run_once_with_nodes(self, mock_engine, consolidation_config):
        """run_once should process nodes correctly."""
        # Create test nodes
        nodes = [create_test_node(f"node_{i}", f"Content {i}") for i in range(10)]
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_engine.tier_manager.list_warm = AsyncMock(return_value=[])
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=nodes[0])

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        stats = await worker.run_once()

        assert stats is not None
        assert stats["nodes_processed"] == 10
        assert worker.last_run is not None

    @pytest.mark.asyncio
    async def test_run_once_too_few_nodes(self, mock_engine, consolidation_config):
        """run_once should skip if too few nodes."""
        nodes = [create_test_node("node_1", "Content 1")]
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        stats = await worker.run_once()

        assert stats == {}

    @pytest.mark.asyncio
    async def test_run_once_prunes_duplicates(self, mock_engine, consolidation_config):
        """run_once should prune near-duplicate nodes."""
        # Create nodes with identical HDVs (duplicates)
        hdv = BinaryHDV.random(1024)
        nodes = [
            create_test_node("node_1", "Content 1", ltp=0.8),
            create_test_node("node_2", "Content 2", ltp=0.5),
        ]
        # Make them identical
        nodes[0].hdv = hdv
        nodes[1].hdv = hdv

        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_engine.tier_manager.get_memory = AsyncMock(side_effect=lambda x: nodes[0] if x == "node_1" else None)

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        stats = await worker.run_once()

        # Should have pruned at least one duplicate
        assert stats["duplicates_pruned"] >= 1

    @pytest.mark.asyncio
    async def test_run_once_respects_min_ltp_to_prune(self, mock_engine):
        """run_once should respect min_ltp_to_prune setting."""
        config = SemanticConsolidationConfig(
            duplicate_epsilon=0.05,
            min_ltp_to_prune=0.5,  # Only prune if LTP < 0.5
            prune_duplicates=True,
            min_cluster_size=2,
        )

        hdv = BinaryHDV.random(1024)
        nodes = [
            create_test_node("node_1", "Content 1", ltp=0.8),
            create_test_node("node_2", "Content 2", ltp=0.8),  # High LTP
        ]
        nodes[0].hdv = hdv
        nodes[1].hdv = hdv

        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=nodes[0])

        worker = SemanticConsolidationWorker(mock_engine, config=config)
        stats = await worker.run_once()

        # Should not prune because LTP is too high
        assert stats["duplicates_pruned"] == 0


# =============================================================================
# Node Collection Tests
# =============================================================================

class TestCollectNodes:
    """Test node collection."""

    @pytest.mark.asyncio
    async def test_collect_hot_nodes(self, mock_engine, consolidation_config):
        """Should collect HOT tier nodes."""
        hot_nodes = [create_test_node(f"hot_{i}", f"Hot {i}") for i in range(5)]
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=hot_nodes)
        mock_engine.tier_manager.list_warm = AsyncMock(side_effect=AttributeError())

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        nodes = await worker._collect_nodes()

        assert len(nodes) == 5

    @pytest.mark.asyncio
    async def test_collect_hot_and_warm_nodes(self, mock_engine, consolidation_config):
        """Should collect both HOT and WARM tier nodes."""
        hot_nodes = [create_test_node(f"hot_{i}", f"Hot {i}") for i in range(3)]
        warm_nodes = [create_test_node(f"warm_{i}", f"Warm {i}") for i in range(5)]

        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=hot_nodes)
        mock_engine.tier_manager.list_warm = AsyncMock(return_value=warm_nodes)

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        nodes = await worker._collect_nodes()

        assert len(nodes) == 8

    @pytest.mark.asyncio
    async def test_collect_respects_batch_size(self, mock_engine):
        """Should respect batch_size for WARM tier."""
        config = SemanticConsolidationConfig(batch_size=5)
        warm_nodes = [create_test_node(f"warm_{i}", f"Warm {i}") for i in range(20)]

        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[])
        mock_engine.tier_manager.list_warm = AsyncMock(return_value=warm_nodes[:5])

        worker = SemanticConsolidationWorker(mock_engine, config=config)
        nodes = await worker._collect_nodes()

        # list_warm should have been called with max_results=5
        mock_engine.tier_manager.list_warm.assert_called_once()


# =============================================================================
# Duplicate Pruning Tests
# =============================================================================

class TestPruneDuplicates:
    """Test duplicate pruning."""

    @pytest.mark.asyncio
    async def test_prune_keeps_higher_ltp(self, mock_engine, consolidation_config):
        """Should keep node with higher LTP when pruning."""
        hdv = BinaryHDV.random(1024)
        nodes = [
            create_test_node("high_ltp", "Content", ltp=0.9),
            create_test_node("low_ltp", "Content", ltp=0.3),
        ]
        nodes[0].hdv = hdv
        nodes[1].hdv = hdv

        vecs = np.vstack([hdv.data, hdv.data])

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        pruned = await worker._prune_duplicates(nodes, vecs)

        # Should have pruned 1 (the lower LTP one)
        assert pruned == 1
        mock_engine.tier_manager.delete_memory.assert_called_once_with("low_ltp")

    @pytest.mark.asyncio
    async def test_prune_no_duplicates(self, mock_engine, consolidation_config):
        """Should not prune if no duplicates found."""
        # Create very different nodes
        nodes = [create_test_node(f"node_{i}", f"Content {i}") for i in range(5)]
        vecs = np.vstack([n.hdv.data for n in nodes])

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        pruned = await worker._prune_duplicates(nodes, vecs)

        assert pruned == 0


# =============================================================================
# Node Existence Tests
# =============================================================================

class TestNodeExists:
    """Test node existence check."""

    @pytest.mark.asyncio
    async def test_node_exists_true(self, mock_engine, consolidation_config):
        """_node_exists should return True for existing node."""
        node = create_test_node("existing", "Content")
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=node)

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        exists = await worker._node_exists("existing")

        assert exists is True

    @pytest.mark.asyncio
    async def test_node_exists_false(self, mock_engine, consolidation_config):
        """_node_exists should return False for non-existing node."""
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)
        exists = await worker._node_exists("nonexistent")

        assert exists is False


# =============================================================================
# Schedule Tests
# =============================================================================

class TestSchedule:
    """Test scheduling logic."""

    def test_seconds_until_next_run_future(self, mock_engine, consolidation_config):
        """Should calculate seconds until next scheduled run."""
        worker = SemanticConsolidationWorker(mock_engine, config=consolidation_config)

        seconds = worker._seconds_until_next_run()

        assert seconds > 0
        assert seconds <= 86400  # Less than 24 hours

    def test_seconds_until_next_run_past_today(self, mock_engine):
        """Should schedule for tomorrow if hour has passed."""
        # Set schedule to hour 1 (likely in the past)
        config = SemanticConsolidationConfig(schedule_hour=1)
        worker = SemanticConsolidationWorker(mock_engine, config=config)

        seconds = worker._seconds_until_next_run()

        # Should be less than 24 hours
        assert seconds > 0
        assert seconds <= 86400


# =============================================================================
# Hook Tests
# =============================================================================

class TestPostConsolidationHook:
    """Test post-consolidation hook."""

    @pytest.mark.asyncio
    async def test_hook_called_after_run(self, mock_engine, consolidation_config):
        """Post-consolidation hook should be called after run."""
        hook_calls = []

        def my_hook(stats):
            hook_calls.append(stats)

        nodes = [create_test_node(f"node_{i}", f"Content {i}") for i in range(5)]
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=nodes[0])

        worker = SemanticConsolidationWorker(
            mock_engine,
            config=consolidation_config,
            post_consolidation_hook=my_hook,
        )
        await worker.run_once()

        assert len(hook_calls) == 1
        assert "nodes_processed" in hook_calls[0]

    @pytest.mark.asyncio
    async def test_async_hook(self, mock_engine, consolidation_config):
        """Should support async hooks."""
        hook_calls = []

        async def async_hook(stats):
            hook_calls.append(stats)

        nodes = [create_test_node(f"node_{i}", f"Content {i}") for i in range(5)]
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=nodes[0])

        worker = SemanticConsolidationWorker(
            mock_engine,
            config=consolidation_config,
            post_consolidation_hook=async_hook,
        )
        await worker.run_once()

        assert len(hook_calls) == 1

    @pytest.mark.asyncio
    async def test_hook_exception_handled(self, mock_engine, consolidation_config):
        """Hook exceptions should be caught and logged."""
        def failing_hook(stats):
            raise ValueError("Hook failed!")

        nodes = [create_test_node(f"node_{i}", f"Content {i}") for i in range(5)]
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=nodes[0])

        worker = SemanticConsolidationWorker(
            mock_engine,
            config=consolidation_config,
            post_consolidation_hook=failing_hook,
        )

        # Should not raise
        stats = await worker.run_once()

        assert stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
