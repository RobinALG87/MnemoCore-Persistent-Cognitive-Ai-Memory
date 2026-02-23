"""
Comprehensive Tests for Semantic Consolidation
=============================================

Tests the SemanticConsolidator and ConsolidationService including:
- Cluster finding with Union-Find
- Cluster merging
- Tier consolidation
- Representative selection
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from mnemocore.core.consolidation import (
    SemanticConsolidator,
    ConsolidationService,
)
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode


class TestSemanticConsolidatorInit:
    """Test SemanticConsolidator initialization."""

    def test_initialization(self):
        """Should initialize with tier manager and config."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(
            mock_tier_manager,
            similarity_threshold=0.85,
            min_cluster_size=2
        )
        assert consolidator.tier_manager == mock_tier_manager
        assert consolidator.similarity_threshold == 0.85
        assert consolidator.min_cluster_size == 2

    def test_default_parameters(self):
        """Should use default parameters."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)
        assert consolidator.similarity_threshold == 0.85
        assert consolidator.min_cluster_size == 2


class TestFindClusters:
    """Test cluster finding functionality."""

    def test_empty_list_returns_empty(self):
        """Empty node list should return empty clusters."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        clusters = consolidator.find_clusters([])

        assert clusters == []

    def test_single_node_returns_empty(self):
        """Single node cannot form a cluster."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        node = MagicMock()
        node.hdv = BinaryHDV.random(1024)

        clusters = consolidator.find_clusters([node])

        assert clusters == []

    def test_finds_similar_clusters(self):
        """Should find clusters of similar nodes."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(
            mock_tier_manager,
            similarity_threshold=0.90,
            min_cluster_size=2
        )

        # Create similar vectors
        base = BinaryHDV.from_seed("base", 1024)
        node1 = MagicMock()
        node1.hdv = base
        node2 = MagicMock()
        node2.hdv = base  # Identical -> very similar
        node3 = MagicMock()
        node3.hdv = BinaryHDV.random(1024)  # Different

        clusters = consolidator.find_clusters([node1, node2, node3])

        # Should have at least one cluster (node1, node2)
        assert len(clusters) >= 1
        assert len(clusters[0]) >= 2

    def test_respects_min_cluster_size(self):
        """Should filter clusters below minimum size."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(
            mock_tier_manager,
            similarity_threshold=0.80,
            min_cluster_size=3
        )

        # Create 2 similar nodes (below threshold)
        base = BinaryHDV.from_seed("base", 1024)
        node1 = MagicMock()
        node1.hdv = base
        node2 = MagicMock()
        node2.hdv = base
        node3 = MagicMock()
        node3.hdv = BinaryHDV.random(1024)

        clusters = consolidator.find_clusters([node1, node2, node3])

        # No cluster should meet min size of 3
        assert len(clusters) == 0

    def test_threshold_parameter(self):
        """Custom threshold should affect clustering."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        # Create vectors with varying similarity
        base = BinaryHDV.from_seed("base", 1024)
        nodes = []
        for i in range(5):
            node = MagicMock()
            # Modify slightly
            if i < 3:
                node.hdv = base  # Same for first 3
            else:
                node.hdv = BinaryHDV.random(1024)
            nodes.append(node)

        # High threshold - fewer clusters
        clusters_high = consolidator.find_clusters(nodes, threshold=0.99)

        # Low threshold - more clusters
        clusters_low = consolidator.find_clusters(nodes, threshold=0.70)

        # Results should be valid lists
        assert isinstance(clusters_high, list)
        assert isinstance(clusters_low, list)


class TestMergeCluster:
    """Test cluster merging functionality."""

    def test_empty_cluster(self):
        """Empty cluster should not crash."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        result = consolidator.merge_cluster([])

        # Should handle gracefully
        assert result is not None

    def test_single_node_cluster(self):
        """Single node cluster should return that node."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        node = MagicMock()
        node.id = "node1"
        node.hdv = BinaryHDV.random(1024)
        node.ltp_strength = 0.7

        representative, pruned = consolidator.merge_cluster([node])

        assert representative.id == "node1"
        assert pruned == []

    def test_selects_highest_ltp_as_representative(self):
        """Should select node with highest LTP as representative."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        node1 = MagicMock()
        node1.id = "node1"
        node1.hdv = BinaryHDV.random(1024)
        node1.ltp_strength = 0.5
        node1.access_count = 10

        node2 = MagicMock()
        node2.id = "node2"
        node2.hdv = BinaryHDV.random(1024)
        node2.ltp_strength = 0.8  # Highest
        node2.access_count = 5

        node3 = MagicMock()
        node3.id = "node3"
        node3.hdv = BinaryHDV.random(1024)
        node3.ltp_strength = 0.6
        node3.access_count = 7

        representative, pruned = consolidator.merge_cluster([node1, node2, node3])

        assert representative.id == "node2"
        assert len(pruned) == 2

    def test_updates_representative_hdv(self):
        """Should update representative HDV to bundled proto-vector."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        node1 = MagicMock()
        node1.id = "node1"
        node1.hdv = BinaryHDV.from_seed("vec1", 1024)
        node1.ltp_strength = 0.8
        node1.access_count = 5
        node1.metadata = {}

        node2 = MagicMock()
        node2.id = "node2"
        node2.hdv = BinaryHDV.from_seed("vec2", 1024)
        node2.ltp_strength = 0.7
        node2.access_count = 3

        original_hdv = node1.hdv
        representative, _ = consolidator.merge_cluster([node1, node2])

        # HDV should be updated (bundled)
        # It may not equal original due to bundling
        assert representative.hdv.dimension == original_hdv.dimension

    def test_aggregates_access_count(self):
        """Should sum access counts from all nodes."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        nodes = []
        total_access = 0
        for i in range(3):
            node = MagicMock()
            node.id = f"node{i}"
            node.hdv = BinaryHDV.random(1024)
            node.ltp_strength = 0.5 + i * 0.1
            node.access_count = (i + 1) * 10
            node.metadata = {}
            nodes.append(node)
            total_access += node.access_count

        representative, _ = consolidator.merge_cluster(nodes)

        assert representative.access_count == total_access

    def test_boosts_ltp_by_cluster_size(self):
        """Should boost LTP based on cluster size."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        nodes = []
        for i in range(5):
            node = MagicMock()
            node.id = f"node{i}"
            node.hdv = BinaryHDV.random(1024)
            node.ltp_strength = 0.7
            node.access_count = 1
            node.metadata = {}
            nodes.append(node)

        original_ltp = nodes[0].ltp_strength
        representative, _ = consolidator.merge_cluster(nodes)

        # LTP should be boosted
        assert representative.ltp_strength > original_ltp
        # But capped at 1.0
        assert representative.ltp_strength <= 1.0

    def test_updates_consolidation_metadata(self):
        """Should update metadata with consolidation history."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        nodes = []
        for i in range(3):
            node = MagicMock()
            node.id = f"node{i}"
            node.hdv = BinaryHDV.random(1024)
            node.ltp_strength = 0.5 + i * 0.1
            node.access_count = 1
            node.metadata = {}
            nodes.append(node)

        representative, pruned = consolidator.merge_cluster(nodes)

        assert "consolidation_history" in representative.metadata
        assert len(representative.metadata["consolidation_history"]) == 1
        history = representative.metadata["consolidation_history"][0]
        assert history["merged_count"] == 2
        assert "merged_ids" in history
        assert "timestamp" in history

    def test_ltp_boost_proportional_to_cluster_size(self):
        """LTP boost should be proportional to cluster size."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        base_ltp = 0.5

        # Small cluster
        small_nodes = []
        for i in range(2):
            node = MagicMock()
            node.id = f"small{i}"
            node.hdv = BinaryHDV.random(1024)
            node.ltp_strength = base_ltp
            node.access_count = 1
            node.metadata = {}
            small_nodes.append(node)

        rep_small, _ = consolidator.merge_cluster(small_nodes)

        # Large cluster
        large_nodes = []
        for i in range(10):
            node = MagicMock()
            node.id = f"large{i}"
            node.hdv = BinaryHDV.random(1024)
            node.ltp_strength = base_ltp
            node.access_count = 1
            node.metadata = {}
            large_nodes.append(node)

        rep_large, _ = consolidator.merge_cluster(large_nodes)

        # Larger cluster should get more boost
        assert rep_large.ltp_strength > rep_small.ltp_strength


class TestConsolidateTier:
    """Test tier consolidation."""

    @pytest.mark.asyncio
    async def test_consolidate_hot_tier(self):
        """Should consolidate hot tier."""
        mock_tier_manager = MagicMock()
        mock_tier_manager.get_hot_snapshot = AsyncMock(return_value=[
            MagicMock(id=f"node{i}", hdv=BinaryHDV.from_seed(f"seed{i}", 1024))
            for i in range(10)
        ])
        mock_tier_manager.delete_memory = AsyncMock(return_value=True)

        consolidator = SemanticConsolidator(mock_tier_manager)

        stats = await consolidator.consolidate_tier("hot", threshold=0.85)

        assert "nodes_processed" in stats
        assert "clusters_found" in stats
        assert stats["nodes_processed"] == 10

    @pytest.mark.asyncio
    async def test_consolidate_warm_tier(self):
        """Should consolidate warm tier."""
        mock_tier_manager = MagicMock()
        mock_tier_manager.list_warm = AsyncMock(return_value=[
            MagicMock(id=f"node{i}", hdv=BinaryHDV.from_seed(f"seed{i}", 1024))
            for i in range(5)
        ])
        mock_tier_manager.delete_memory = AsyncMock(return_value=True)

        consolidator = SemanticConsolidator(mock_tier_manager)

        stats = await consolidator.consolidate_tier("warm", threshold=0.85)

        assert stats["nodes_processed"] == 5

    @pytest.mark.asyncio
    async def test_consolidate_unknown_tier(self):
        """Should return empty stats for unknown tier."""
        mock_tier_manager = MagicMock()
        consolidator = SemanticConsolidator(mock_tier_manager)

        stats = await consolidator.consolidate_tier("unknown", threshold=0.85)

        assert stats["nodes_processed"] == 0
        assert stats["clusters_found"] == 0

    @pytest.mark.asyncio
    async def test_consolidate_returns_stats(self):
        """Should return proper statistics."""
        mock_tier_manager = MagicMock()

        # Create similar nodes that will cluster
        base_vec = BinaryHDV.from_seed("base", 1024)
        nodes = []
        for i in range(5):
            node = MagicMock()
            node.id = f"node{i}"
            node.hdv = base_vec  # All same -> will cluster
            node.ltp_strength = 0.5
            node.access_count = 1
            node.metadata = {}
            nodes.append(node)

        mock_tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_tier_manager.delete_memory = AsyncMock(return_value=True)

        consolidator = SemanticConsolidator(mock_tier_manager)

        stats = await consolidator.consolidate_tier("hot", threshold=0.90)

        assert stats["nodes_processed"] == 5
        # Should have found at least one cluster
        assert stats["clusters_found"] >= 1

    @pytest.mark.asyncio
    async def test_consolidate_handles_delete_failures(self):
        """Should continue if some deletions fail."""
        mock_tier_manager = MagicMock()

        nodes = []
        for i in range(3):
            node = MagicMock()
            node.id = f"node{i}"
            node.hdv = BinaryHDV.from_seed("base", 1024)
            node.ltp_strength = 0.5
            node.access_count = 1
            node.metadata = {}
            nodes.append(node)

        mock_tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        # First delete succeeds, others fail
        mock_tier_manager.delete_memory = AsyncMock(
            side_effect=[True, Exception("delete failed"), Exception("delete failed")]
        )

        consolidator = SemanticConsolidator(mock_tier_manager)

        # Should not raise
        stats = await consolidator.consolidate_tier("hot", threshold=0.90)

        assert stats["nodes_processed"] == 3


class TestConsolidationService:
    """Test ConsolidationService class."""

    def test_consolidate_memories_method_exists(self):
        """ConsolidationService should have consolidate_memories method."""
        service = ConsolidationService()
        assert hasattr(service, 'consolidate_memories')

    def test_consolidate_with_mock_engine(self):
        """Should work with mock engine."""
        service = ConsolidationService()

        mock_engine = MagicMock()
        now = datetime.now(timezone.utc)

        # Create mock nodes
        for i in range(5):
            node = MagicMock()
            node.created_at = now - timedelta(days=10)  # Old enough
            node.metadata = {"tags": ["test", f"tag{i}"]}
            node.get_free_energy_score = MagicMock(return_value=0.1)  # Low energy

            mock_engine.memory_nodes = {f"node{i}": node}

        # Mock soul
        mock_engine.soul = MagicMock()
        mock_engine.soul.append_to_concept = MagicMock()

        result = service.consolidate_memories(mock_engine, min_age_days=7, threshold=0.2)

        # Should have consolidated some nodes
        assert isinstance(result, list)

    def test_consolidate_respects_age_threshold(self):
        """Should only consolidate nodes old enough."""
        service = ConsolidationService()
        mock_engine = MagicMock()

        now = datetime.now(timezone.utc)

        old_node = MagicMock()
        old_node.created_at = now - timedelta(days=10)
        old_node.metadata = {}
        old_node.get_free_energy_score = MagicMock(return_value=0.1)

        young_node = MagicMock()
        young_node.created_at = now - timedelta(days=1)  # Too young
        young_node.metadata = {}
        young_node.get_free_energy_score = MagicMock(return_value=0.1)

        mock_engine.memory_nodes = {"old": old_node, "young": young_node}
        mock_engine.soul = MagicMock()
        mock_engine.soul.append_to_concept = MagicMock()

        result = service.consolidate_memories(mock_engine, min_age_days=7, threshold=0.2)

        # Only old node should be consolidated
        assert "old" in result
        assert "young" not in result

    def test_consolidate_respects_energy_threshold(self):
        """Should only consolidate nodes with low energy."""
        service = ConsolidationService()
        mock_engine = MagicMock()

        now = datetime.now(timezone.utc) - timedelta(days=10)

        low_energy = MagicMock()
        low_energy.created_at = now
        low_energy.metadata = {}
        low_energy.get_free_energy_score = MagicMock(return_value=0.1)  # Low

        high_energy = MagicMock()
        high_energy.created_at = now
        high_energy.metadata = {}
        high_energy.get_free_energy_score = MagicMock(return_value=0.5)  # Too high

        mock_engine.memory_nodes = {"low": low_energy, "high": high_energy}
        mock_engine.soul = MagicMock()
        mock_engine.soul.append_to_concept = MagicMock()

        result = service.consolidate_memories(mock_engine, min_age_days=7, threshold=0.2)

        # Only low energy node should be consolidated
        assert "low" in result
        assert "high" not in result

    def test_consolidate_handles_missing_metadata(self):
        """Should handle nodes without tags metadata."""
        service = ConsolidationService()
        mock_engine = MagicMock()

        now = datetime.now(timezone.utc) - timedelta(days=10)

        node_no_tags = MagicMock()
        node_no_tags.created_at = now
        node_no_tags.metadata = {}  # No tags
        node_no_tags.get_free_energy_score = MagicMock(return_value=0.1)

        mock_engine.memory_nodes = {"notags": node_no_tags}
        mock_engine.soul = MagicMock()
        mock_engine.soul.append_to_concept = MagicMock()

        # Should not crash
        result = service.consolidate_memories(mock_engine)
        assert isinstance(result, list)

    def test_consolidate_handles_non_list_tags(self):
        """Should handle non-list tags (e.g., string or None)."""
        service = ConsolidationService()
        mock_engine = MagicMock()

        now = datetime.now(timezone.utc) - timedelta(days=10)

        node_string_tags = MagicMock()
        node_string_tags.created_at = now
        node_string_tags.metadata = {"tags": "single_tag"}  # String, not list
        node_string_tags.get_free_energy_score = MagicMock(return_value=0.1)

        mock_engine.memory_nodes = {"stringtags": node_string_tags}
        mock_engine.soul = MagicMock()
        mock_engine.soul.append_to_concept = MagicMock()

        # Should not crash
        result = service.consolidate_memories(mock_engine)
        assert isinstance(result, list)


class TestConsolidationEdgeCases:
    """Test edge cases."""

    def test_handles_naive_datetime(self):
        """Should handle naive datetime (no timezone)."""
        service = ConsolidationService()
        mock_engine = MagicMock()

        # Naive datetime
        from datetime import datetime
        naive_time = datetime.now() - timedelta(days=10)

        node = MagicMock()
        node.created_at = naive_time
        node.metadata = {}
        node.get_free_energy_score = MagicMock(return_value=0.1)

        mock_engine.memory_nodes = {"naive": node}
        mock_engine.soul = MagicMock()
        mock_engine.soul.append_to_concept = MagicMock()

        # Should not crash
        result = service.consolidate_memories(mock_engine)
        assert isinstance(result, list)

    def test_handles_exception_during_consolidation(self):
        """Should continue if one node fails."""
        service = ConsolidationService()
        mock_engine = MagicMock()

        now = datetime.now(timezone.utc) - timedelta(days=10)

        good_node = MagicMock()
        good_node.created_at = now
        good_node.metadata = {}
        good_node.get_free_energy_score = MagicMock(return_value=0.1)

        bad_node = MagicMock()
        bad_node.created_at = now
        bad_node.metadata = {}
        bad_node.get_free_energy_score = MagicMock(side_effect=Exception("error"))

        mock_engine.memory_nodes = {"good": good_node, "bad": bad_node}
        mock_engine.soul = MagicMock()
        mock_engine.soul.append_to_concept = MagicMock()

        # Should not crash, should consolidate good node
        result = service.consolidate_memories(mock_engine)
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
