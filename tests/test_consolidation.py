"""
Tests for Semantic Consolidation (Phase 4.0+)
=============================================
Tests for SemanticConsolidator class verifying:
- Similar memories are merged correctly
- Distinct memories are preserved
- Queries find consolidated memories
- Highest strength is preserved during consolidation
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from mnemocore.core.consolidation import SemanticConsolidator
from mnemocore.core.binary_hdv import BinaryHDV, majority_bundle
from mnemocore.core.node import MemoryNode


# Helper to create similar vectors
def create_similar_vector(base: BinaryHDV, flip_bits: int = 100) -> BinaryHDV:
    """Create a vector similar to base by flipping a small number of bits."""
    bits = np.unpackbits(base.data).copy()
    dim = len(bits)

    # Flip random bits
    indices = np.random.choice(dim, size=flip_bits, replace=False)
    bits[indices] = 1 - bits[indices]

    return BinaryHDV(data=np.packbits(bits), dimension=dim)


def create_distinct_vector(dimension: int = 16384) -> BinaryHDV:
    """Create a random vector (expected ~0.5 similarity to any other)."""
    return BinaryHDV.random(dimension)


class TestSemanticConsolidator:
    """Tests for the SemanticConsolidator class."""

    @pytest.fixture
    def mock_tier_manager(self):
        """Create a mock TierManager for testing."""
        manager = MagicMock()
        manager.get_hot_snapshot = AsyncMock(return_value=[])
        manager.list_warm = AsyncMock(return_value=[])
        manager.delete_memory = AsyncMock(return_value=True)
        return manager

    @pytest.fixture
    def consolidator(self, mock_tier_manager):
        """Create a SemanticConsolidator instance."""
        return SemanticConsolidator(
            tier_manager=mock_tier_manager,
            similarity_threshold=0.85,
            min_cluster_size=2,
        )

    def test_find_clusters_empty_list(self, consolidator):
        """Test find_clusters with empty input returns empty list."""
        clusters = consolidator.find_clusters([], threshold=0.85)
        assert clusters == []

    def test_find_clusters_single_node(self, consolidator):
        """Test find_clusters with single node returns empty list."""
        node = MemoryNode(
            id="test1",
            hdv=BinaryHDV.random(16384),
            content="test content",
        )
        clusters = consolidator.find_clusters([node], threshold=0.85)
        assert clusters == []

    def test_similar_memories_are_clustered(self, consolidator):
        """Test that similar memories are grouped into clusters."""
        # Create a base vector
        base_vec = BinaryHDV.random(16384)

        # Create similar vectors (flip ~200 bits = ~1.2% different = ~98.8% similar)
        similar_vecs = [create_similar_vector(base_vec, flip_bits=200) for _ in range(3)]

        # Create nodes
        nodes = [
            MemoryNode(id=f"similar_{i}", hdv=vec, content=f"similar content {i}")
            for i, vec in enumerate(similar_vecs)
        ]

        # Find clusters with high threshold
        clusters = consolidator.find_clusters(nodes, threshold=0.95)

        # All similar nodes should be in one cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_distinct_memories_are_preserved(self, consolidator):
        """Test that distinct memories form separate clusters or no clusters."""
        # Create distinct vectors (random = ~50% similar)
        distinct_vecs = [BinaryHDV.random(16384) for _ in range(4)]

        # Create nodes
        nodes = [
            MemoryNode(id=f"distinct_{i}", hdv=vec, content=f"distinct content {i}")
            for i, vec in enumerate(distinct_vecs)
        ]

        # Find clusters with high threshold
        clusters = consolidator.find_clusters(nodes, threshold=0.85)

        # With random vectors at 0.85 threshold, no clusters should form
        # (random vectors have ~0.5 similarity)
        assert len(clusters) == 0

    def test_mixed_similar_and_distinct(self, consolidator):
        """Test with both similar and distinct memories."""
        # Create two groups of similar vectors
        base1 = BinaryHDV.random(16384)
        base2 = BinaryHDV.random(16384)

        group1 = [create_similar_vector(base1, 150) for _ in range(3)]
        group2 = [create_similar_vector(base2, 150) for _ in range(2)]

        # Create nodes
        nodes = [
            *[MemoryNode(id=f"group1_{i}", hdv=vec, content=f"group1 {i}")
              for i, vec in enumerate(group1)],
            *[MemoryNode(id=f"group2_{i}", hdv=vec, content=f"group2 {i}")
              for i, vec in enumerate(group2)],
        ]

        # Find clusters
        clusters = consolidator.find_clusters(nodes, threshold=0.95)

        # Should have 2 clusters
        assert len(clusters) == 2
        cluster_sizes = sorted([len(c) for c in clusters])
        assert cluster_sizes == [2, 3]

    def test_merge_cluster_selects_highest_ltp(self, consolidator):
        """Test that merge_cluster selects the node with highest LTP as representative."""
        base_vec = BinaryHDV.random(16384)
        similar_vecs = [create_similar_vector(base_vec, 100) for _ in range(3)]

        # Create nodes with different LTP strengths
        nodes = []
        for i, vec in enumerate(similar_vecs):
            node = MemoryNode(
                id=f"node_{i}",
                hdv=vec,
                content=f"content {i}",
                ltp_strength=0.3 + i * 0.2,  # 0.3, 0.5, 0.7
            )
            nodes.append(node)

        # Merge cluster
        representative, pruned_ids = consolidator.merge_cluster(nodes)

        # Representative should be node_2 (highest LTP = 0.7)
        assert representative.id == "node_2"
        assert len(pruned_ids) == 2
        assert "node_0" in pruned_ids
        assert "node_1" in pruned_ids
        assert "node_2" not in pruned_ids

    def test_merge_cluster_updates_metadata(self, consolidator):
        """Test that merge_cluster updates metadata correctly."""
        base_vec = BinaryHDV.random(16384)
        similar_vecs = [create_similar_vector(base_vec, 100) for _ in range(3)]

        nodes = []
        for i, vec in enumerate(similar_vecs):
            node = MemoryNode(
                id=f"node_{i}",
                hdv=vec,
                content=f"content {i}",
                ltp_strength=0.5,
                access_count=10 + i * 5,
            )
            nodes.append(node)

        # Merge cluster
        representative, _ = consolidator.merge_cluster(nodes)

        # Check metadata updates
        assert "consolidation_history" in representative.metadata
        assert representative.metadata["consolidation_history"][0]["merged_count"] == 2
        assert representative.access_count == 10 + 15 + 20  # Sum of all access counts
        assert representative.ltp_strength > 0.5  # Should be boosted

    def test_merge_cluster_produces_proto_vector(self, consolidator):
        """Test that merge_cluster creates a proper proto-vector via bundling."""
        # Create vectors that will produce a predictable bundle
        vec1 = BinaryHDV.random(16384)
        vec2 = create_similar_vector(vec1, 200)
        vec3 = create_similar_vector(vec1, 200)

        nodes = [
            MemoryNode(id="n1", hdv=vec1, content="c1", ltp_strength=0.9),
            MemoryNode(id="n2", hdv=vec2, content="c2", ltp_strength=0.5),
            MemoryNode(id="n3", hdv=vec3, content="c3", ltp_strength=0.3),
        ]

        # Expected proto vector
        expected_proto = majority_bundle([vec1, vec2, vec3])

        # Merge cluster
        representative, _ = consolidator.merge_cluster(nodes)

        # Check the proto-vector matches expected
        assert representative.hdv == expected_proto

    @pytest.mark.asyncio
    async def test_consolidate_tier_hot(self, consolidator, mock_tier_manager):
        """Test consolidate_tier with HOT tier."""
        # Create similar nodes
        base_vec = BinaryHDV.random(16384)
        similar_vecs = [create_similar_vector(base_vec, 150) for _ in range(3)]

        nodes = [
            MemoryNode(id=f"hot_{i}", hdv=vec, content=f"hot content {i}", ltp_strength=0.5 + i * 0.1)
            for i, vec in enumerate(similar_vecs)
        ]

        mock_tier_manager.get_hot_snapshot.return_value = nodes

        # Run consolidation
        stats = await consolidator.consolidate_tier("hot", threshold=0.95)

        # Check stats
        assert stats["nodes_processed"] == 3
        assert stats["clusters_found"] == 1
        assert stats["nodes_merged"] == 2
        assert stats["nodes_pruned"] == 2

        # Verify delete was called for pruned nodes
        assert mock_tier_manager.delete_memory.call_count == 2

    @pytest.mark.asyncio
    async def test_consolidate_tier_warm(self, consolidator, mock_tier_manager):
        """Test consolidate_tier with WARM tier."""
        # Create distinct nodes (no clustering expected)
        nodes = [
            MemoryNode(id=f"warm_{i}", hdv=BinaryHDV.random(16384), content=f"warm content {i}")
            for i in range(4)
        ]

        mock_tier_manager.list_warm.return_value = nodes

        # Run consolidation
        stats = await consolidator.consolidate_tier("warm", threshold=0.85)

        # With distinct vectors, no clusters should form
        assert stats["nodes_processed"] == 4
        assert stats["clusters_found"] == 0
        assert stats["nodes_merged"] == 0
        assert stats["nodes_pruned"] == 0

    @pytest.mark.asyncio
    async def test_consolidate_tier_empty(self, consolidator, mock_tier_manager):
        """Test consolidate_tier with empty tier."""
        mock_tier_manager.get_hot_snapshot.return_value = []

        stats = await consolidator.consolidate_tier("hot", threshold=0.85)

        assert stats["nodes_processed"] == 0
        assert stats["clusters_found"] == 0


class TestConsolidationIntegration:
    """Integration tests for consolidation with query finding."""

    @pytest.fixture
    def mock_tier_manager(self):
        """Create a mock TierManager with in-memory storage."""
        manager = MagicMock()
        manager.hot_storage = {}
        manager.get_hot_snapshot = AsyncMock(return_value=[])
        manager.list_warm = AsyncMock(return_value=[])
        manager.delete_memory = AsyncMock(side_effect=lambda nid: True)
        return manager

    @pytest.fixture
    def consolidator(self, mock_tier_manager):
        """Create a SemanticConsolidator instance."""
        return SemanticConsolidator(
            tier_manager=mock_tier_manager,
            similarity_threshold=0.85,
            min_cluster_size=2,
        )

    @pytest.mark.asyncio
    async def test_query_finds_consolidated_memory(self, consolidator, mock_tier_manager):
        """Test that a query can find a consolidated/merged memory."""
        # Create a cluster of similar memories about "machine learning"
        base_vec = BinaryHDV.random(16384)
        similar_vecs = [create_similar_vector(base_vec, 150) for _ in range(3)]

        nodes = [
            MemoryNode(
                id=f"ml_{i}",
                hdv=vec,
                content=f"machine learning concept {i}",
                ltp_strength=0.5 + i * 0.1,
            )
            for i, vec in enumerate(similar_vecs)
        ]

        # Set up mock to return these nodes
        mock_tier_manager.get_hot_snapshot.return_value = nodes

        # Store original vectors for later query simulation
        query_vec = similar_vecs[0]  # Query with one of the similar vectors

        # Run consolidation
        stats = await consolidator.consolidate_tier("hot", threshold=0.95)

        # Verify consolidation happened
        assert stats["nodes_merged"] == 2
        assert stats["nodes_pruned"] == 2

        # The representative should have a proto-vector that is still similar
        # to the query vector (majority bundle preserves semantic content)
        representative = max(nodes, key=lambda n: n.ltp_strength)
        similarity = query_vec.similarity(representative.hdv)

        # The proto-vector should be highly similar to the query
        # (since all cluster members were similar)
        assert similarity >= 0.90, f"Expected similarity >= 0.90, got {similarity}"

    @pytest.mark.asyncio
    async def test_consolidation_preserves_highest_strength(self, consolidator, mock_tier_manager):
        """Test that consolidation preserves and boosts the highest LTP strength."""
        base_vec = BinaryHDV.random(16384)
        similar_vecs = [create_similar_vector(base_vec, 100) for _ in range(4)]

        # Create nodes with varying LTP strengths
        ltp_values = [0.3, 0.5, 0.9, 0.4]  # Index 2 has highest
        nodes = [
            MemoryNode(
                id=f"node_{i}",
                hdv=vec,
                content=f"content {i}",
                ltp_strength=ltp_values[i],
            )
            for i, vec in enumerate(similar_vecs)
        ]

        original_highest_ltp = max(ltp_values)  # 0.9
        original_highest_id = "node_2"

        mock_tier_manager.get_hot_snapshot.return_value = nodes

        # Run consolidation
        await consolidator.consolidate_tier("hot", threshold=0.95)

        # Find the representative (should be node_2)
        representative = next(n for n in nodes if n.id == original_highest_id)

        # Verify LTP was boosted
        assert representative.ltp_strength > original_highest_ltp, \
            f"LTP should be boosted from {original_highest_ltp} to {representative.ltp_strength}"

        # Verify other nodes would be pruned
        assert representative.id == original_highest_id


class TestConsolidationThreshold:
    """Tests for threshold behavior."""

    @pytest.fixture
    def consolidator(self):
        """Create a basic consolidator."""
        manager = MagicMock()
        manager.get_hot_snapshot = AsyncMock(return_value=[])
        manager.list_warm = AsyncMock(return_value=[])
        manager.delete_memory = AsyncMock(return_value=True)
        return SemanticConsolidator(
            tier_manager=manager,
            similarity_threshold=0.85,
        )

    def test_threshold_85_clusters_similar(self, consolidator):
        """Test that 0.85 threshold correctly clusters similar memories."""
        # Create vectors with ~10% difference (0.90 similarity)
        base = BinaryHDV.random(16384)
        # Flip ~1640 bits for 10% difference
        similar = create_similar_vector(base, flip_bits=1640)

        nodes = [
            MemoryNode(id="n1", hdv=base, content="base"),
            MemoryNode(id="n2", hdv=similar, content="similar"),
        ]

        # At 0.85 threshold, these should cluster
        clusters = consolidator.find_clusters(nodes, threshold=0.85)
        assert len(clusters) == 1

    def test_threshold_85_separates_distinct(self, consolidator):
        """Test that 0.85 threshold keeps distinct memories separate."""
        # Create truly random vectors (expected ~0.5 similarity)
        nodes = [
            MemoryNode(id=f"rand_{i}", hdv=BinaryHDV.random(16384), content=f"random {i}")
            for i in range(3)
        ]

        # At 0.85 threshold, these should NOT cluster
        clusters = consolidator.find_clusters(nodes, threshold=0.85)

        # Random vectors are ~0.5 similar, so no clusters at 0.85
        assert len(clusters) == 0

    def test_threshold_motivation(self, consolidator):
        """
        Test the motivation for 0.85 threshold.

        Rationale:
        - Random binary HDVs have expected similarity ~0.5 (Kanerva, 2009)
        - Similarity >= 0.85 is well above random chance
        - For 16384 dimensions: 0.85 similarity = 2457 differing bits
        - This captures semantic kinship while avoiding false positives
        """
        # Verify random vectors cluster at ~0.5 similarity
        random_pairs_similarities = []
        for _ in range(10):
            v1 = BinaryHDV.random(16384)
            v2 = BinaryHDV.random(16384)
            sim = v1.similarity(v2)
            random_pairs_similarities.append(sim)

        avg_random_similarity = np.mean(random_pairs_similarities)

        # Random vectors should be ~0.5 similar
        assert 0.45 <= avg_random_similarity <= 0.55, \
            f"Random vectors should be ~0.5 similar, got {avg_random_similarity}"

        # Create semantically similar vectors (flip 10% of bits)
        base = BinaryHDV.random(16384)
        similar = create_similar_vector(base, flip_bits=1640)
        similar_similarity = base.similarity(similar)

        # Should be ~0.90 similar (10% flipped)
        assert similar_similarity >= 0.85, \
            f"Similar vectors should be >= 0.85 similar, got {similar_similarity}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
