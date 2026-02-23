"""
Example usage and tests for the Association Network Module.

This file demonstrates how to use the association network for:
1. Creating and managing memory associations
2. Strengthening associations on co-retrieval
3. Querying for related memories
4. Visualizing the association network
5. Exporting as GraphQL

Run with: pytest tests/test_associations_example.py -v
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil

import pytest

from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode
from mnemocore.cognitive.associations import (
    AssociationType,
    AssociationConfig,
    AssociationEdge,
    AssociationStrengthener,
    AssociationsNetwork,
    AssociationRecallIntegrator,
    create_associations_network,
    reinforce_associations,
    find_related_memories,
)


# Module-level fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_nodes():
    """Create sample memory nodes for testing."""
    nodes = []
    contents = [
        "The cat sat on the mat",
        "Dogs are loyal pets",
        "Birds fly in the sky",
        "Fish swim in water",
        "Python is a programming language",
    ]
    for i, content in enumerate(contents):
        node = MemoryNode(
            id=f"node_{i}",
            hdv=BinaryHDV.from_seed(content, 1024),
            content=content,
            created_at=datetime.now(timezone.utc),
        )
        nodes.append(node)
    return nodes


class TestAssociationsNetwork:
    """Test suite for the association network."""

    @pytest.fixture
    def network(self, temp_dir):
        """Create an association network for testing."""
        config = AssociationConfig(
            persist_path=str(temp_dir / "associations.json"),
            auto_save=False,  # Disable auto-save during tests
            decay_enabled=False,  # Disable decay for predictable tests
        )
        return AssociationsNetwork(config=config, storage_dir=str(temp_dir))

    def test_network_creation(self, network):
        """Test that an association network can be created."""
        assert network is not None
        assert len(network) == 0
        assert network.compute_metrics().node_count == 0

    def test_add_nodes(self, network, sample_nodes):
        """Test adding nodes to the network."""
        for node in sample_nodes:
            result = network.add_node(node)
            assert result is True

        assert len(network) == 5
        metrics = network.compute_metrics()
        assert metrics.node_count == 5

    def test_add_association(self, network, sample_nodes):
        """Test creating associations between nodes."""
        for node in sample_nodes:
            network.add_node(node)

        edge = network.add_association(
            source_id="node_0",
            target_id="node_1",
            strength=0.5,
            association_type=AssociationType.SEMANTIC,
        )

        assert edge is not None
        assert edge.source_id == "node_0"
        assert edge.target_id == "node_1"
        assert edge.strength == 0.5
        assert edge.association_type == AssociationType.SEMANTIC

    def test_reinforce_association(self, network, sample_nodes):
        """Test strengthening an association."""
        for node in sample_nodes:
            network.add_node(node)

        # Create initial association
        network.add_association("node_0", "node_1", strength=0.3)

        # Reinforce it
        edge = network.reinforce("node_0", "node_1", context={"co_retrieval": True})

        assert edge is not None
        assert edge.strength > 0.3  # Should be stronger

    def test_get_associations(self, network, sample_nodes):
        """Test retrieving associations for a node."""
        for node in sample_nodes:
            network.add_node(node)

        # Create multiple associations for node_0
        network.add_association("node_0", "node_1", strength=0.8)
        network.add_association("node_0", "node_2", strength=0.5)
        network.add_association("node_0", "node_3", strength=0.3)

        associations = network.get_associations("node_0", min_strength=0.4)

        assert len(associations) == 2  # Only 0.8 and 0.5 are >= 0.4
        assert associations[0].strength >= associations[1].strength  # Should be sorted

    def test_find_shortest_path(self, network, sample_nodes):
        """Test finding shortest path between nodes."""
        for node in sample_nodes:
            network.add_node(node)

        # Create a chain: 0 -> 1 -> 2 -> 3
        network.add_association("node_0", "node_1", strength=0.5)
        network.add_association("node_1", "node_2", strength=0.5)
        network.add_association("node_2", "node_3", strength=0.5)

        path = network.find_shortest_path("node_0", "node_3")

        assert path is not None
        assert len(path) == 4  # 0 -> 1 -> 2 -> 3
        assert path[0] == "node_0"
        assert path[-1] == "node_3"

    def test_find_clusters(self, network, sample_nodes):
        """Test finding clusters of associated memories."""
        for node in sample_nodes:
            network.add_node(node)

        # Create a tight cluster: 0-1-2 all connected
        network.add_association("node_0", "node_1", strength=0.8)
        network.add_association("node_1", "node_2", strength=0.8)
        network.add_association("node_0", "node_2", strength=0.7)

        # node_3 and node_4 are isolated

        clusters = network.find_clusters(min_cluster_size=3, min_strength=0.5)

        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_persistence(self, temp_dir, sample_nodes):
        """Test saving and loading the association network."""
        # Create and populate network
        config = AssociationConfig(
            persist_path=str(temp_dir / "associations.json"),
            auto_save=False,
        )
        network1 = AssociationsNetwork(config=config, storage_dir=str(temp_dir))

        for node in sample_nodes:
            network1.add_node(node)
        network1.add_association("node_0", "node_1", strength=0.7)

        # Save
        assert network1.save() is True

        # Load into new network
        config2 = AssociationConfig(
            persist_path=str(temp_dir / "associations.json"),
            auto_save=False,
        )
        network2 = AssociationsNetwork(config=config2, storage_dir=str(temp_dir))

        assert len(network2) == 5
        edge = network2.get_edge("node_0", "node_1")
        assert edge is not None
        assert edge.strength == 0.7


class TestAssociationStrengthener:
    """Test suite for the AssociationStrengthener."""

    @pytest.fixture
    def strengthener(self):
        """Create an association strengthener for testing."""
        config = AssociationConfig(
            base_strengthen_amount=0.1,
            co_retrieval_boost=0.2,
        )
        return AssociationStrengthener(config=config)

    @pytest.fixture
    def pair_nodes(self):
        """Create a pair of sample memory nodes."""
        node_a = MemoryNode(
            id="a",
            hdv=BinaryHDV.from_seed("alpha", 1024),
            content="alpha content",
            created_at=datetime.now(timezone.utc),
            access_count=5,
        )
        node_b = MemoryNode(
            id="b",
            hdv=BinaryHDV.from_seed("beta", 1024),
            content="beta content",
            created_at=datetime.now(timezone.utc),
            access_count=3,
        )
        return node_a, node_b

    def test_calculate_strengthen_amount(self, strengthener, pair_nodes):
        """Test calculation of strengthen amount."""
        node_a, node_b = pair_nodes

        amount = strengthener.calculate_strengthen_amount(
            node_a, node_b,
            context={"co_retrieval": True}
        )

        # Base amount + co-retrieval boost
        # The calculation also includes access_count factor (avg_access = 4, so no boost since < 10)
        # Expected is base * (1 + co_retrieval_boost) = 0.1 * 1.2 = 0.12
        # Plus possibly semantic similarity boost (0.1 * similarity_bonus if similarity > 0.6)
        # We just check it's in a reasonable range
        assert 0.1 <= amount <= 0.25


class TestAssociationRecallIntegrator:
    """Test suite for the AssociationRecallIntegrator."""

    @pytest.fixture
    def integrator(self, temp_dir):
        """Create an integrator with a test network."""
        config = AssociationConfig(auto_save=False)
        network = AssociationsNetwork(config=config, storage_dir=str(temp_dir))
        return AssociationRecallIntegrator(
            network=network,
            auto_strengthen=True,
            strengthen_threshold=2,
        )

    def test_on_recall_strengthening(self, integrator, sample_nodes):
        """Test that co-retrieval strengthens associations."""
        # Add nodes to network
        for node in sample_nodes[:3]:
            integrator.network.add_node(node)

        # Simulate co-retrieval
        integrator.on_recall(sample_nodes[:3], query="test query")

        # Check that associations were created/strengthened
        associations = integrator.network.get_associations("node_0")
        assert len(associations) >= 2  # Should have associations to node_1 and node_2


class TestGraphQLExport:
    """Test GraphQL schema generation and export."""

    @pytest.fixture
    def network(self, temp_dir):
        """Create an association network for GraphQL testing."""
        config = AssociationConfig(
            persist_path=str(temp_dir / "associations.json"),
            auto_save=False,
        )
        return AssociationsNetwork(config=config, storage_dir=str(temp_dir))

    def test_graphql_schema_generation(self, network):
        """Test that GraphQL schema can be generated."""
        schema = network.to_graphql_schema()

        assert schema is not None
        assert "type MemoryNode" in schema
        assert "type AssociationEdge" in schema
        assert "type AssociationsQuery" in schema
        assert "type AssociationsMutation" in schema

    def test_graphql_data_export(self, network, sample_nodes):
        """Test exporting data in GraphQL format."""
        for node in sample_nodes:
            network.add_node(node)
        network.add_association("node_0", "node_1", strength=0.6)

        data = network.to_graphql_data()

        assert "nodes" in data
        assert "edges" in data
        assert "metrics" in data
        assert len(data["nodes"]) == 5
        assert len(data["edges"]) == 1


# Example usage functions
def example_basic_usage():
    """
    Demonstrate basic usage of the association network.
    """
    # Create a network
    config = AssociationConfig(
        persist_path="./data/associations.json",
        auto_save=True,
    )
    network = AssociationsNetwork(config=config)

    # Create some memory nodes
    nodes = []
    contents = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Neural networks learn from data",
    ]

    for i, content in enumerate(contents):
        node = MemoryNode(
            id=f"mem_{i}",
            hdv=BinaryHDV.from_seed(content, 1024),
            content=content,
            created_at=datetime.now(timezone.utc),
        )
        nodes.append(node)
        network.add_node(node)

    # Create associations
    network.add_association("mem_0", "mem_1", strength=0.7, association_type=AssociationType.SEMANTIC)
    network.add_association("mem_1", "mem_2", strength=0.8, association_type=AssociationType.SEMANTIC)

    # Reinforce associations on co-retrieval
    network.reinforce("mem_0", "mem_1", context={"co_retrieval": True})

    # Query associations
    associations = network.get_associations("mem_0", min_strength=0.5)
    print(f"Found {len(associations)} associations for mem_0")

    # Find paths
    path = network.find_shortest_path("mem_0", "mem_2")
    print(f"Shortest path: {path}")

    # Get metrics
    metrics = network.compute_metrics()
    print(f"Network: {metrics.node_count} nodes, {metrics.edge_count} edges")

    # Save to disk
    network.save()


async def example_with_engine():
    """
    Demonstrate integration with HAIMEngine.

    When using HAIMEngine, associations are automatically tracked:
    - On store(): Nodes are added to the network
    - On query(): Co-retrieved memories have their associations strengthened
    """
    from mnemocore.core.engine import HAIMEngine

    # The engine initializes the association network automatically
    engine = HAIMEngine()
    await engine.initialize()

    # When the engine is initialized, the association network is available
    if hasattr(engine, 'associations'):
        # Query for associated memories
        associated = await engine.get_associated_memories(
            node_id="some_memory_id",
            max_results=10,
            min_strength=0.2,
        )
        print(f"Found {len(associated)} associated memories")

        # Get network metrics
        metrics = engine.associations.compute_metrics()
        print(f"Association network: {metrics.node_count} nodes")

    await engine.close()


if __name__ == "__main__":
    # Run a simple example
    print("Running association network example...")
    example_basic_usage()
    print("Example complete!")
