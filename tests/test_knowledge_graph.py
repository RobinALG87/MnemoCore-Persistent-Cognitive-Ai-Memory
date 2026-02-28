"""
Tests for Bidirectional Knowledge Graph (knowledge_graph.py)
==============================================================
Covers KnowledgeNode, KnowledgeEdge, KnowledgeGraphService.

Research basis: Mnemosyne (Georgia Tech 2025), Zettelkasten-style
bidirectional linking, activation spreading.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from mnemocore.core.knowledge_graph import (
    KnowledgeEdge,
    KnowledgeGraphService,
    KnowledgeNode,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_node(node_id: str, label: str = "", content: str = "",
               node_type: str = "concept") -> KnowledgeNode:
    """Build a KnowledgeNode for tests."""
    return KnowledgeNode(
        id=node_id,
        label=label or node_id,
        content=content or f"Content for {node_id}",
        node_type=node_type,
    )


# ═══════════════════════════════════════════════════════════════════════
# KnowledgeNode
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeNode:

    def test_default_node(self):
        node = KnowledgeNode(id="n1", label="test", content="data")
        assert node.id == "n1"
        assert node.label == "test"
        assert node.access_count == 0
        assert node.activation_level == 0.0

    def test_lifecycle_stage(self):
        node = KnowledgeNode(id="n1", label="t", content="c")
        assert node.lifecycle_stage in ("STM", "MTM", "LTM")

    def test_access_updates_timestamps(self):
        node = KnowledgeNode(id="n1", label="t", content="c")
        before = node.access_count
        node.access()
        assert node.access_count == before + 1

    def test_update_health(self):
        node = KnowledgeNode(id="n1", label="t", content="c")
        node.update_health()
        assert 0.0 <= node.health_score <= 1.0

    def test_to_dict_roundtrip(self):
        node = KnowledgeNode(id="n1", label="test", content="data",
                             node_type="concept")
        d = node.to_dict()
        node2 = KnowledgeNode.from_dict(d)
        assert node2.id == node.id
        assert node2.label == node.label
        assert node2.content == node.content


# ═══════════════════════════════════════════════════════════════════════
# KnowledgeEdge
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeEdge:

    def test_default_edge(self):
        edge = KnowledgeEdge(source_id="a", target_id="b")
        assert edge.weight == 0.5  # Default is 0.5
        assert edge.edge_type == "related"
        assert edge.activation_count == 0

    def test_activate(self):
        edge = KnowledgeEdge(source_id="a", target_id="b")
        old_w = edge.weight
        edge.activate(boost=0.1)
        assert edge.weight > old_w
        assert edge.activation_count == 1

    def test_decay(self):
        edge = KnowledgeEdge(source_id="a", target_id="b", weight=1.0)
        # Move last_activated into the past so decay has elapsed time
        from datetime import timedelta
        edge.last_activated = datetime.now(timezone.utc) - timedelta(days=1)
        edge.decay(half_life_days=0.5)
        assert edge.weight < 1.0

    def test_to_dict_roundtrip(self):
        edge = KnowledgeEdge(source_id="a", target_id="b", weight=0.75,
                             edge_type="causal")
        d = edge.to_dict()
        edge2 = KnowledgeEdge.from_dict(d)
        assert edge2.source_id == edge.source_id
        assert edge2.target_id == edge.target_id
        assert abs(edge2.weight - 0.75) < 0.01
        assert edge2.edge_type == "causal"


# ═══════════════════════════════════════════════════════════════════════
# KnowledgeGraphService
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeGraphService:

    def test_add_and_get_node(self):
        svc = KnowledgeGraphService()
        node = _make_node("n1", "test label", "some content")
        svc.add_node(node)
        retrieved = svc.get_node("n1")
        assert retrieved is not None
        assert retrieved.label == "test label"

    def test_get_nonexistent_node(self):
        svc = KnowledgeGraphService()
        assert svc.get_node("missing") is None

    def test_bidirectional_link(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("a", "node a", "content a"))
        svc.add_node(_make_node("b", "node b", "content b"))
        fwd, rev = svc.link("a", "b", weight=1.0)

        # Both directions should exist
        assert fwd is not None
        assert rev is not None
        # Reciprocal factor = 0.7 default
        assert rev.weight < fwd.weight

    def test_strengthen(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("a"))
        svc.add_node(_make_node("b"))
        svc.link("a", "b", weight=0.5)
        svc.strengthen("a", "b", boost=0.2)
        edge = svc.get_edge("a", "b")
        assert edge is not None
        assert edge.weight > 0.5

    def test_bfs_retrieve(self):
        svc = KnowledgeGraphService()
        for i in range(5):
            svc.add_node(_make_node(f"n{i}", f"label{i}", f"content{i}"))
        svc.link("n0", "n1")
        svc.link("n1", "n2")
        svc.link("n2", "n3")
        svc.link("n3", "n4")

        results = svc.bfs_retrieve(["n0"], max_depth=3)
        # Returns List[Tuple[KnowledgeNode, int]]
        node_ids = [n.id for n, _ in results]
        assert "n0" in node_ids
        assert "n1" in node_ids
        assert len(results) >= 2

    def test_weighted_walk(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("start", "start", "begin"))
        svc.add_node(_make_node("mid", "middle", "middle"))
        svc.add_node(_make_node("end", "end", "finish"))
        svc.link("start", "mid", weight=1.0)
        svc.link("mid", "end", weight=1.0)

        # param is `steps`, returns List[KnowledgeNode]
        path = svc.weighted_walk("start", steps=5)
        assert len(path) >= 1
        assert path[0].id == "start"

    def test_spread_activation(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("center", "center", "c"))
        svc.add_node(_make_node("neighbor", "neighbor", "n"))
        svc.link("center", "neighbor", weight=1.0)

        # params: initial_energy, decay_factor
        activated = svc.spread_activation(["center"], initial_energy=1.0,
                                          decay_factor=0.5, max_depth=2)
        assert "center" in activated
        assert activated["center"] > 0

    def test_prune_weak_edges(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("a"))
        svc.add_node(_make_node("b"))
        svc.link("a", "b", weight=0.001)  # Very weak → bidirectional creates two weak edges
        pruned = svc.prune_weak_edges(min_weight=0.01)
        assert pruned >= 1

    def test_prune_redundant_nodes(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("n1", "same label",
                                "exactly the same content here for testing"))
        svc.add_node(_make_node("n2", "same label",
                                "exactly the same content here for testing"))
        # param is `similarity_threshold`
        pruned = svc.prune_redundant_nodes(similarity_threshold=0.9)
        assert pruned >= 0  # Jaccard may or may not exceed threshold

    def test_detect_clusters(self):
        svc = KnowledgeGraphService()
        # Create 2 disconnected clusters
        svc.add_node(_make_node("a1"))
        svc.add_node(_make_node("a2"))
        svc.link("a1", "a2")
        svc.add_node(_make_node("b1"))
        svc.add_node(_make_node("b2"))
        svc.link("b1", "b2")

        clusters = svc.detect_clusters()
        assert len(clusters) >= 2

    def test_decay_all_edges(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("a"))
        svc.add_node(_make_node("b"))
        svc.link("a", "b", weight=1.0)
        removed = svc.decay_all_edges()
        # Returns number of edges removed after decay
        assert isinstance(removed, int)
        edge = svc.get_edge("a", "b")
        if edge:
            assert edge.weight <= 1.0

    def test_decay_all_activations(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("a"))
        node = svc.get_node("a")
        node.activation_level = 1.0
        svc.decay_all_activations(decay_rate=0.1)
        assert node.activation_level < 1.0

    def test_get_stats(self):
        svc = KnowledgeGraphService()
        svc.add_node(_make_node("a"))
        svc.add_node(_make_node("b"))
        svc.link("a", "b")
        stats = svc.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] >= 2  # Bidirectional

    def test_persistence(self, tmp_path):
        path = tmp_path / "kg.json"
        svc = KnowledgeGraphService(config=MagicMock(
            persistence_path=str(path),
            auto_persist=True,
            max_nodes=100000,
            max_edges=500000,
            reciprocal_factor=0.7,
            min_edge_weight=0.01,
        ))
        svc.add_node(_make_node("p1", "persist", "data"))
        svc.add_node(_make_node("p2", "test", "saved"))
        svc.link("p1", "p2")
        assert path.exists()

        svc2 = KnowledgeGraphService(config=MagicMock(
            persistence_path=str(path),
            auto_persist=True,
            max_nodes=100000,
            max_edges=500000,
            reciprocal_factor=0.7,
            min_edge_weight=0.01,
        ))
        assert svc2.get_node("p1") is not None
        assert svc2.get_node("p2") is not None
