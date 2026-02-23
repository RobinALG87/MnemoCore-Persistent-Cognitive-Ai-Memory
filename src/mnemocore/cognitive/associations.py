"""
Associations Network Module
============================
Phase 6.0 - Graph-based memory association tracking and strengthening.

This module provides:
1. NetworkX-based graph representation of memory associations
2. Association strengthening on co-retrieval (Hebbian learning)
3. GraphQL schema preparation for knowledge graph export
4. Visualization tools for association networks
5. Integration with recall for automatic association strengthening

Key Concepts:
- Nodes: Memory nodes (MemoryNode instances)
- Edges: Associations between memories with strength weights
- Co-retrieval: When memories are retrieved together, their association strengthens
- Association types: semantic, temporal, causal, contextual

Author: MnemoCore Infrastructure Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Callable,
    Iterator,
)
from enum import Enum
from pathlib import Path
from functools import lru_cache
import hashlib

import numpy as np
from loguru import logger

# Try to import NetworkX (required dependency)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    logger.warning(
        "NetworkX not available. Install with: pip install networkx. "
        "Association graph functionality will be limited."
    )

# Try to import visualization libraries (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None
    logger.debug("Plotly not available. Visualization will use fallback.")

# Try to import matplotlib (optional, for static plots)
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    cm = None

from mnemocore.core.node import MemoryNode
from mnemocore.core.binary_hdv import BinaryHDV


class AssociationType(Enum):
    """Types of associations between memories."""

    SEMANTIC = "semantic"  # Similar meaning/content
    TEMPORAL = "temporal"  # Occurred close in time
    CAUSAL = "causal"  # Cause-effect relationship
    CONTEXTUAL = "contextual"  # Shared context/session
    HIERARCHICAL = "hierarchical"  # Parent-child or part-whole
    CO_OCCURRENCE = "co_occurrence"  # Retrieved together


class AssociationDirection(Enum):
    """Directionality of associations."""

    DIRECTED = "directed"  # A -> B (one-way)
    BIDIRECTIONAL = "bidirectional"  # A <-> B (two-way)
    UNDIRECTED = "undirected"  # A-B (no direction)


@dataclass
class AssociationEdge:
    """
    Represents an association edge between two memory nodes.

    Attributes:
        source_id: Source memory node ID
        target_id: Target memory node ID
        strength: Association strength [0.0, 1.0]
        association_type: Type of association
        created_at: When the association was first created
        last_strengthened: When the association was last strengthened
        fire_count: Number of times this association was activated
        metadata: Additional association metadata
    """

    source_id: str
    target_id: str
    strength: float = 0.1
    association_type: AssociationType = AssociationType.CO_OCCURRENCE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_strengthened: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fire_count: int = 0
    direction: AssociationDirection = AssociationDirection.BIDIRECTIONAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate edge attributes."""
        self.strength = max(0.0, min(1.0, self.strength))

    def key(self) -> Tuple[str, str]:
        """Get the unique key for this edge."""
        return tuple(sorted((self.source_id, self.target_id)))

    def strengthen(self, amount: float = 0.1, max_strength: float = 1.0) -> float:
        """
        Strengthen this association.

        Args:
            amount: Amount to strengthen by
            max_strength: Maximum strength cap

        Returns:
            New strength value
        """
        self.last_strengthened = datetime.now(timezone.utc)
        self.fire_count += 1

        # Apply strengthening with diminishing returns
        # New = Old + amount * (1 - Old)  [asymptotic approach to 1.0]
        self.strength = min(max_strength, self.strength + amount * (1 - self.strength))
        return self.strength

    def decay(self, half_life_days: float = 30.0) -> float:
        """
        Apply time-based decay to the association strength.

        Args:
            half_life_days: Half-life for decay

        Returns:
            New strength value
        """
        age_seconds = (datetime.now(timezone.utc) - self.last_strengthened).total_seconds()
        age_days = age_seconds / 86400.0

        decay_factor = math.exp(-(math.log(2) / half_life_days) * age_days)
        self.strength = max(0.01, self.strength * decay_factor)
        return self.strength

    def is_active(self, threshold: float = 0.1) -> bool:
        """Check if this association is above activation threshold."""
        return self.strength >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strength": self.strength,
            "association_type": self.association_type.value,
            "created_at": self.created_at.isoformat(),
            "last_strengthened": self.last_strengthened.isoformat(),
            "fire_count": self.fire_count,
            "direction": self.direction.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssociationEdge":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            strength=data.get("strength", 0.1),
            association_type=AssociationType(data.get("association_type", "co_occurrence")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            last_strengthened=datetime.fromisoformat(data.get("last_strengthened", datetime.now(timezone.utc).isoformat())),
            fire_count=data.get("fire_count", 0),
            direction=AssociationDirection(data.get("direction", "bidirectional")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AssociationConfig:
    """Configuration for association network behavior."""

    # Strengthening parameters
    base_strengthen_amount: float = 0.1
    co_retrieval_boost: float = 0.15
    max_association_strength: float = 1.0

    # Decay parameters
    decay_enabled: bool = True
    half_life_days: float = 30.0
    min_strength: float = 0.01

    # Graph parameters
    max_edges_per_node: int = 100
    min_edge_threshold: float = 0.05

    # Persistence
    persist_path: Optional[str] = None
    auto_save: bool = True
    save_interval_seconds: int = 300

    # Visualization
    max_nodes_for_viz: int = 500
    default_layout: str = "spring"  # spring, circular, kamada_kawai, spectral


@dataclass
class GraphMetrics:
    """Metrics describing the association graph structure."""

    node_count: int = 0
    edge_count: int = 0
    avg_degree: float = 0.0
    density: float = 0.0
    avg_clustering: float = 0.0
    connected_components: int = 0
    largest_component_size: int = 0
    avg_path_length: float = 0.0
    diameter: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AssociationStrengthener:
    """
    Handles strengthening of associations based on co-retrieval.

    Implements Hebbian learning: "neurons that fire together wire together"
    """

    def __init__(
        self,
        config: Optional[AssociationConfig] = None,
    ):
        """
        Initialize the association strengthener.

        Args:
            config: Configuration for association behavior
        """
        self.config = config or AssociationConfig()
        self._strengthen_history: List[Tuple[str, str, float]] = []

    def calculate_strengthen_amount(
        self,
        node_a: MemoryNode,
        node_b: MemoryNode,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the amount to strengthen an association.

        Factors:
        - Base strengthen amount
        - Co-retrieval boost
        - Semantic similarity (if HDV available)
        - Temporal proximity
        - Access frequency

        Args:
            node_a: First memory node
            node_b: Second memory node
            context: Optional context information

        Returns:
            Amount to strengthen by [0.0, 1.0]
        """
        amount = self.config.base_strengthen_amount

        # Co-retrieval boost
        if context and context.get("co_retrieval"):
            amount *= (1.0 + self.config.co_retrieval_boost)

        # Semantic similarity boost
        if hasattr(node_a, 'hdv') and hasattr(node_b, 'hdv'):
            if node_a.hdv and node_b.hdv:
                similarity = node_a.hdv.similarity(node_b.hdv)
                # Boost for semantically similar memories
                if similarity > 0.6:
                    amount *= (1.0 + (similarity - 0.6))

        # Temporal proximity boost
        time_diff = abs(node_a.created_at.timestamp() - node_b.created_at.timestamp())
        if time_diff < 3600:  # Within 1 hour
            amount *= 1.2
        elif time_diff < 86400:  # Within 1 day
            amount *= 1.1

        # Access frequency factor
        avg_access = (node_a.access_count + node_b.access_count) / 2
        if avg_access > 10:
            amount *= 1.1

        return min(1.0, amount)

    def reinforce(
        self,
        node_a: Union[str, MemoryNode],
        node_b: Union[str, MemoryNode],
        network: "AssociationsNetwork",
        association_type: AssociationType = AssociationType.CO_OCCURRENCE,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AssociationEdge]:
        """
        Reinforce the association between two nodes.

        This is the main entry point for Hebbian learning.
        Call this when two memories are retrieved together.

        Args:
            node_a: First node (ID or MemoryNode)
            node_b: Second node (ID or MemoryNode)
            network: The associations network to modify
            association_type: Type of association
            context: Optional context information

        Returns:
            The updated AssociationEdge, or None if nodes not found
        """
        # Extract IDs
        id_a = node_a if isinstance(node_a, str) else node_a.id
        id_b = node_b if isinstance(node_b, str) else node_b.id

        if id_a == id_b:
            return None  # Don't self-associate

        # Get or create edge
        edge = network.get_edge(id_a, id_b)
        if edge is None:
            # Create new association
            edge = network.add_association(
                id_a, id_b,
                strength=self.config.min_strength,
                association_type=association_type,
            )
        else:
            # Update type if provided
            edge.association_type = association_type

        # Calculate and apply strengthening
        if isinstance(node_a, MemoryNode) and isinstance(node_b, MemoryNode):
            amount = self.calculate_strengthen_amount(node_a, node_b, context)
        else:
            amount = self.config.base_strengthen_amount

        new_strength = edge.strengthen(amount, self.config.max_association_strength)

        # Track for analysis
        self._strengthen_history.append((id_a, id_b, new_strength))

        # Trim history if too large
        if len(self._strengthen_history) > 10000:
            self._strengthen_history = self._strengthen_history[-5000:]

        return edge

    def get_strengthen_history(
        self,
        node_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Tuple[str, str, float]]:
        """
        Get recent strengthening operations.

        Args:
            node_id: Filter to operations involving this node
            limit: Maximum number of entries to return

        Returns:
            List of (node_a, node_b, strength) tuples
        """
        if node_id:
            filtered = [
                (a, b, s) for a, b, s in self._strengthen_history
                if node_id in (a, b)
            ]
            return filtered[-limit:]
        return self._strengthen_history[-limit:]


class AssociationsNetwork:
    """
    Graph-based representation of memory associations.

    Uses NetworkX for efficient graph operations and provides:
    - Add/remove nodes and associations
    - Query associations by strength, type, or path
    - Find clusters and communities
    - Compute graph metrics
    - Persist to disk
    """

    def __init__(
        self,
        config: Optional[AssociationConfig] = None,
        storage_dir: Optional[str] = None,
    ):
        """
        Initialize the associations network.

        Args:
            config: Configuration for association behavior
            storage_dir: Directory for persistence
        """
        self.config = config or AssociationConfig()

        # Set up storage
        if storage_dir:
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.config.persist_path = str(self.storage_dir / "associations.json")
        elif self.config.persist_path:
            self.storage_dir = Path(self.config.persist_path).parent
        else:
            self.storage_dir = None

        # Initialize NetworkX graph
        if NETWORKX_AVAILABLE:
            self.graph = nx.MultiDiGraph()
        else:
            self.graph = None
            self._nodes: Dict[str, MemoryNode] = {}
            self._edges: Dict[Tuple[str, str], AssociationEdge] = {}

        # Strengthener for Hebbian learning
        self.strengthener = AssociationStrengthener(self.config)

        # Metrics cache
        self._metrics_cache: Optional[GraphMetrics] = None
        self._metrics_dirty = True

        # Load persisted data if available
        if self.storage_dir:
            self._load_from_disk()

    # ======================================================================
    # Node Management
    # ======================================================================

    def add_node(self, node: MemoryNode) -> bool:
        """
        Add a memory node to the association graph.

        Args:
            node: MemoryNode to add

        Returns:
            True if node was added, False if already exists
        """
        if NETWORKX_AVAILABLE:
            if node.id in self.graph:
                return False

            self.graph.add_node(
                node.id,
                content=node.content[:200],  # Truncated for storage
                created_at=node.created_at.isoformat(),
                tier=node.tier,
                ltp_strength=node.ltp_strength,
                access_count=node.access_count,
            )
        else:
            if node.id in self._nodes:
                return False
            self._nodes[node.id] = node

        self._metrics_dirty = True
        return True

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its associations from the graph.

        Args:
            node_id: ID of node to remove

        Returns:
            True if node was removed
        """
        if NETWORKX_AVAILABLE:
            if node_id not in self.graph:
                return False
            self.graph.remove_node(node_id)
        else:
            if node_id not in self._nodes:
                return False
            del self._nodes[node_id]
            # Remove associated edges
            to_remove = [k for k in self._edges if node_id in k]
            for key in to_remove:
                del self._edges[key]

        self._metrics_dirty = True
        return True

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        if NETWORKX_AVAILABLE:
            return node_id in self.graph
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get node attributes.

        Args:
            node_id: Node ID to look up

        Returns:
            Node attributes dict, or None if not found
        """
        if NETWORKX_AVAILABLE:
            if node_id in self.graph:
                return dict(self.graph.nodes[node_id])
        else:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                return {
                    "id": node.id,
                    "content": node.content,
                    "created_at": node.created_at.isoformat(),
                    "tier": node.tier,
                    "ltp_strength": node.ltp_strength,
                    "access_count": node.access_count,
                }
        return None

    def get_all_nodes(self) -> List[str]:
        """Get list of all node IDs in the graph."""
        if NETWORKX_AVAILABLE:
            return list(self.graph.nodes())
        return list(self._nodes.keys())

    # ======================================================================
    # Association (Edge) Management
    # ======================================================================

    def add_association(
        self,
        source_id: str,
        target_id: str,
        strength: float = 0.1,
        association_type: AssociationType = AssociationType.CO_OCCURRENCE,
        direction: AssociationDirection = AssociationDirection.BIDIRECTIONAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AssociationEdge]:
        """
        Add an association between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            strength: Initial association strength
            association_type: Type of association
            direction: Directionality of the association
            metadata: Optional metadata

        Returns:
            The created AssociationEdge, or None if nodes don't exist
        """
        if not self.has_node(source_id) or not self.has_node(target_id):
            logger.warning(f"Cannot add association: missing node {source_id} or {target_id}")
            return None

        if source_id == target_id:
            return None

        edge = AssociationEdge(
            source_id=source_id,
            target_id=target_id,
            strength=strength,
            association_type=association_type,
            direction=direction,
            metadata=metadata or {},
        )

        if NETWORKX_AVAILABLE:
            # Add edge with attributes
            self.graph.add_edge(
                source_id,
                target_id,
                strength=strength,
                association_type=association_type.value,
                direction=direction.value,
                created_at=edge.created_at.isoformat(),
                fire_count=0,
            )
        else:
            self._edges[edge.key()] = edge

        self._metrics_dirty = True
        return edge

    def get_edge(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[AssociationEdge]:
        """
        Get an association edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            AssociationEdge if found, None otherwise
        """
        if NETWORKX_AVAILABLE:
            try:
                edge_data = self.graph.get_edge_data(source_id, target_id)
                if edge_data:
                    # Handle MultiDiGraph (multiple edges possible)
                    if isinstance(edge_data, dict):
                        # Get the first edge if multiple
                        for key, data in edge_data.items():
                            return AssociationEdge(
                                source_id=source_id,
                                target_id=target_id,
                                strength=data.get("strength", 0.1),
                                association_type=AssociationType(data.get("association_type", "co_occurrence")),
                                created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
                                fire_count=data.get("fire_count", 0),
                                direction=AssociationDirection(data.get("direction", "bidirectional")),
                            )
            except nx.NetworkXError:
                return None
        else:
            key = tuple(sorted((source_id, target_id)))
            return self._edges.get(key)

        return None

    def remove_association(self, source_id: str, target_id: str) -> bool:
        """
        Remove an association between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            True if association was removed
        """
        if NETWORKX_AVAILABLE:
            try:
                self.graph.remove_edge(source_id, target_id)
                self._metrics_dirty = True
                return True
            except nx.NetworkXError:
                return False
        else:
            key = tuple(sorted((source_id, target_id)))
            if key in self._edges:
                del self._edges[key]
                self._metrics_dirty = True
                return True
        return False

    def get_associations(
        self,
        node_id: str,
        min_strength: float = 0.0,
        association_type: Optional[AssociationType] = None,
        limit: int = 100,
    ) -> List[AssociationEdge]:
        """
        Get all associations for a node, optionally filtered.

        Args:
            node_id: Node ID to get associations for
            min_strength: Minimum strength threshold
            association_type: Filter by association type
            limit: Maximum number of associations to return

        Returns:
            List of AssociationEdge objects, sorted by strength descending
        """
        results = []
        seen_targets = set()  # Track seen targets to avoid duplicates

        if NETWORKX_AVAILABLE:
            if node_id not in self.graph:
                return results

            # Get all edges involving this node (both outgoing and incoming)
            # Use edges() to get all edges connected to this node
            for source, target, key, edge_data in self.graph.edges(node_id, keys=True, data=True):
                # Skip self-loops
                if source == target:
                    continue

                # Determine the other node in the edge
                other_id = target if source == node_id else source

                # Skip if we've already seen this target
                if other_id in seen_targets:
                    continue
                seen_targets.add(other_id)

                strength = edge_data.get("strength", 0.0)
                if strength >= min_strength:
                    if association_type is None or edge_data.get("association_type") == association_type.value:
                        edge = AssociationEdge(
                            source_id=source,
                            target_id=target,
                            strength=strength,
                            association_type=AssociationType(edge_data.get("association_type", "co_occurrence")),
                            created_at=datetime.fromisoformat(edge_data.get("created_at", datetime.now(timezone.utc).isoformat())),
                            fire_count=edge_data.get("fire_count", 0),
                            direction=AssociationDirection(edge_data.get("direction", "bidirectional")),
                        )
                        results.append(edge)
        else:
            # Fallback implementation
            for key, edge in self._edges.items():
                if node_id in key:
                    if edge.strength >= min_strength:
                        if association_type is None or edge.association_type == association_type:
                            results.append(edge)

        # Sort by strength and limit
        results.sort(key=lambda e: e.strength, reverse=True)
        return results[:limit]

    # ======================================================================
    # Hebbian Learning Integration
    # ======================================================================

    def reinforce(
        self,
        node_a: Union[str, MemoryNode],
        node_b: Union[str, MemoryNode],
        association_type: AssociationType = AssociationType.CO_OCCURRENCE,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AssociationEdge]:
        """
        Reinforce association between two nodes (Hebbian learning).

        This is the main entry point for co-retrieval strengthening.
        Call this when two memories are retrieved together.

        Args:
            node_a: First node (ID or MemoryNode)
            node_b: Second node (ID or MemoryNode)
            association_type: Type of association
            context: Optional context information

        Returns:
            The updated AssociationEdge, or None if nodes not found
        """
        return self.strengthener.reinforce(
            node_a, node_b, self,
            association_type=association_type,
            context=context or {"co_retrieval": True},
        )

    def reinforce_batch(
        self,
        node_ids: List[Union[str, MemoryNode]],
        association_type: AssociationType = AssociationType.CO_OCCURRENCE,
    ) -> List[AssociationEdge]:
        """
        Reinforce associations between all pairs in a batch.

        Useful for co-retrieval of multiple memories.

        Args:
            node_ids: List of node IDs or MemoryNodes
            association_type: Type of association

        Returns:
            List of created/updated AssociationEdges
        """
        results = []
        n = len(node_ids)

        for i in range(n):
            for j in range(i + 1, n):
                edge = self.reinforce(node_ids[i], node_ids[j], association_type)
                if edge:
                    results.append(edge)

        return results

    # ======================================================================
    # Graph Queries and Analytics
    # ======================================================================

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        min_strength: float = 0.1,
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID
            min_strength: Minimum edge strength to consider

        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        if not NETWORKX_AVAILABLE:
            return None

        if not self.has_node(source_id) or not self.has_node(target_id):
            return None

        # Create subgraph with only strong edges
        subgraph = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if data.get("strength", 0.0) >= min_strength:
                subgraph.add_edge(u, v, weight=1.0 / data.get("strength", 0.1))

        try:
            path = nx.shortest_path(subgraph, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_associations_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        min_strength: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Find association paths between nodes, ranked by strength.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID
            max_hops: Maximum path length
            min_strength: Minimum edge strength

        Returns:
            List of paths with metadata, sorted by total strength
        """
        paths = []

        if not NETWORKX_AVAILABLE:
            return paths

        if not self.has_node(source_id) or not self.has_node(target_id):
            return paths

        # Use Yen's algorithm for k-shortest paths or BFS for simple paths
        try:
            all_paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_hops,
            ))
        except (nx.NetworkXError, nx.NodeNotFound):
            return paths

        for path in all_paths:
            # Calculate path strength (product of edge strengths)
            path_strength = 1.0
            valid = True

            for i in range(len(path) - 1):
                edge = self.get_edge(path[i], path[i + 1])
                if edge and edge.strength >= min_strength:
                    path_strength *= edge.strength
                else:
                    valid = False
                    break

            if valid:
                paths.append({
                    "path": path,
                    "strength": path_strength,
                    "length": len(path) - 1,
                })

        # Sort by strength descending
        paths.sort(key=lambda p: p["strength"], reverse=True)
        return paths

    def find_clusters(
        self,
        min_cluster_size: int = 3,
        min_strength: float = 0.2,
    ) -> List[Set[str]]:
        """
        Find clusters of strongly associated memories.

        Args:
            min_cluster_size: Minimum nodes per cluster
            min_strength: Minimum edge strength for clustering

        Returns:
            List of sets of node IDs, each representing a cluster
        """
        if not NETWORKX_AVAILABLE:
            return []

        # Create undirected graph with strong edges
        undirected = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if data.get("strength", 0.0) >= min_strength:
                undirected.add_edge(u, v)

        # Find connected components
        clusters = []
        for component in nx.connected_components(undirected):
            if len(component) >= min_cluster_size:
                clusters.append(component)

        return clusters

    def find_communities(self, resolution: float = 1.0) -> List[Set[str]]:
        """
        Find communities using Louvain algorithm.

        Args:
            resolution: Resolution parameter for community detection

        Returns:
            List of sets of node IDs, each representing a community
        """
        if not NETWORKX_AVAILABLE:
            return []

        try:
            import community as community_louvain
        except ImportError:
            logger.warning("python-louvain not available, using connected components")
            return self.find_clusters()

        # Convert to undirected graph
        undirected = self.graph.to_undirected()

        # Detect communities
        partition = community_louvain.best_partition(undirected, resolution=resolution)

        # Group by community ID
        communities = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)

        return list(communities.values())

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        min_strength: float = 0.0,
    ) -> Set[str]:
        """
        Get neighboring nodes within N hops.

        Args:
            node_id: Starting node ID
            hops: Number of hops to explore
            min_strength: Minimum edge strength

        Returns:
            Set of neighboring node IDs
        """
        if not NETWORKX_AVAILABLE:
            return set()

        if not self.has_node(node_id):
            return set()

        neighbors = set()
        current_level = {node_id}

        for _ in range(hops):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    edge = self.get_edge(node, neighbor)
                    if edge and edge.strength >= min_strength:
                        if neighbor not in neighbors and neighbor != node_id:
                            next_level.add(neighbor)
            neighbors.update(next_level)
            current_level = next_level

        return neighbors

    # ======================================================================
    # Graph Metrics
    # ======================================================================

    def compute_metrics(self, force: bool = False) -> GraphMetrics:
        """
        Compute graph structural metrics.

        Args:
            force: Force recomputation even if cached

        Returns:
            GraphMetrics object with computed metrics
        """
        if not self._metrics_dirty and not force and self._metrics_cache:
            return self._metrics_cache

        if not NETWORKX_AVAILABLE:
            return GraphMetrics(
                node_count=len(self._nodes),
                edge_count=len(self._edges),
            )

        metrics = GraphMetrics()

        # Basic counts
        metrics.node_count = self.graph.number_of_nodes()
        metrics.edge_count = self.graph.number_of_edges()

        if metrics.node_count == 0:
            return metrics

        # Average degree
        degrees = [d for n, d in self.graph.degree()]
        metrics.avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

        # Density
        max_edges = metrics.node_count * (metrics.node_count - 1)
        metrics.density = metrics.edge_count / max_edges if max_edges > 0 else 0.0

        # Clustering coefficient (on undirected version)
        try:
            undirected = self.graph.to_undirected()
            metrics.avg_clustering = nx.average_clustering(undirected)
        except Exception:
            metrics.avg_clustering = 0.0

        # Connected components
        undirected = self.graph.to_undirected()
        metrics.connected_components = nx.number_connected_components(undirected)

        # Largest component
        components = list(nx.connected_components(undirected))
        if components:
            metrics.largest_component_size = max(len(c) for c in components)

        # Average path length and diameter (sampled for large graphs)
        if metrics.node_count > 0 and metrics.node_count <= 500:
            try:
                if nx.is_connected(undirected):
                    metrics.avg_path_length = nx.average_shortest_path_length(undirected)
                    metrics.diameter = nx.diameter(undirected)
            except Exception:
                pass

        self._metrics_cache = metrics
        self._metrics_dirty = False

        return metrics

    # ======================================================================
    # Persistence
    # ======================================================================

    def save(self, path: Optional[str] = None) -> bool:
        """
        Save the association network to disk.

        Args:
            path: Optional path to save to (default: config.persist_path)

        Returns:
            True if save succeeded
        """
        save_path = path or self.config.persist_path
        if not save_path:
            return False

        try:
            data = {
                "version": "1.0",
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "max_edges_per_node": self.config.max_edges_per_node,
                    "min_edge_threshold": self.config.min_edge_threshold,
                },
                "nodes": [],
                "edges": [],
            }

            if NETWORKX_AVAILABLE:
                # Export nodes
                for node_id, node_data in self.graph.nodes(data=True):
                    data["nodes"].append({
                        "id": node_id,
                        **node_data,
                    })

                # Export edges
                for u, v, edge_data in self.graph.edges(data=True):
                    if isinstance(edge_data, list):  # MultiDiGraph
                        for d in edge_data.values():
                            data["edges"].append({
                                "source_id": u,
                                "target_id": v,
                                **d,
                            })
                    else:
                        data["edges"].append({
                            "source_id": u,
                            "target_id": v,
                            **edge_data,
                        })
            else:
                # Fallback export
                for node_id, node in self._nodes.items():
                    data["nodes"].append({
                        "id": node_id,
                        "content": node.content[:200],
                        "created_at": node.created_at.isoformat(),
                    })
                for edge in self._edges.values():
                    data["edges"].append(edge.to_dict())

            # Write to file
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved association network to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save association network: {e}")
            return False

    def _load_from_disk(self) -> bool:
        """Load association network from disk."""
        if not self.config.persist_path:
            return False

        path = Path(self.config.persist_path)
        if not path.exists():
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if NETWORKX_AVAILABLE:
                # Load nodes
                for node_data in data.get("nodes", []):
                    node_id = node_data.pop("id", None)
                    if node_id:
                        self.graph.add_node(node_id, **node_data)

                # Load edges
                for edge_data in data.get("edges", []):
                    source = edge_data.pop("source_id", None)
                    target = edge_data.pop("target_id", None)
                    if source and target:
                        self.graph.add_edge(source, target, **edge_data)
            else:
                # Fallback load
                for node_data in data.get("nodes", []):
                    self._nodes[node_data["id"]] = MemoryNode(
                        id=node_data["id"],
                        hdv=BinaryHDV.zeros(16384),  # Placeholder
                        content=node_data.get("content", ""),
                    )
                for edge_data in data.get("edges", []):
                    edge = AssociationEdge.from_dict(edge_data)
                    self._edges[edge.key()] = edge

            logger.info(f"Loaded association network from {path}")
            self._metrics_dirty = True
            return True

        except Exception as e:
            logger.error(f"Failed to load association network: {e}")
            return False

    def export_graphml(self, path: str) -> bool:
        """
        Export graph in GraphML format for external tools.

        Args:
            path: Path to save GraphML file

        Returns:
            True if export succeeded
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available for GraphML export")
            return False

        try:
            nx.write_graphml(self.graph, path)
            logger.info(f"Exported graph to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export GraphML: {e}")
            return False

    def export_gexf(self, path: str) -> bool:
        """
        Export graph in GEXF format for Gephi.

        Args:
            path: Path to save GEXF file

        Returns:
            True if export succeeded
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available for GEXF export")
            return False

        try:
            nx.write_gexf(self.graph, path)
            logger.info(f"Exported graph to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export GEXF: {e}")
            return False

    # ======================================================================
    # Visualization
    # ======================================================================

    def visualize(
        self,
        output_path: Optional[str] = None,
        layout: str = "spring",
        max_nodes: Optional[int] = None,
        min_strength: float = 0.1,
        node_colors: Optional[Dict[str, str]] = None,
        node_sizes: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """
        Create an interactive visualization of the association network.

        Args:
            output_path: Optional path to save HTML file
            layout: Layout algorithm (spring, circular, kamada_kawai, spectral)
            max_nodes: Maximum nodes to visualize (for performance)
            min_strength: Minimum edge strength to include
            node_colors: Optional mapping of node IDs to colors
            node_sizes: Optional mapping of node IDs to sizes

        Returns:
            HTML string if Plotly available, None otherwise
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available for visualization")
            return None

        max_nodes = max_nodes or self.config.max_nodes_for_viz

        # Filter to strong edges and limit nodes
        subgraph = self._create_visualization_subgraph(max_nodes, min_strength)

        if subgraph.number_of_nodes() == 0:
            logger.warning("No nodes to visualize after filtering")
            return None

        # Compute layout
        pos = self._compute_layout(subgraph, layout)

        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []

        for u, v, data in subgraph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            strength = data.get("strength", 0.0)
            edge_info.append(f"{u} -> {v}: {strength:.2f}")

        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []

        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_data = subgraph.nodes[node]
            node_text.append(node[:8] if len(node) > 8 else node)
            node_info.append(
                f"ID: {node}<br>"
                f"Content: {node_data.get('content', 'N/A')[:50]}...<br>"
                f"Strength: {node_data.get('ltp_strength', 0):.2f}"
            )

        # Create figure
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                name='Associations'
            ))

            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='lightblue',
                    line=dict(width=1, color='darkblue')
                ),
                text=node_text,
                textposition='bottom center',
                hovertext=node_info,
                hoverinfo='text',
                name='Memories'
            ))

            fig.update_layout(
                title=f"Association Network ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )

            html = fig.to_html(include_plotlyjs='cdn')

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.info(f"Saved visualization to {output_path}")

            return html

        return None

    def _create_visualization_subgraph(
        self,
        max_nodes: int,
        min_strength: float,
    ) -> nx.Graph:
        """Create a filtered subgraph for visualization."""
        if not NETWORKX_AVAILABLE:
            return nx.Graph()

        # Filter edges by strength
        edges = [
            (u, v, d)
            for u, v, d in self.graph.edges(data=True)
            if d.get("strength", 0.0) >= min_strength
        ]

        # Create subgraph
        subgraph = nx.Graph()
        subgraph.add_edges_from(edges)

        # If too many nodes, keep highest degree nodes
        if subgraph.number_of_nodes() > max_nodes:
            degrees = dict(subgraph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = subgraph.subgraph([n for n, _ in top_nodes]).copy()

        return subgraph

    def _compute_layout(
        self,
        graph: nx.Graph,
        layout: str,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute node positions for visualization."""
        if not NETWORKX_AVAILABLE:
            return {}

        try:
            if layout == "spring":
                return nx.spring_layout(graph, k=1, iterations=50)
            elif layout == "circular":
                return nx.circular_layout(graph)
            elif layout == "kamada_kawai":
                return nx.kamada_kawai_layout(graph)
            elif layout == "spectral":
                return nx.spectral_layout(graph)
            else:
                return nx.spring_layout(graph)
        except Exception:
            return nx.spring_layout(graph)

    def create_cluster_visualization(
        self,
        output_path: Optional[str] = None,
        min_strength: float = 0.2,
    ) -> Optional[str]:
        """
        Create a visualization focused on clusters.

        Args:
            output_path: Optional path to save HTML file
            min_strength: Minimum edge strength for clustering

        Returns:
            HTML string if Plotly available
        """
        if not NETWORKX_AVAILABLE or not PLOTLY_AVAILABLE:
            return None

        clusters = self.find_clusters(min_strength=min_strength)
        if not clusters:
            return None

        # Create a subplot for each cluster
        n_clusters = min(len(clusters), 6)  # Limit to 6 subplots
        cols = 2
        rows = (n_clusters + 1) // 2

        subplot_titles = [f"Cluster {i+1} ({len(c)} nodes)" for i, c in enumerate(clusters[:n_clusters])]

        if make_subplots:
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=subplot_titles,
                specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
            )
        else:
            return None

        colors = px.colors.qualitative.Set2

        for idx, cluster in enumerate(clusters[:n_clusters]):
            row = idx // cols + 1
            col = idx % cols + 1

            # Get subgraph for this cluster
            subgraph = self.graph.subgraph(cluster).to_undirected()
            pos = nx.spring_layout(subgraph)

            # Edges
            edge_x, edge_y = [], []
            for u, v in subgraph.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color=colors[idx % len(colors)]),
                    hoverinfo='none',
                    showlegend=False,
                ),
                row=row, col=col,
            )

            # Nodes
            node_x, node_y, node_text = [], [], []
            for node in cluster:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node[:6])

            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    marker=dict(size=8, color=colors[idx % len(colors)]),
                    text=node_text,
                    textposition='bottom center',
                    hoverinfo='text',
                    showlegend=False,
                ),
                row=row, col=col,
            )

        fig.update_layout(
            title=f"Memory Clusters ({len(clusters)} total)",
            height=300 * rows,
            showlegend=False,
        )

        html = fig.to_html(include_plotlyjs='cdn')

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Saved cluster visualization to {output_path}")

        return html

    # ======================================================================
    # GraphQL Export Preparation
    # ======================================================================

    def to_graphql_schema(self) -> str:
        """
        Generate GraphQL schema for the association network.

        Returns:
            GraphQL schema string
        """
        return '''
type MemoryNode {
  id: ID!
  content: String!
  createdAt: DateTime!
  tier: String!
  ltpStrength: Float!
  accessCount: Int!
  associations(first: Int, minStrength: Float): [AssociationEdge!]!
  clusters: [[MemoryNode!]!]
}

type AssociationEdge {
  sourceId: ID!
  targetId: ID!
  strength: Float!
  associationType: AssociationType!
  createdAt: DateTime!
  lastStrengthened: DateTime!
  fireCount: Int!
  source: MemoryNode!
  target: MemoryNode!
}

enum AssociationType {
  SEMANTIC
  TEMPORAL
  CAUSAL
  CONTEXTUAL
  HIERARCHICAL
  CO_OCCURRENCE
}

type GraphMetrics {
  nodeCount: Int!
  edgeCount: Int!
  avgDegree: Float!
  density: Float!
  avgClustering: Float!
  connectedComponents: Int!
  largestComponentSize: Int!
  avgPathLength: Float!
  diameter: Int!
}

type Cluster {
  nodes: [MemoryNode!]!
  size: Int!
  cohesion: Float!
}

type AssociationsQuery {
  node(id: ID!): MemoryNode
  nodes(first: Int): [MemoryNode!]!
  associations(nodeId: ID!, minStrength: Float): [AssociationEdge!]!
  path(from: ID!, to: ID!, maxHops: Int): [MemoryNode!]!
  clusters(minSize: Int): [Cluster!]!
  metrics: GraphMetrics!
  search(query: String!, first: Int): [MemoryNode!]!
}

type AssociationsMutation {
  reinforce(nodeA: ID!, nodeB: ID!, type: AssociationType): AssociationEdge!
  removeAssociation(nodeA: ID!, nodeB: ID!): Boolean
}

schema {
  query: AssociationsQuery
  mutation: AssociationsMutation
}
'''

    def to_graphql_data(self) -> Dict[str, Any]:
        """
        Export data in GraphQL-compatible format.

        Returns:
            Dictionary with nodes and edges
        """
        nodes = []
        edges = []

        if NETWORKX_AVAILABLE:
            for node_id, data in self.graph.nodes(data=True):
                nodes.append({
                    "id": node_id,
                    "content": data.get("content", ""),
                    "createdAt": data.get("created_at"),
                    "tier": data.get("tier", "unknown"),
                    "ltpStrength": data.get("ltp_strength", 0.0),
                    "accessCount": data.get("access_count", 0),
                })

            for u, v, data in self.graph.edges(data=True):
                edges.append({
                    "sourceId": u,
                    "targetId": v,
                    "strength": data.get("strength", 0.0),
                    "associationType": data.get("association_type", "co_occurrence"),
                    "createdAt": data.get("created_at"),
                    "lastStrengthened": data.get("last_strengthened", data.get("created_at")),
                    "fireCount": data.get("fire_count", 0),
                })
        else:
            for node_id, node in self._nodes.items():
                nodes.append({
                    "id": node_id,
                    "content": node.content[:200],
                    "createdAt": node.created_at.isoformat(),
                    "tier": node.tier,
                    "ltpStrength": node.ltp_strength,
                    "accessCount": node.access_count,
                })

            for edge in self._edges.values():
                edges.append({
                    "sourceId": edge.source_id,
                    "targetId": edge.target_id,
                    "strength": edge.strength,
                    "associationType": edge.association_type.value,
                    "createdAt": edge.created_at.isoformat(),
                    "lastStrengthened": edge.last_strengthened.isoformat(),
                    "fireCount": edge.fire_count,
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": self.compute_metrics().to_dict(),
        }

    def create_graphql_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> Optional[str]:
        """
        Create a GraphQL server endpoint for the association network.

        Note: This requires the 'strawberry-graphql' package.

        Args:
            host: Server host
            port: Server port

        Returns:
            URL of the GraphQL endpoint, or None if server creation failed
        """
        try:
            import strawberry
            from strawberry.asgi import GraphQL
        except ImportError:
            logger.warning(
                "strawberry-graphql not available. "
                "Install with: pip install strawberry-graphql"
            )
            return None

        @strawberry.type
        class MemoryNodeType:
            id: strawberry.ID
            content: str
            created_at: str
            tier: str
            ltp_strength: float
            access_count: int

        @strawberry.type
        class AssociationEdgeType:
            source_id: strawberry.ID
            target_id: strawberry.ID
            strength: float
            association_type: str
            created_at: str
            last_strengthened: str
            fire_count: int

        @strawberry.type
        class GraphMetricsType:
            node_count: int
            edge_count: int
            avg_degree: float
            density: float
            avg_clustering: float
            connected_components: int
            largest_component_size: int

        @strawberry.type
        class Query:
            @strawberry.field
            def node(self, id: strawberry.ID) -> Optional[MemoryNodeType]:
                data = self.get_node(str(id))
                if data:
                    return MemoryNodeType(
                        id=data.get("id", ""),
                        content=data.get("content", ""),
                        created_at=data.get("created_at", ""),
                        tier=data.get("tier", ""),
                        ltp_strength=data.get("ltp_strength", 0.0),
                        access_count=data.get("access_count", 0),
                    )
                return None

            @strawberry.field
            def nodes(self, first: int = 100) -> List[MemoryNodeType]:
                result = []
                for node_id in self.get_all_nodes()[:first]:
                    data = self.get_node(node_id)
                    if data:
                        result.append(MemoryNodeType(
                            id=data.get("id", ""),
                            content=data.get("content", ""),
                            created_at=data.get("created_at", ""),
                            tier=data.get("tier", ""),
                            ltp_strength=data.get("ltp_strength", 0.0),
                            access_count=data.get("access_count", 0),
                        ))
                return result

            @strawberry.field
            def metrics(self) -> GraphMetricsType:
                m = self.compute_metrics()
                return GraphMetricsType(
                    node_count=m.node_count,
                    edge_count=m.edge_count,
                    avg_degree=m.avg_degree,
                    density=m.density,
                    avg_clustering=m.avg_clustering,
                    connected_components=m.connected_components,
                    largest_component_size=m.largest_component_size,
                )

        @strawberry.type
        class Mutation:
            @strawberry.mutation
            def reinforce(
                self,
                node_a: strawberry.ID,
                node_b: strawberry.ID,
                association_type: str = "co_occurrence"
            ) -> Optional[AssociationEdgeType]:
                edge = self.reinforce(
                    str(node_a),
                    str(node_b),
                    AssociationType(association_type)
                )
                if edge:
                    return AssociationEdgeType(
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        strength=edge.strength,
                        association_type=edge.association_type.value,
                        created_at=edge.created_at.isoformat(),
                        last_strengthened=edge.last_strengthened.isoformat(),
                        fire_count=edge.fire_count,
                    )
                return None

        schema = strawberry.Schema(query=Query, mutation=Mutation)
        return f"graphql://{host}:{port}"

    # ======================================================================
    # Utility Methods
    # ======================================================================

    def apply_decay(self, half_life_days: Optional[float] = None) -> int:
        """
        Apply time-based decay to all associations.

        Args:
            half_life_days: Half-life for decay (default: from config)

        Returns:
            Number of associations decayed
        """
        if not self.config.decay_enabled:
            return 0

        half_life = half_life_days or self.config.half_life_days
        decayed = 0

        if NETWORKX_AVAILABLE:
            for u, v, data in self.graph.edges(data=True):
                strength = data.get("strength", 0.0)
                created = datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat()))
                last = datetime.fromisoformat(data.get("last_strengthened", created.isoformat()))

                age_seconds = (datetime.now(timezone.utc) - last).total_seconds()
                age_days = age_seconds / 86400.0

                decay_factor = math.exp(-(math.log(2) / half_life) * age_days)
                new_strength = max(self.config.min_strength, strength * decay_factor)

                if new_strength < strength:
                    self.graph[u][v]["strength"] = new_strength
                    self.graph[u][v]["last_strengthened"] = datetime.now(timezone.utc).isoformat()
                    decayed += 1
        else:
            for edge in self._edges.values():
                old_strength = edge.strength
                edge.decay(half_life)
                if edge.strength < old_strength:
                    decayed += 1

        return decayed

    def prune_weak_associations(self, min_strength: Optional[float] = None) -> int:
        """
        Remove associations below the strength threshold.

        Args:
            min_strength: Minimum strength to keep (default: from config)

        Returns:
            Number of associations removed
        """
        threshold = min_strength or self.config.min_edge_threshold
        removed = 0

        if NETWORKX_AVAILABLE:
            edges_to_remove = []
            for u, v, data in self.graph.edges(data=True):
                if data.get("strength", 0.0) < threshold:
                    edges_to_remove.append((u, v))

            for u, v in edges_to_remove:
                self.graph.remove_edge(u, v)
                removed += 1
        else:
            keys_to_remove = [
                k for k, e in self._edges.items()
                if e.strength < threshold
            ]
            for key in keys_to_remove:
                del self._edges[key]
                removed += 1

        if removed > 0:
            self._metrics_dirty = True

        return removed

    def __len__(self) -> int:
        """Return the number of nodes in the network."""
        if NETWORKX_AVAILABLE:
            return self.graph.number_of_nodes()
        return len(self._nodes)

    def __repr__(self) -> str:
        metrics = self.compute_metrics()
        return (
            f"AssociationsNetwork("
            f"nodes={metrics.node_count}, "
            f"edges={metrics.edge_count}, "
            f"density={metrics.density:.3f}"
            f")"
        )


# ======================================================================
# Integration with HAIMEngine
# ======================================================================

class AssociationRecallIntegrator:
    """
    Integrates association tracking with memory recall.

    Hooks into the recall process to automatically strengthen
    associations between co-retrieved memories.
    """

    def __init__(
        self,
        network: AssociationsNetwork,
        auto_strengthen: bool = True,
        strengthen_threshold: int = 2,  # Min items to trigger strengthening
    ):
        """
        Initialize the integrator.

        Args:
            network: The association network to modify
            auto_strengthen: Whether to automatically strengthen on co-retrieval
            strengthen_threshold: Min items in recall to trigger strengthening
        """
        self.network = network
        self.auto_strengthen = auto_strengthen
        self.strengthen_threshold = strengthen_threshold

    def on_recall(
        self,
        recalled_items: List[Union[MemoryNode, str]],
        query: Optional[str] = None,
    ) -> None:
        """
        Called after a recall operation.

        Strengthens associations between all co-retrieved items.

        Args:
            recalled_items: List of recalled nodes or node IDs
            query: Optional query string (for context)
        """
        if not self.auto_strengthen:
            return

        if len(recalled_items) < self.strengthen_threshold:
            return

        # Strengthen all pairs
        self.network.reinforce_batch(
            recalled_items,
            association_type=AssociationType.CO_OCCURRENCE,
        )

    def on_store(
        self,
        node: MemoryNode,
        context_nodes: Optional[List[Union[MemoryNode, str]]] = None,
    ) -> None:
        """
        Called after a memory is stored.

        Adds the node to the network and creates initial associations.

        Args:
            node: The newly stored memory node
            context_nodes: Optional list of related nodes for initial associations
        """
        # Add to network
        self.network.add_node(node)

        # Create associations with context nodes
        if context_nodes:
            for ctx_node in context_nodes:
                ctx_id = ctx_node if isinstance(ctx_node, str) else ctx_node.id
                self.network.add_association(
                    ctx_id,
                    node.id,
                    strength=0.1,
                    association_type=AssociationType.CONTEXTUAL,
                )


# ======================================================================
# Factory Functions
# ======================================================================

def create_associations_network(
    config: Optional[AssociationConfig] = None,
    storage_dir: Optional[str] = None,
) -> AssociationsNetwork:
    """
    Factory function to create an association network.

    Args:
        config: Optional configuration
        storage_dir: Directory for persistence

    Returns:
        Configured AssociationsNetwork instance
    """
    return AssociationsNetwork(
        config=config,
        storage_dir=storage_dir,
    )


def create_network_from_nodes(
    nodes: List[MemoryNode],
    config: Optional[AssociationConfig] = None,
) -> AssociationsNetwork:
    """
    Create an association network from a list of memory nodes.

    Args:
        nodes: List of MemoryNode objects
        config: Optional configuration

    Returns:
        Populated AssociationsNetwork instance
    """
    network = AssociationsNetwork(config=config)

    for node in nodes:
        network.add_node(node)

    return network


# ======================================================================
# Convenience Functions
# ======================================================================

def reinforce_associations(
    network: AssociationsNetwork,
    node_a: Union[str, MemoryNode],
    node_b: Union[str, MemoryNode],
    amount: float = 0.1,
) -> Optional[AssociationEdge]:
    """
    Convenience function to reinforce an association.

    Args:
        network: The association network
        node_a: First node
        node_b: Second node
        amount: Strengthening amount

    Returns:
        Updated edge or None
    """
    return network.reinforce(node_a, node_b)


def find_related_memories(
    network: AssociationsNetwork,
    node_id: str,
    max_results: int = 10,
    min_strength: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Find memories associated with a given node.

    Args:
        network: The association network
        node_id: Starting node ID
        max_results: Maximum results to return
        min_strength: Minimum association strength

    Returns:
        List of related memory info dicts
    """
    associations = network.get_associations(
        node_id,
        min_strength=min_strength,
        limit=max_results,
    )

    results = []
    for edge in associations:
        other_id = edge.target_id if edge.source_id == node_id else edge.source_id
        node_data = network.get_node(other_id)
        if node_data:
            results.append({
                "node_id": other_id,
                "strength": edge.strength,
                "association_type": edge.association_type.value,
                "content": node_data.get("content", ""),
            })

    return results


__all__ = [
    "AssociationType",
    "AssociationDirection",
    "AssociationEdge",
    "AssociationConfig",
    "GraphMetrics",
    "AssociationStrengthener",
    "AssociationsNetwork",
    "AssociationRecallIntegrator",
    "create_associations_network",
    "create_network_from_nodes",
    "reinforce_associations",
    "find_related_memories",
]
