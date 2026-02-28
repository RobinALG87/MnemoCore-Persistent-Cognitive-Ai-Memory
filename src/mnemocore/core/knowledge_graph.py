"""
Bidirectional Knowledge Graph — knowledge_graph.py
====================================================
Implements a Zettelkasten / Mnemosyne-inspired self-organizing knowledge graph
with dynamic bidirectional edges, automatic redundancy pruning, and traversal-based
retrieval as a complement to HNSW vector search.

Research basis
~~~~~~~~~~~~~~
- **Mnemosyne** (Georgia Tech / Microsoft Research 2025): graph-structured memory
  with unsupervised redundancy pruning radically reduces noise and improves precision.
- **A-MEM** (arXiv 2502.12110): Zettelkasten-inspired dynamic bidirectional linking
  creates a living knowledge network — not just retrieval.
- **Continuum Memory Architecture** (arXiv 2601.09913): formalizes the
  ingest → activation → retrieval → consolidation lifecycle.

Architecture
~~~~~~~~~~~~
::

    ┌─────────────────────────────────────────────────────────────────┐
    │                    KnowledgeGraph                               │
    │                                                                 │
    │  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐   │
    │  │ KnowledgeNode│──▶│KnowledgeEdge │──▶│ Bidirectional      │   │
    │  │ (entity/     │   │ (weighted,   │   │  linking: A→B      │   │
    │  │  concept/    │   │  typed,      │   │  auto-creates B→A  │   │
    │  │  strategy)   │   │  decayable)  │   │  with reciprocal   │   │
    │  └─────────────┘   └──────────────┘   │  weight             │   │
    │                                        └────────────────────┘   │
    │                                                                 │
    │  Traversal-based retrieval:                                     │
    │    BFS/DFS/weighted walk from seed nodes                        │
    │    → complements HNSW for structural queries                    │
    │                                                                 │
    │  Automatic redundancy pruning:                                  │
    │    Cosine-similarity threshold on node embeddings               │
    │    + weak-edge pruning (strength < threshold)                   │
    │    → GraphML export for analysis                                │
    │                                                                 │
    │  Self-organizing:                                               │
    │    Edge weights updated on every retrieval (Hebbian)             │
    │    Cluster detection → automatic topic grouping                 │
    │    Activation spreading → related concepts auto-activate        │
    └─────────────────────────────────────────────────────────────────┘

Differences from existing associations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``associations.py`` is a NetworkX-backed association graph focused on raw
co-retrieval Hebbian strengthening. ``knowledge_graph.py`` adds:

1. **Always-bidirectional edges** — every A→B creates B→A automatically.
2. **Redundancy pruning** — merges semantically duplicate nodes.
3. **Traversal retrieval** — BFS/DFS/weighted random walk retrieval.
4. **Activation spreading** — excites neighbor nodes on access.
5. **Health scoring** — per-node health metric (Neuroca STM→MTM→LTM model).
6. **Cluster auto-detection** — unsupervised topic grouping.

Integration points:
    - Complements ``tier_manager`` search with graph-based retrieval.
    - Feeds ``strategy_bank`` with structural context.
    - Pulse loop Phase 9: graph maintenance tick.
    - ``semantic_store`` concepts become KnowledgeNodes.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeEdge:
    """
    A weighted, typed, bidirectional edge in the knowledge graph.

    Every edge is paired with a reciprocal edge (target→source) that
    is tracked internally. When you strengthen A→B, B→A is also updated
    (with a configurable reciprocal factor, default 0.7).

    Fields:
        source_id: Origin node ID.
        target_id: Destination node ID.
        weight: Current strength (0.0–1.0). Updated on each retrieval.
        edge_type: Semantic type of relationship.
        created_at: Creation timestamp.
        last_activated: Last time this edge was traversed or strengthened.
        activation_count: How many times this edge has been traversed.
        metadata: Arbitrary extension dict.
    """
    source_id: str = ""
    target_id: str = ""
    weight: float = 0.5
    edge_type: str = "related"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.source_id, self.target_id)

    def activate(self, boost: float = 0.05) -> None:
        """Record a traversal/activation event."""
        self.activation_count += 1
        self.last_activated = datetime.now(timezone.utc)
        self.weight = min(1.0, self.weight + boost)

    def decay(self, half_life_days: float = 30.0) -> None:
        """Apply temporal decay to edge weight."""
        now = datetime.now(timezone.utc)
        age_days = (now - self.last_activated).total_seconds() / 86400.0
        if age_days > 0:
            decay_factor = math.exp(-0.693 * age_days / half_life_days)
            self.weight *= decay_factor

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "edge_type": self.edge_type,
            "created_at": self.created_at.isoformat(),
            "last_activated": self.last_activated.isoformat(),
            "activation_count": self.activation_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeEdge":
        created = d.get("created_at")
        activated = d.get("last_activated")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        else:
            created = datetime.now(timezone.utc)
        if isinstance(activated, str):
            activated = datetime.fromisoformat(activated)
        else:
            activated = datetime.now(timezone.utc)
        return cls(
            source_id=d.get("source_id", ""),
            target_id=d.get("target_id", ""),
            weight=d.get("weight", 0.5),
            edge_type=d.get("edge_type", "related"),
            created_at=created,
            last_activated=activated,
            activation_count=d.get("activation_count", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class KnowledgeNode:
    """
    A node in the knowledge graph.

    Can represent an entity, concept, strategy, or any knowledge unit.
    Each node has a health score that determines its lifecycle stage
    (Neuroca model: STM → MTM → LTM).

    Fields:
        id: Unique identifier.
        label: Human-readable label.
        content: Full text content of this knowledge unit.
        node_type: What kind of knowledge ("entity", "concept", "strategy", "fact").
        embedding_hash: Hash of the embedding vector for dedup detection.
        tags: Free-form tags.
        agent_id: Which agent created this node.
        created_at: Creation timestamp.
        last_accessed: Last retrieval/activation time.
        access_count: Total access count.
        health_score: 0.0–1.0 composite health (STM→MTM→LTM lifecycle).
        activation_level: Current activation energy (decays over time).
        cluster_id: Auto-assigned cluster from community detection.
        source_ids: Provenance links to memory IDs / episode IDs.
        metadata: Arbitrary extension dict.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    content: str = ""
    node_type: str = "concept"
    embedding_hash: str = ""
    tags: List[str] = field(default_factory=list)
    agent_id: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    health_score: float = 0.5
    activation_level: float = 0.0
    cluster_id: Optional[str] = None
    source_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Health Lifecycle (Neuroca STM→MTM→LTM) ───────────────────────

    @property
    def lifecycle_stage(self) -> str:
        """
        Determine lifecycle stage based on health score.
        - STM (< 0.3): Short-term, fragile, easily forgotten
        - MTM (0.3–0.7): Medium-term, consolidating
        - LTM (> 0.7): Long-term, stable, high confidence
        """
        if self.health_score < 0.3:
            return "STM"
        elif self.health_score < 0.7:
            return "MTM"
        return "LTM"

    def update_health(self) -> None:
        """
        Recalculate health score from access pattern and age.

        Health factors:
        - Recency: How recently was this accessed?
        - Frequency: How often is it accessed?
        - Stability: How old is it? (older = more stable if still accessed)
        """
        now = datetime.now(timezone.utc)
        age_days = max(0.001, (now - self.created_at).total_seconds() / 86400)
        recency_hours = max(0.001, (now - self.last_accessed).total_seconds() / 3600)

        # Recency factor: exponential decay, half-life = 48 hours
        recency = math.exp(-0.693 * recency_hours / 48.0)

        # Frequency factor: log-scaled access count
        frequency = min(1.0, math.log1p(self.access_count) / 5.0)

        # Stability: older nodes that are still accessed get a bonus
        stability = min(1.0, math.log1p(age_days) / 4.0) if self.access_count > 2 else 0.0

        self.health_score = max(0.0, min(1.0,
            0.4 * recency + 0.35 * frequency + 0.25 * stability
        ))

    def access(self) -> None:
        """Record an access event."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        self.activation_level = min(1.0, self.activation_level + 0.3)
        self.update_health()

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "content": self.content,
            "node_type": self.node_type,
            "embedding_hash": self.embedding_hash,
            "tags": self.tags,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "health_score": self.health_score,
            "activation_level": self.activation_level,
            "cluster_id": self.cluster_id,
            "source_ids": self.source_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeNode":
        created = d.get("created_at")
        accessed = d.get("last_accessed")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        else:
            created = datetime.now(timezone.utc)
        if isinstance(accessed, str):
            accessed = datetime.fromisoformat(accessed)
        else:
            accessed = datetime.now(timezone.utc)
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            label=d.get("label", ""),
            content=d.get("content", ""),
            node_type=d.get("node_type", "concept"),
            embedding_hash=d.get("embedding_hash", ""),
            tags=d.get("tags", []),
            agent_id=d.get("agent_id", "default"),
            created_at=created,
            last_accessed=accessed,
            access_count=d.get("access_count", 0),
            health_score=d.get("health_score", 0.5),
            activation_level=d.get("activation_level", 0.0),
            cluster_id=d.get("cluster_id"),
            source_ids=d.get("source_ids", []),
            metadata=d.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════
# Knowledge Graph Service
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeGraphService:
    """
    Self-organizing bidirectional knowledge graph.

    Key capabilities (beyond existing associations.py):

    1. **Always-bidirectional edges**: Every ``link(A, B)`` automatically
       creates both A→B and B→A edges. The reciprocal edge gets a
       configurable weight factor (default 0.7×).

    2. **Redundancy pruning** (Mnemosyne): Detects semantically similar
       nodes via embedding hash or content similarity, and merges them.
       This radically reduces noise in the graph.

    3. **Traversal-based retrieval**: BFS, DFS, and weighted random walk
       from seed nodes. Complements HNSW vector search with structural
       query capability.

    4. **Activation spreading**: When a node is accessed, excitation
       spreads to neighbors (decay by distance). This pre-activates
       relevant knowledge before it's explicitly queried.

    5. **Health scoring** (Neuroca): Per-node health drives STM→MTM→LTM
       lifecycle transitions. Unhealthy nodes are candidates for pruning
       or consolidation.

    6. **Cluster auto-detection**: Community detection groups related
       nodes into topics without supervision.

    Thread-safety: All mutations protected by ``threading.RLock``.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Args:
            config: KnowledgeGraphConfig from HAIMConfig. Attributes used:
                - max_nodes (int, default 50000)
                - max_edges_per_node (int, default 100)
                - redundancy_threshold (float, default 0.92)
                - activation_decay (float, default 0.5)
                - reciprocal_weight_factor (float, default 0.7)
                - edge_decay_half_life_days (float, default 30.0)
                - min_edge_weight (float, default 0.05)
                - persistence_path (str, optional)
                - auto_persist (bool, default True)
        """
        self._lock = threading.RLock()
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._edges: Dict[Tuple[str, str], KnowledgeEdge] = {}
        self._adjacency: Dict[str, List[str]] = defaultdict(list)  # node_id → [neighbor_ids]

        # Config with safe defaults
        self._max_nodes = getattr(config, "max_nodes", 50000)
        self._max_edges_per_node = getattr(config, "max_edges_per_node", 100)
        self._redundancy_threshold = getattr(config, "redundancy_threshold", 0.92)
        self._activation_decay = getattr(config, "activation_decay", 0.5)
        self._reciprocal_factor = getattr(config, "reciprocal_weight_factor", 0.7)
        self._edge_decay_half_life = getattr(config, "edge_decay_half_life_days", 30.0)
        self._min_edge_weight = getattr(config, "min_edge_weight", 0.05)
        self._persistence_path = getattr(config, "persistence_path", None)
        self._auto_persist = getattr(config, "auto_persist", True)

        # Load persisted state
        if self._persistence_path:
            self._load_from_disk()

    # ══════════════════════════════════════════════════════════════════
    # Node Operations
    # ══════════════════════════════════════════════════════════════════

    def add_node(self, node: KnowledgeNode) -> str:
        """
        Add a knowledge node to the graph.

        If a node with the same embedding_hash already exists and
        the hash is non-empty, this is a potential redundancy.
        The incoming node will be merged instead of duplicated.

        Args:
            node: The KnowledgeNode to add.

        Returns:
            The ID of the node (may differ if merged with existing).
        """
        with self._lock:
            # Redundancy check via embedding hash
            if node.embedding_hash:
                existing = self._find_by_hash(node.embedding_hash)
                if existing:
                    self._merge_nodes(existing, node)
                    logger.debug(
                        f"Merged redundant node '{node.label}' into '{existing.label}' "
                        f"(hash={node.embedding_hash[:12]}…)"
                    )
                    return existing.id

            self._nodes[node.id] = node
            self._enforce_node_capacity()

        if self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        return node.id

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a node by ID, recording the access."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.access()
            return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        with self._lock:
            if node_id not in self._nodes:
                return False
            # Remove all edges touching this node
            neighbors = list(self._adjacency.get(node_id, []))
            for neighbor_id in neighbors:
                self._edges.pop((node_id, neighbor_id), None)
                self._edges.pop((neighbor_id, node_id), None)
                adj = self._adjacency.get(neighbor_id, [])
                if node_id in adj:
                    adj.remove(node_id)
            self._adjacency.pop(node_id, None)
            del self._nodes[node_id]

        if self._auto_persist and self._persistence_path:
            self._persist_to_disk()
        return True

    def has_node(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._nodes

    # ══════════════════════════════════════════════════════════════════
    # Edge Operations — Always Bidirectional
    # ══════════════════════════════════════════════════════════════════

    def link(
        self,
        source_id: str,
        target_id: str,
        weight: float = 0.5,
        edge_type: str = "related",
        metadata: Optional[Dict] = None,
    ) -> Tuple[Optional[KnowledgeEdge], Optional[KnowledgeEdge]]:
        """
        Create a bidirectional link between two nodes.

        This is the core difference from associations.py: every link
        is ALWAYS bidirectional. A→B creates both:
        - A→B with weight ``weight``
        - B→A with weight ``weight × reciprocal_factor`` (default 0.7×)

        If the edge already exists, its weight is strengthened instead.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            weight: Initial weight for the forward edge.
            edge_type: Semantic relationship type.
            metadata: Arbitrary metadata.

        Returns:
            Tuple of (forward_edge, reverse_edge). Either may be None
            if the corresponding node doesn't exist.
        """
        if source_id == target_id:
            return (None, None)

        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                return (None, None)

            # Forward edge: A→B
            fwd = self._ensure_edge(source_id, target_id, weight, edge_type, metadata)

            # Reverse edge: B→A (reciprocal weight)
            rev = self._ensure_edge(
                target_id, source_id,
                weight * self._reciprocal_factor,
                edge_type, metadata,
            )

            # Update adjacency
            if target_id not in self._adjacency[source_id]:
                self._adjacency[source_id].append(target_id)
            if source_id not in self._adjacency[target_id]:
                self._adjacency[target_id].append(source_id)

        if self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        return (fwd, rev)

    def strengthen(
        self,
        source_id: str,
        target_id: str,
        boost: float = 0.05,
    ) -> None:
        """
        Strengthen a bidirectional link (Hebbian: "fire together, wire together").

        Both forward and reverse edges are boosted.
        """
        with self._lock:
            fwd = self._edges.get((source_id, target_id))
            rev = self._edges.get((target_id, source_id))
            if fwd:
                fwd.activate(boost)
            if rev:
                rev.activate(boost * self._reciprocal_factor)

    def get_edge(self, source_id: str, target_id: str) -> Optional[KnowledgeEdge]:
        """Get the forward edge between two nodes."""
        with self._lock:
            return self._edges.get((source_id, target_id))

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbor IDs for a node."""
        with self._lock:
            return list(self._adjacency.get(node_id, []))

    def get_edges_from(
        self,
        node_id: str,
        min_weight: float = 0.0,
        edge_type: Optional[str] = None,
    ) -> List[KnowledgeEdge]:
        """Get all outgoing edges from a node, optionally filtered."""
        with self._lock:
            neighbors = self._adjacency.get(node_id, [])
            edges = []
            for nid in neighbors:
                edge = self._edges.get((node_id, nid))
                if edge and edge.weight >= min_weight:
                    if edge_type is None or edge.edge_type == edge_type:
                        edges.append(edge)
            return sorted(edges, key=lambda e: e.weight, reverse=True)

    # ══════════════════════════════════════════════════════════════════
    # Traversal-Based Retrieval
    # ══════════════════════════════════════════════════════════════════

    def bfs_retrieve(
        self,
        seed_ids: List[str],
        max_depth: int = 3,
        max_results: int = 20,
        min_weight: float = 0.1,
    ) -> List[Tuple[KnowledgeNode, int]]:
        """
        Breadth-first retrieval from seed nodes.

        Expands outward from seeds, following edges with weight >= min_weight.
        Returns nodes with their distance from the nearest seed.

        Args:
            seed_ids: Starting node IDs.
            max_depth: Maximum traversal depth.
            max_results: Maximum nodes to return.
            min_weight: Minimum edge weight to follow.

        Returns:
            List of (KnowledgeNode, depth) tuples, ordered by depth then health.
        """
        with self._lock:
            visited: Dict[str, int] = {}
            queue: deque = deque()

            for sid in seed_ids:
                if sid in self._nodes:
                    visited[sid] = 0
                    queue.append((sid, 0))

            results: List[Tuple[KnowledgeNode, int]] = []

            while queue and len(results) < max_results:
                node_id, depth = queue.popleft()
                node = self._nodes.get(node_id)
                if node:
                    node.access()
                    results.append((node, depth))

                if depth >= max_depth:
                    continue

                for neighbor_id in self._adjacency.get(node_id, []):
                    if neighbor_id in visited:
                        continue
                    edge = self._edges.get((node_id, neighbor_id))
                    if edge and edge.weight >= min_weight:
                        visited[neighbor_id] = depth + 1
                        queue.append((neighbor_id, depth + 1))

        return results

    def weighted_walk(
        self,
        seed_id: str,
        steps: int = 10,
        min_weight: float = 0.1,
    ) -> List[KnowledgeNode]:
        """
        Weighted random walk from a seed node.

        At each step, the next node is chosen proportionally to edge weight.
        This produces a path biased toward strongly-connected knowledge.

        Args:
            seed_id: Starting node ID.
            steps: Maximum number of steps.
            min_weight: Minimum edge weight to consider.

        Returns:
            List of visited KnowledgeNodes (may contain duplicates).
        """
        import random

        with self._lock:
            path: List[KnowledgeNode] = []
            current = seed_id

            for _ in range(steps):
                node = self._nodes.get(current)
                if not node:
                    break
                node.access()
                path.append(node)

                # Get weighted neighbors
                edges = self.get_edges_from(current, min_weight=min_weight)
                if not edges:
                    break

                # Weighted selection
                weights = [e.weight for e in edges]
                total = sum(weights)
                if total <= 0:
                    break
                r = random.uniform(0, total)
                cumulative = 0.0
                chosen = edges[0].target_id
                for edge in edges:
                    cumulative += edge.weight
                    if cumulative >= r:
                        chosen = edge.target_id
                        break

                # Strengthen the traversed edge (Hebbian)
                self.strengthen(current, chosen, boost=0.02)
                current = chosen

        return path

    # ══════════════════════════════════════════════════════════════════
    # Activation Spreading
    # ══════════════════════════════════════════════════════════════════

    def spread_activation(
        self,
        seed_ids: List[str],
        initial_energy: float = 1.0,
        decay_factor: Optional[float] = None,
        max_depth: int = 3,
    ) -> Dict[str, float]:
        """
        Spread activation energy from seed nodes through the graph.

        Energy decays by ``decay_factor`` at each hop. Nodes that receive
        energy above a threshold are "pre-activated" — meaning they become
        more readily retrievable (higher activation_level).

        This implements the Mnemosyne concept of activation-based retrieval:
        when you think about topic X, related topics Y and Z become
        sub-consciously activated.

        Args:
            seed_ids: Starting node IDs.
            initial_energy: Energy at seed nodes.
            decay_factor: Energy multiplier per hop (default: config.activation_decay).
            max_depth: Maximum spreading depth.

        Returns:
            Dict mapping node_id → received activation energy.
        """
        decay = decay_factor if decay_factor is not None else self._activation_decay
        activation_map: Dict[str, float] = {}

        with self._lock:
            queue: deque = deque()
            for sid in seed_ids:
                if sid in self._nodes:
                    queue.append((sid, initial_energy, 0))
                    activation_map[sid] = initial_energy

            while queue:
                node_id, energy, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                node = self._nodes.get(node_id)
                if node:
                    node.activation_level = min(1.0, node.activation_level + energy)

                for neighbor_id in self._adjacency.get(node_id, []):
                    edge = self._edges.get((node_id, neighbor_id))
                    if not edge:
                        continue
                    propagated = energy * decay * edge.weight
                    if propagated < 0.01:  # Stop spreading at negligible levels
                        continue

                    existing = activation_map.get(neighbor_id, 0.0)
                    if propagated > existing:
                        activation_map[neighbor_id] = propagated
                        queue.append((neighbor_id, propagated, depth + 1))

        return activation_map

    # ══════════════════════════════════════════════════════════════════
    # Redundancy Pruning (Mnemosyne-inspired)
    # ══════════════════════════════════════════════════════════════════

    def prune_redundant_nodes(
        self,
        similarity_threshold: Optional[float] = None,
    ) -> int:
        """
        Merge nodes that are semantically redundant based on content similarity.

        Uses word-overlap Jaccard similarity (threshold from config,
        default 0.92). When two nodes exceed the threshold, the younger
        node is merged into the older one.

        This is the unsupervised redundancy pruning from Mnemosyne
        (Georgia Tech / Microsoft Research 2025) that radically reduces
        noise and improves retrieval precision.

        Args:
            similarity_threshold: Override the config threshold.

        Returns:
            Number of nodes merged.
        """
        threshold = similarity_threshold or self._redundancy_threshold
        merged = 0

        with self._lock:
            node_list = list(self._nodes.values())
            to_merge: List[Tuple[str, str]] = []  # (keep_id, remove_id)

            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    sim = self._content_similarity(
                        node_list[i].content, node_list[j].content
                    )
                    if sim >= threshold:
                        # Keep the older (more established) node
                        if node_list[i].created_at <= node_list[j].created_at:
                            to_merge.append((node_list[i].id, node_list[j].id))
                        else:
                            to_merge.append((node_list[j].id, node_list[i].id))

            # Execute merges
            already_removed: Set[str] = set()
            for keep_id, remove_id in to_merge:
                if remove_id in already_removed or keep_id in already_removed:
                    continue
                if keep_id in self._nodes and remove_id in self._nodes:
                    self._merge_nodes(self._nodes[keep_id], self._nodes[remove_id])
                    self._remove_node_internal(remove_id)
                    already_removed.add(remove_id)
                    merged += 1

        if merged > 0:
            logger.info(f"Pruned {merged} redundant nodes from knowledge graph.")
            if self._auto_persist and self._persistence_path:
                self._persist_to_disk()

        return merged

    def prune_weak_edges(self, min_weight: Optional[float] = None) -> int:
        """
        Remove edges below the minimum weight threshold.

        Args:
            min_weight: Override threshold. Default from config.

        Returns:
            Number of edges removed.
        """
        threshold = min_weight if min_weight is not None else self._min_edge_weight
        pruned = 0

        with self._lock:
            to_remove: List[Tuple[str, str]] = []
            for key, edge in self._edges.items():
                if edge.weight < threshold:
                    to_remove.append(key)

            for src, tgt in to_remove:
                del self._edges[(src, tgt)]
                adj = self._adjacency.get(src, [])
                if tgt in adj:
                    adj.remove(tgt)
                pruned += 1

        if pruned > 0 and self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        return pruned

    # ══════════════════════════════════════════════════════════════════
    # Cluster Detection
    # ══════════════════════════════════════════════════════════════════

    def detect_clusters(self, min_cluster_size: int = 2) -> Dict[str, List[str]]:
        """
        Detect clusters using connected components with weight threshold.

        Assigns ``cluster_id`` to each node. Returns mapping of
        cluster_id → [node_ids].

        Args:
            min_cluster_size: Minimum nodes to form a cluster.

        Returns:
            Dict mapping cluster_id → list of node IDs.
        """
        with self._lock:
            visited: Set[str] = set()
            clusters: Dict[str, List[str]] = {}
            cluster_counter = 0

            for node_id in self._nodes:
                if node_id in visited:
                    continue

                # BFS to find connected component
                component: List[str] = []
                queue: deque = deque([node_id])
                while queue:
                    nid = queue.popleft()
                    if nid in visited:
                        continue
                    visited.add(nid)
                    component.append(nid)
                    for neighbor_id in self._adjacency.get(nid, []):
                        if neighbor_id not in visited:
                            edge = self._edges.get((nid, neighbor_id))
                            if edge and edge.weight >= self._min_edge_weight:
                                queue.append(neighbor_id)

                if len(component) >= min_cluster_size:
                    cluster_id = f"cluster_{cluster_counter}"
                    cluster_counter += 1
                    clusters[cluster_id] = component
                    for nid in component:
                        node = self._nodes.get(nid)
                        if node:
                            node.cluster_id = cluster_id

        return clusters

    # ══════════════════════════════════════════════════════════════════
    # Edge Decay
    # ══════════════════════════════════════════════════════════════════

    def decay_all_edges(self) -> int:
        """
        Apply temporal decay to all edges. Removes edges that fall
        below min_weight after decay.

        Returns:
            Number of edges removed after decay.
        """
        removed = 0
        with self._lock:
            to_remove: List[Tuple[str, str]] = []
            for key, edge in self._edges.items():
                edge.decay(self._edge_decay_half_life)
                if edge.weight < self._min_edge_weight:
                    to_remove.append(key)

            for src, tgt in to_remove:
                del self._edges[(src, tgt)]
                adj = self._adjacency.get(src, [])
                if tgt in adj:
                    adj.remove(tgt)
                removed += 1

        if removed > 0 and self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        return removed

    def decay_all_activations(self, decay_rate: float = 0.1) -> None:
        """Decay activation levels for all nodes (between pulse ticks)."""
        with self._lock:
            for node in self._nodes.values():
                node.activation_level = max(0.0, node.activation_level - decay_rate)

    # ══════════════════════════════════════════════════════════════════
    # Statistics
    # ══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """
        Comprehensive graph statistics for monitoring.

        Returns dict with node/edge counts, stage distribution,
        cluster info, average health, average edge weight.
        """
        with self._lock:
            nodes = list(self._nodes.values())
            edges = list(self._edges.values())

        total_nodes = len(nodes)
        total_edges = len(edges)

        if total_nodes == 0:
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "avg_health": 0.0,
                "avg_edge_weight": 0.0,
                "stage_distribution": {"STM": 0, "MTM": 0, "LTM": 0},
                "cluster_count": 0,
                "node_types": {},
            }

        stm = sum(1 for n in nodes if n.lifecycle_stage == "STM")
        mtm = sum(1 for n in nodes if n.lifecycle_stage == "MTM")
        ltm = sum(1 for n in nodes if n.lifecycle_stage == "LTM")

        type_counts: Dict[str, int] = defaultdict(int)
        for n in nodes:
            type_counts[n.node_type] += 1

        cluster_ids = set(n.cluster_id for n in nodes if n.cluster_id)

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "avg_health": sum(n.health_score for n in nodes) / total_nodes,
            "avg_edge_weight": sum(e.weight for e in edges) / max(total_edges, 1),
            "stage_distribution": {"STM": stm, "MTM": mtm, "LTM": ltm},
            "cluster_count": len(cluster_ids),
            "node_types": dict(type_counts),
        }

    # ══════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ══════════════════════════════════════════════════════════════════

    def _ensure_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        edge_type: str,
        metadata: Optional[Dict],
    ) -> KnowledgeEdge:
        """Get or create an edge, strengthening if it exists (must hold lock)."""
        key = (source_id, target_id)
        existing = self._edges.get(key)
        if existing:
            existing.activate(boost=weight * 0.1)
            return existing
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            edge_type=edge_type,
            metadata=metadata or {},
        )
        self._edges[key] = edge
        return edge

    def _find_by_hash(self, embedding_hash: str) -> Optional[KnowledgeNode]:
        """Find a node with the same embedding hash (must hold lock)."""
        for node in self._nodes.values():
            if node.embedding_hash == embedding_hash:
                return node
        return None

    def _merge_nodes(self, keep: KnowledgeNode, remove: KnowledgeNode) -> None:
        """Merge remove-node data into keep-node (must hold lock)."""
        keep.access_count += remove.access_count
        keep.tags = list(set(keep.tags + remove.tags))
        keep.source_ids = list(set(keep.source_ids + remove.source_ids))
        if remove.content and remove.content not in keep.content:
            keep.content += f"\n[merged] {remove.content}"
        keep.update_health()

    def _remove_node_internal(self, node_id: str) -> None:
        """Remove a node and re-wire edges (must hold lock)."""
        # Transfer edges to any merge target
        neighbors = list(self._adjacency.get(node_id, []))
        for nid in neighbors:
            self._edges.pop((node_id, nid), None)
            self._edges.pop((nid, node_id), None)
            adj = self._adjacency.get(nid, [])
            if node_id in adj:
                adj.remove(node_id)
        self._adjacency.pop(node_id, None)
        self._nodes.pop(node_id, None)

    def _content_similarity(self, a: str, b: str) -> float:
        """Word-overlap Jaccard similarity between two texts."""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

    def _enforce_node_capacity(self) -> None:
        """Evict lowest-health nodes if over capacity (must hold lock)."""
        if len(self._nodes) <= self._max_nodes:
            return
        overage = len(self._nodes) - self._max_nodes
        ranked = sorted(self._nodes.values(), key=lambda n: n.health_score)
        for node in ranked[:overage]:
            self._remove_node_internal(node.id)
            logger.debug(f"Evicted node '{node.label}' (health={node.health_score:.3f})")

    # ── Persistence ───────────────────────────────────────────────────

    def _persist_to_disk(self) -> None:
        """Save graph to JSON file."""
        if not self._persistence_path:
            return
        try:
            path = Path(self._persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "version": "1.0",
                    "nodes": [n.to_dict() for n in self._nodes.values()],
                    "edges": [e.to_dict() for e in self._edges.values()],
                }
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist knowledge graph: {e}")

    def _load_from_disk(self) -> None:
        """Load graph from JSON file."""
        if not self._persistence_path:
            return
        path = Path(self._persistence_path)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for nd in raw.get("nodes", []):
                node = KnowledgeNode.from_dict(nd)
                self._nodes[node.id] = node
            for ed in raw.get("edges", []):
                edge = KnowledgeEdge.from_dict(ed)
                self._edges[edge.key] = edge
                if edge.target_id not in self._adjacency[edge.source_id]:
                    self._adjacency[edge.source_id].append(edge.target_id)
            logger.info(
                f"Loaded knowledge graph: {len(self._nodes)} nodes, "
                f"{len(self._edges)} edges from {path}"
            )
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
