"""
Synapse Adjacency Index — O(1) Lookup (Phase 4.0)
=================================================
Provides a hardened, thread-safe adjacency data structure for synaptic
connections with guaranteed O(1) lookup, O(1) insert, and O(k) neighbour
enumeration (where k = degree of a node).

Design:
  - Primary store: Dict[Tuple[str,str], SynapticConnection]
    Key is always sorted(a, b) to ensure uniqueness regardless of direction.
  - Adjacency index: Dict[str, set[str]]
    Maps node_id → set of connected node_ids.  Lookup is O(1), iteration O(k).
  - The set-based adjacency replaces the previous list-based one to prevent
    duplicate edges and make removals O(1) instead of O(k).

Phase 4.0 additions:
  - Bayesian LTP state is serialised alongside Hebbian state on save.
  - adjacency_degree() exposes per-node connectivity (used by immunology sweep).
  - to_dict() / from_dict() for full serialisation round-trips.
  - compact() removes all edges below a strength threshold in O(E) time.

This module is intentionally dependency-light: it only imports stdlib +
SynapticConnection + BayesianLTPUpdater from this package.
"""

from __future__ import annotations

from mnemocore.utils import json_compat as json
import asyncio
import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Set, Tuple
from loguru import logger


# ------------------------------------------------------------------ #
#  Cached Bayesian updater import                                     #
# ------------------------------------------------------------------ #

_bayesian_updater_cache = None

def _get_bayesian_updater():
    """
    Get the cached Bayesian updater instance.

    Caches after first import to avoid repeated module loading.
    Thread-safe via module-level import lock in Python.
    """
    global _bayesian_updater_cache
    if _bayesian_updater_cache is None:
        from .bayesian_ltp import get_bayesian_updater
        _bayesian_updater_cache = get_bayesian_updater()
    return _bayesian_updater_cache


# ------------------------------------------------------------------ #
#  Adjacency index                                                    #
# ------------------------------------------------------------------ #

class SynapseIndex:
    """
    O(1) synaptic adjacency index.

    Thread-safety: This class has internal asyncio.Lock protection for all
    mutating operations. Callers can optionally use external locking for
    compound operations, but the class is safe to use on its own.

    Internals:
        _edges: Dict[Tuple[str,str], SynapticConnection]   primary edge store
        _adj:   Dict[str, Set[str]]                        adjacency sets
        _lock:  asyncio.Lock for thread-safe mutations
    """

    def __init__(self):
        from .synapse import SynapticConnection  # local import
        self._SynapticConnection = SynapticConnection
        self._edges: Dict[Tuple[str, str], "SynapticConnection"] = {}
        self._adj: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    # ---- Public API --------------------------------------------- #

    async def register(self, syn: "SynapticConnection") -> None:
        """
        Register an already-constructed SynapticConnection into the index.

        Use this instead of poking at _edges/_adj directly when you already
        have a SynapticConnection object (e.g. during legacy-dict sync in
        cleanup_decay).  No Bayesian observation is made – the connection is
        accepted as-is.

        O(1).
        """
        async with self._lock:
            key = _key(syn.neuron_a_id, syn.neuron_b_id)
            if key not in self._edges:
                self._edges[key] = syn
                self._adj.setdefault(key[0], set()).add(key[1])
                self._adj.setdefault(key[1], set()).add(key[0])

    async def add_or_fire(self, id_a: str, id_b: str, success: bool = True, weight: float = 1.0) -> "SynapticConnection":
        """
        Create a synapse if it doesn't exist, then fire it.

        O(1) operation.
        Returns the (potentially new) SynapticConnection.
        """
        async with self._lock:
            key = _key(id_a, id_b)
            if key not in self._edges:
                syn = self._SynapticConnection(key[0], key[1])
                self._edges[key] = syn
                self._adj.setdefault(key[0], set()).add(key[1])
                self._adj.setdefault(key[1], set()).add(key[0])
                logger.debug(f"Synapse created: {key[0][:8]} <-> {key[1][:8]}")

            syn = self._edges[key]

            # Phase 4.0: Bayesian update first, then Hebbian fire
            upd = _get_bayesian_updater()
            upd.observe_synapse(syn, success=success)
            # Also call the Hebbian fire for backward compat (updates fire_count etc.)
            syn.fire(success=success, weight=weight)

            return syn

    async def get(self, id_a: str, id_b: str) -> Optional["SynapticConnection"]:
        """O(1) edge lookup. Returns None if no edge exists."""
        async with self._lock:
            return self._edges.get(_key(id_a, id_b))

    async def neighbours(self, node_id: str) -> List["SynapticConnection"]:
        """
        Return all SynapticConnections adjacent to node_id.

        O(k) where k is the degree.
        """
        async with self._lock:
            neighbour_ids = self._adj.get(node_id, set())
            result = []
            for nid in neighbour_ids:
                syn = self._edges.get(_key(node_id, nid))
                if syn:
                    result.append(syn)
            return result

    async def get_multi_hop_neighbors(self, node_id: str, depth: int = 2) -> Dict[str, float]:
        """
        Phase 12.1: Traverse graph up to `depth` hops away.
        Returns a mapping of node_id -> maximum cumulative connection strength path.
        """
        async with self._lock:
            visited = {node_id: 1.0}
            current_layer = {node_id: 1.0}

            for _ in range(depth):
                next_layer = {}
                for curr_node, cum_weight in current_layer.items():
                    neighbour_ids = self._adj.get(curr_node, set())
                    for nid in neighbour_ids:
                        syn = self._edges.get(_key(curr_node, nid))
                        if not syn:
                            continue
                        neighbor_id = syn.neuron_b_id if syn.neuron_a_id == curr_node else syn.neuron_a_id
                        if neighbor_id == node_id:
                            continue

                        edge_weight = syn.get_current_strength()
                        new_weight = cum_weight * edge_weight

                        # Store the strongest path to the node
                        if neighbor_id not in visited or new_weight > visited[neighbor_id]:
                            visited[neighbor_id] = new_weight

                        if neighbor_id not in next_layer or new_weight > next_layer[neighbor_id]:
                            next_layer[neighbor_id] = new_weight
                current_layer = next_layer

            visited.pop(node_id, None)
            return visited

    async def neighbour_ids(self, node_id: str) -> Set[str]:
        """O(1) set of connected node IDs."""
        async with self._lock:
            return self._adj.get(node_id, set()).copy()

    async def remove_node(self, node_id: str) -> int:
        """
        Remove all edges involving node_id.

        O(k) where k is the degree.
        Returns number of edges removed.
        """
        async with self._lock:
            neighbours = self._adj.pop(node_id, set())
            removed = 0
            for nid in neighbours:
                key = _key(node_id, nid)
                if self._edges.pop(key, None) is not None:
                    removed += 1
                # Remove the reverse adjacency entry
                if nid in self._adj:
                    self._adj[nid].discard(node_id)
                    if not self._adj[nid]:
                        del self._adj[nid]
            return removed

    async def remove_edge(self, id_a: str, id_b: str) -> bool:
        """
        Remove a single edge.

        O(1).  Returns True if the edge existed.
        """
        async with self._lock:
            key = _key(id_a, id_b)
            syn = self._edges.pop(key, None)
            if syn is None:
                return False
            # Clean adjacency sets
            if key[0] in self._adj:
                self._adj[key[0]].discard(key[1])
                if not self._adj[key[0]]:
                    del self._adj[key[0]]
            if key[1] in self._adj:
                self._adj[key[1]].discard(key[0])
                if not self._adj[key[1]]:
                    del self._adj[key[1]]
            return True

    async def compact(self, threshold: float = 0.05) -> int:
        """
        Remove all edges whose decayed strength is below `threshold`.

        O(E) where E = total edge count.
        Returns number of edges removed.
        """
        async with self._lock:
            dead_keys = [
                k for k, s in self._edges.items()
                if s.get_current_strength() < threshold
            ]
            for key in dead_keys:
                syn = self._edges.pop(key)
                if syn.neuron_a_id in self._adj:
                    self._adj[syn.neuron_a_id].discard(syn.neuron_b_id)
                    if not self._adj[syn.neuron_a_id]:
                        del self._adj[syn.neuron_a_id]
                if syn.neuron_b_id in self._adj:
                    self._adj[syn.neuron_b_id].discard(syn.neuron_a_id)
                    if not self._adj[syn.neuron_b_id]:
                        del self._adj[syn.neuron_b_id]

            if dead_keys:
                logger.info(f"SynapseIndex.compact: removed {len(dead_keys)} dead edges.")
            return len(dead_keys)

    async def adjacency_degree(self, node_id: str) -> int:
        """O(1) degree query."""
        async with self._lock:
            return len(self._adj.get(node_id, set()))

    async def boost(self, node_id: str) -> float:
        """
        Compute synaptic boost multiplier for a node (used in scoring).

        boost = product (1 + strength_i)  over all edges i adjacent to node_id.

        Returns 1.0 for isolated nodes.
        """
        async with self._lock:
            # Phase 4.5 Hotfix (Robin's Score Bug e+195):
            # Instead of exponential product scaling which explodes for hub nodes,
            # we aggregate the strengths and bound the multiplier logarithmically.
            # User Fix:
            # Logarithmic boost - capped at ~10x
            neighbour_ids = self._adj.get(node_id, set())
            total = 0.0
            for nid in neighbour_ids:
                syn = self._edges.get(_key(node_id, nid))
                if syn:
                    total += syn.get_current_strength()
            return 1.0 + min(math.log(1.0 + total), 9.0)

    def __bool__(self) -> bool:
        """Sync truth check — always True (instance exists)."""
        return True

    async def __len__(self) -> int:
        async with self._lock:
            return len(self._edges)

    def __iter__(self) -> Iterator[Tuple[Tuple[str, str], "SynapticConnection"]]:
        """Synchronous iterator over edge items (snapshot). Use items() for async."""
        return iter(list(self._edges.items()))

    async def items(self):
        async with self._lock:
            return list(self._edges.items())

    async def values(self):
        async with self._lock:
            return list(self._edges.values())

    # ---- Persistence -------------------------------------------- #

    async def to_jsonl(self) -> List[str]:
        """
        Serialise all edges to JSONL records (Phase 4.0: includes Bayesian state).
        """
        async with self._lock:
            lines = []
            upd = _get_bayesian_updater()
            for syn in self._edges.values():
                rec = {
                    "neuron_a_id": syn.neuron_a_id,
                    "neuron_b_id": syn.neuron_b_id,
                    "strength": syn.strength,
                    "fire_count": syn.fire_count,
                    "success_count": syn.success_count,
                    "last_fired": syn.last_fired.isoformat() if syn.last_fired else None,
                    "bayes": upd.synapse_to_dict(syn),  # Phase 4.0
                }
                lines.append(json.dumps(rec))
            return lines

    async def load_jsonl(self, lines: List[str]) -> None:
        """
        Restore edges from JSONL records (Phase 4.0: restores Bayesian state).
        """
        async with self._lock:
            self._edges.clear()
            self._adj.clear()
            upd = _get_bayesian_updater()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    syn = self._SynapticConnection(
                        rec["neuron_a_id"], rec["neuron_b_id"], rec["strength"]
                    )
                    syn.fire_count = rec.get("fire_count", 0)
                    syn.success_count = rec.get("success_count", 0)
                    if rec.get("last_fired"):
                        syn.last_fired = datetime.fromisoformat(rec["last_fired"])

                    # Phase 4.0: restore Bayesian state
                    if "bayes" in rec:
                        upd.synapse_from_dict(syn, rec["bayes"])

                    key = _key(syn.neuron_a_id, syn.neuron_b_id)
                    self._edges[key] = syn
                    self._adj.setdefault(key[0], set()).add(key[1])
                    self._adj.setdefault(key[1], set()).add(key[0])
                except Exception as exc:
                    logger.warning(f"SynapseIndex.load_jsonl: skipping bad record: {exc}")

    async def save_to_file(self, path: str) -> None:
        """Save index to a JSONL file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            lines = await self.to_jsonl()
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        except Exception as exc:
            logger.error(f"SynapseIndex.save_to_file failed: {exc}")

    async def load_from_file(self, path: str) -> None:
        """Load index from a JSONL file."""
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            await self.load_jsonl(lines)
            async with self._lock:
                logger.info(
                    f"SynapseIndex loaded {len(self._edges)} edges from {path}"
                )
        except Exception as exc:
            logger.error(f"SynapseIndex.load_from_file failed: {exc}")

    @property
    async def stats(self) -> Dict:
        async with self._lock:
            return {
                "edge_count": len(self._edges),
                "node_count": len(self._adj),
                "avg_degree": (
                    sum(len(v) for v in self._adj.values()) / len(self._adj)
                    if self._adj else 0.0
                ),
            }


# ------------------------------------------------------------------ #
#  Helper                                                             #
# ------------------------------------------------------------------ #

def _key(a: str, b: str) -> Tuple[str, str]:
    """Canonical, order-independent edge key."""
    return (a, b) if a < b else (b, a)
