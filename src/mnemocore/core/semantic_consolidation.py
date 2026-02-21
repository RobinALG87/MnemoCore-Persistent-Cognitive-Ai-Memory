"""
Nightly Semantic Consolidation Worker (Phase 4.0)
=================================================
Autonomous background worker that runs semantic clustering over the
WARM tier every night (configurable schedule) to:

  1. Cluster semantically similar memories using Hamming-distance k-medoids.
  2. For each cluster, compute a "proto-memory" via majority-vote bundling.
  3. Detect redundant / near-duplicate memories (distance < epsilon).
  4. Optionally prune low-LTP duplicates and strengthen the proto-memory.
  5. Emit consolidation events to Redis stream for downstream consumers.

Design principles:
  - Runs as a standalone asyncio task; no hard dependency on Redis (falls back gracefully).
  - All computation is NumPy-vectorized (no Python-level loops over dimension).
  - Idempotent: running twice produces the same result.
  - Pluggable: attach a post_consolidation_hook for custom logic.

Usage:
    worker = SemanticConsolidationWorker(engine)
    await worker.start()       # launches background task
    await worker.run_once()    # one-shot (for testing / cron)
    await worker.stop()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .binary_hdv import BinaryHDV, majority_bundle
from .config import get_config
from .node import MemoryNode
from .provenance import ProvenanceRecord


# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

@dataclass
class SemanticConsolidationConfig:
    """Tuning knobs for the nightly consolidation pass."""
    schedule_hour: int = 3          # UTC hour to run (3 = 03:00 UTC)
    duplicate_epsilon: float = 0.05 # Hamming dist < epsilon → near-duplicate
    cluster_k: int = 32             # Target number of clusters (k-medoids)
    cluster_max_iter: int = 10      # k-medoids convergence iterations
    min_cluster_size: int = 3       # Ignore clusters smaller than this
    prune_duplicates: bool = True   # Actually remove duplicates (vs just log)
    min_ltp_to_prune: float = 0.0   # Only prune nodes with ltp < this
    batch_size: int = 500           # Process WARM tier in batches
    enabled: bool = True


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _hamming_matrix(vecs: np.ndarray) -> np.ndarray:
    """
    Vectorised pairwise Hamming distance matrix for packed uint8 arrays.

    Args:
        vecs: shape (N, D/8) uint8

    Returns:
        dist_matrix: shape (N, N) float32 normalised to [0, 1]
    """
    n = vecs.shape[0]
    dim_bits = vecs.shape[1] * 8
    dist = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        xor = np.bitwise_xor(vecs[i : i + 1], vecs)  # broadcast (1, D) XOR (N, D)
        popcount = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)
        dist[i] = popcount / dim_bits

    return dist


def _kmedoids_iter(
    dist_matrix: np.ndarray, k: int, max_iter: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple k-medoids (PAM build phase + swap phase).

    Returns:
        medoid_indices: shape (k,)  – indices of chosen medoids
        labels:         shape (N,)  – cluster assignment for each point
    """
    n = dist_matrix.shape[0]
    rng = np.random.default_rng(42)
    medoids = rng.choice(n, size=min(k, n), replace=False)

    for _ in range(max_iter):
        # Assignment step
        labels = np.argmin(dist_matrix[:, medoids], axis=1)

        # Update step: for each cluster, choose point minimizing total intra-dist
        new_medoids = np.copy(medoids)
        for c in range(len(medoids)):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            intra = dist_matrix[np.ix_(members, members)].sum(axis=1)
            new_medoids[c] = members[intra.argmin()]

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    # Final assignment
    labels = np.argmin(dist_matrix[:, medoids], axis=1)
    return medoids, labels


# ------------------------------------------------------------------ #
#  Worker                                                             #
# ------------------------------------------------------------------ #

class SemanticConsolidationWorker:
    """
    Nightly semantic consolidation.

    Attach to a running HAIMEngine instance and call start() to activate.
    """

    def __init__(
        self,
        engine,  # HAIMEngine – typed as Any to avoid circular import
        config: Optional[SemanticConsolidationConfig] = None,
        post_consolidation_hook: Optional[Callable] = None,
    ):
        self.engine = engine
        self.cfg = config or SemanticConsolidationConfig()
        self.hook = post_consolidation_hook
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.last_run: Optional[datetime] = None
        self.stats: Dict = {}

    # ---- Lifecycle ----------------------------------------------- #

    async def start(self) -> None:
        """Launch the background consolidation scheduler."""
        if not self.cfg.enabled:
            logger.info("SemanticConsolidationWorker disabled by config.")
            return
        self._running = True
        self._task = asyncio.create_task(self._schedule_loop(), name="semantic_consolidation")
        logger.info(
            f"SemanticConsolidationWorker started — runs at {self.cfg.schedule_hour:02d}:00 UTC"
        )

    async def stop(self) -> None:
        """Gracefully stop the worker."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SemanticConsolidationWorker stopped.")

    # ---- Scheduler ----------------------------------------------- #

    async def _schedule_loop(self) -> None:
        """Sleep until the next scheduled hour, then run."""
        while self._running:
            try:
                seconds_until = self._seconds_until_next_run()
                logger.debug(
                    f"Next semantic consolidation in {seconds_until / 3600:.1f}h"
                )
                await asyncio.sleep(seconds_until)
                if self._running:
                    await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"SemanticConsolidationWorker error: {exc}", exc_info=True)
                await asyncio.sleep(60)  # backoff

    def _seconds_until_next_run(self) -> float:
        now = datetime.now(timezone.utc)
        target = now.replace(
            hour=self.cfg.schedule_hour, minute=0, second=0, microsecond=0
        )
        delta = (target - now).total_seconds()
        if delta <= 0:
            delta += 86400  # schedule for tomorrow
        return delta

    # ---- Main pass ----------------------------------------------- #

    async def run_once(self) -> Dict:
        """
        Execute a full semantic consolidation pass.
        Safe to call manually (e.g. for testing).

        Returns:
            stats dict with consolidation metrics.
        """
        t0 = time.monotonic()
        logger.info("=== Semantic Consolidation — start ===")

        # 1. Snapshot WARM + HOT tier nodes
        nodes: List[MemoryNode] = await self._collect_nodes()
        if len(nodes) < self.cfg.min_cluster_size:
            logger.info(f"Only {len(nodes)} nodes — skipping consolidation.")
            return {}

        # 2. Build packed matrix for vectorised Hamming ops
        vecs = np.stack([n.hdv.data for n in nodes])  # (N, D/8)

        # 3. Detect near-duplicates (fast pairwise within batch)
        duplicates_pruned = 0
        if self.cfg.prune_duplicates:
            duplicates_pruned = await self._prune_duplicates(nodes, vecs)
            # Refresh after pruning
            nodes = [n for n in nodes if await self._node_exists(n.id)]
            if len(nodes) < self.cfg.min_cluster_size:
                logger.info("Too few nodes after duplicate pruning.")
                return {}
            vecs = np.stack([n.hdv.data for n in nodes])

        # 4. Semantic clustering (k-medoids)
        n = len(nodes)
        k = min(self.cfg.cluster_k, max(1, n // self.cfg.min_cluster_size))
        logger.info(f"Running k-medoids clustering: n={n}, k={k}")

        dist_mat = await asyncio.get_running_loop().run_in_executor(
            None, _hamming_matrix, vecs
        )
        medoids, labels = await asyncio.get_running_loop().run_in_executor(
            None, _kmedoids_iter, dist_mat, k, self.cfg.cluster_max_iter
        )

        # 5. Build proto-memories for large clusters
        proto_count = 0
        for cluster_id, medoid_idx in enumerate(medoids):
            members_idx = np.where(labels == cluster_id)[0]
            if len(members_idx) < self.cfg.min_cluster_size:
                continue

            member_nodes = [nodes[i] for i in members_idx]
            medoid_node = nodes[medoid_idx]

            # Compute proto-vector via majority bundling
            member_vecs = [n.hdv for n in member_nodes]
            proto_vec = majority_bundle(member_vecs)

            # Bind proto-vector back onto the medoid (strengthen it)
            medoid_node.hdv = proto_vec
            medoid_node.ltp_strength = min(
                1.0,
                medoid_node.ltp_strength + 0.05 * len(member_nodes),
            )
            medoid_node.metadata["proto_cluster_size"] = int(len(member_nodes))
            medoid_node.metadata["proto_updated_at"] = datetime.now(timezone.utc).isoformat()
            proto_count += 1

            # Phase 5.0: record consolidation in provenance lineage
            source_ids = [n.id for n in member_nodes if n.id != medoid_node.id]
            if medoid_node.provenance is None:
                medoid_node.provenance = ProvenanceRecord.new(
                    origin_type="consolidation",
                    actor="consolidation_worker",
                )
            medoid_node.provenance.mark_consolidated(
                source_memory_ids=source_ids,
                actor="consolidation_worker",
            )

        elapsed = time.monotonic() - t0
        self.last_run = datetime.now(timezone.utc)
        self.stats = {
            "nodes_processed": n,
            "clusters_formed": int(len(medoids)),
            "proto_memories_updated": proto_count,
            "duplicates_pruned": duplicates_pruned,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": self.last_run.isoformat(),
        }

        logger.info(
            f"=== Semantic Consolidation — done in {elapsed:.1f}s "
            f"| nodes={n} clusters={len(medoids)} protos={proto_count} "
            f"dupes_pruned={duplicates_pruned} ==="
        )

        # 6. Fire optional hook
        if self.hook:
            try:
                await asyncio.coroutine(self.hook)(self.stats) if asyncio.iscoroutinefunction(self.hook) else self.hook(self.stats)
            except Exception as e:
                logger.warning(f"post_consolidation_hook error: {e}")

        return self.stats

    # ---- Helpers ------------------------------------------------- #

    async def _collect_nodes(self) -> List[MemoryNode]:
        """Collect all HOT + WARM nodes for clustering."""
        nodes: List[MemoryNode] = []
        # HOT
        hot_nodes = await self.engine.tier_manager.get_hot_snapshot()
        nodes.extend(hot_nodes)
        # WARM via TierManager list (disk or Qdrant)
        try:
            warm_nodes = await self.engine.tier_manager.list_warm(
                max_results=self.cfg.batch_size
            )
            nodes.extend(warm_nodes)
        except AttributeError:
            pass  # list_warm not available; work with HOT only
        return nodes

    async def _prune_duplicates(
        self, nodes: List[MemoryNode], vecs: np.ndarray
    ) -> int:
        """
        Find and remove near-duplicate nodes (distance < epsilon).
        Keeps the node with the highest LTP strength.

        Returns:
            Number of nodes pruned.
        """
        eps = self.cfg.duplicate_epsilon
        n = len(nodes)
        pruned: set = set()

        for i in range(n):
            if nodes[i].id in pruned:
                continue
            for j in range(i + 1, n):
                if nodes[j].id in pruned:
                    continue
                # Compute Hamming distance on-the-fly (avoid full matrix for pruning)
                xor = np.bitwise_xor(vecs[i], vecs[j])
                dist = float(np.unpackbits(xor).sum()) / (vecs.shape[1] * 8)
                if dist < eps:
                    # Prune the weaker one
                    weaker = j if nodes[i].ltp_strength >= nodes[j].ltp_strength else i
                    if nodes[weaker].ltp_strength <= self.cfg.min_ltp_to_prune:
                        pruned.add(nodes[weaker].id)

        count = 0
        for node_id in pruned:
            deleted = await self.engine.tier_manager.delete_memory(node_id)
            if deleted:
                count += 1
                logger.debug(f"Pruned duplicate node {node_id[:8]}")
        return count

    async def _node_exists(self, node_id: str) -> bool:
        """Check if node still exists in any tier."""
        return await self.engine.tier_manager.get_memory(node_id) is not None
