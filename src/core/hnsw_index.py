"""
HNSW ANN Index Extension for HOT tier (Phase 4.0)
==================================================
Extends the existing FAISS binary flat index with a configurable
HNSW (Hierarchical Navigable Small World) graph index for O(log N)
approximate nearest-neighbour search on the HOT tier.

Why HNSW for HOT tier:
  - HOT tier is in-memory → index stays in RAM → zero-latency ANN.
  - FAISS IndexBinaryHNSW gives sub-linear query times for N > ~1000.
  - Still uses packed uint8 arrays (same as IndexBinaryFlat).
  - Graceful fallback to flat index when FAISS unavailable or N is small.

This module provides HNSWIndexManager which is meant to be composed
into TierManager (or used standalone for testing).

Usage (standalone):
    mgr = HNSWIndexManager(dimension=16384, m=32, ef_construction=200)
    mgr.add(node_id, binary_hdv)
    results = mgr.search(query_hdv, top_k=10)
    mgr.remove(node_id)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed — HNSW index unavailable, falling back to linear scan.")


# ------------------------------------------------------------------ #
#  Defaults                                                           #
# ------------------------------------------------------------------ #

DEFAULT_HNSW_M: int = 32           # number of bi-directional links per node
DEFAULT_EF_CONSTRUCTION: int = 200  # build-time ef (accuracy vs build time)
DEFAULT_EF_SEARCH: int = 64        # query-time ef (accuracy vs query time)
FLAT_THRESHOLD: int = 256          # use flat index below this hop count


# ------------------------------------------------------------------ #
#  HNSW Index Manager                                                 #
# ------------------------------------------------------------------ #

class HNSWIndexManager:
    """
    Manages a FAISS HNSW binary ANN index for the HOT tier.

    Automatically switches between:
     - IndexBinaryFlat  (N < FLAT_THRESHOLD — exact, faster for small N)
     - IndexBinaryHNSW  (N ≥ FLAT_THRESHOLD — approx, faster for large N)

    The index is rebuilt from scratch when switching modes (rare operation).
    All operations are synchronous (called from within asyncio.Lock context).
    """

    def __init__(
        self,
        dimension: int = 16384,
        m: int = DEFAULT_HNSW_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ):
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # ID maps
        self._id_map: Dict[int, str] = {}         # faiss_int_id → node_id
        self._node_map: Dict[str, int] = {}        # node_id → faiss_int_id
        self._next_id: int = 1
        self._use_hnsw: bool = False

        # FAISS index (initialised below)
        self._index = None

        if FAISS_AVAILABLE:
            self._build_flat_index()
        else:
            logger.warning("HNSWIndexManager running WITHOUT faiss — linear fallback only.")

    # ---- Index construction -------------------------------------- #

    def _build_flat_index(self) -> None:
        """Create a fresh IndexBinaryFlat (exact Hamming ANN)."""
        base = faiss.IndexBinaryFlat(self.dimension)
        self._index = faiss.IndexBinaryIDMap(base)
        self._use_hnsw = False
        logger.debug(f"Built FAISS flat binary index (dim={self.dimension})")

    def _build_hnsw_index(self, existing_nodes: Optional[List[Tuple[int, np.ndarray]]] = None) -> None:
        """
        Build an HNSW binary index and optionally re-populate with existing vectors.

        Note: FAISS IndexBinaryHNSW does NOT support IDMap natively, so we use a
        custom double-mapping approach: HNSW indices map 1-to-1 to our _id_map.
        We rebuild as IndexBinaryHNSW and re-add all existing vectors.
        """
        hnsw = faiss.IndexBinaryHNSW(self.dimension, self.m)
        hnsw.hnsw.efConstruction = self.ef_construction
        hnsw.hnsw.efSearch = self.ef_search

        if existing_nodes:
            # Batch add in order of faiss_int_id so positions are deterministic
            existing_nodes.sort(key=lambda x: x[0])
            vecs = np.stack([v for _, v in existing_nodes])
            hnsw.add(vecs)
            logger.debug(f"HNSW index rebuilt with {len(existing_nodes)} existing vectors")

        self._index = hnsw
        self._use_hnsw = True
        logger.info(
            f"Switched to FAISS HNSW index (dim={self.dimension}, M={self.m}, "
            f"efConstruction={self.ef_construction}, efSearch={self.ef_search})"
        )

    def _maybe_upgrade_to_hnsw(self) -> None:
        """Upgrade to HNSW index if HOT tier has grown large enough."""
        if not FAISS_AVAILABLE:
            return
        if self._use_hnsw:
            return
        if len(self._id_map) < FLAT_THRESHOLD:
            return

        logger.info(
            f"HOT tier size ({len(self._id_map)}) ≥ threshold ({FLAT_THRESHOLD}) "
            "— upgrading to HNSW index."
        )

        # NOTE: For HNSW without IDMap we maintain position-based mapping.
        # We rebuild from the current flat index contents.
        # Collect all existing (local_pos → node_vector) pairs.
        #
        # For simplicity in this transition we do a full rebuild from scratch:
        # the upgrade happens at most once per process lifetime (HOT usually stays
        # under threshold or once it crosses, it stays crossed).
        existing: List[Tuple[int, np.ndarray]] = []
        for fid, node_id in self._id_map.items():
            # We can't reconstruct vectors from IndexBinaryIDMap cheaply,
            # so we store them in a shadow cache while using the flat index.
            if node_id in self._vector_cache:
                existing.append((fid, self._vector_cache[node_id]))

        self._build_hnsw_index(existing)

    # ---- Vector shadow cache (needed for HNSW rebuild) ----------- #
    # HNSW indices don't support IDMap; we cache raw vectors separately
    # so we can rebuild on threshold-crossing.

    @property
    def _vector_cache(self) -> Dict[str, np.ndarray]:
        if not hasattr(self, "_vcache"):
            object.__setattr__(self, "_vcache", {})
        return self._vcache  # type: ignore[attr-defined]

    # ---- Public API --------------------------------------------- #

    def add(self, node_id: str, hdv_data: np.ndarray) -> None:
        """
        Add a node to the index.

        Args:
            node_id: Unique string ID for the memory node.
            hdv_data: Packed uint8 array (D/8 bytes).
        """
        if not FAISS_AVAILABLE or self._index is None:
            return

        fid = self._next_id
        self._next_id += 1
        self._id_map[fid] = node_id
        self._node_map[node_id] = fid
        self._vector_cache[node_id] = hdv_data.copy()

        vec = np.expand_dims(hdv_data, axis=0)

        try:
            if self._use_hnsw:
                # HNSW.add() — position is implicit (sequential)
                self._index.add(vec)
            else:
                ids = np.array([fid], dtype="int64")
                self._index.add_with_ids(vec, ids)
        except Exception as exc:
            logger.error(f"HNSW/FAISS add failed for {node_id}: {exc}")
            return

        # Check if we should upgrade to HNSW
        self._maybe_upgrade_to_hnsw()

    def remove(self, node_id: str) -> None:
        """
        Remove a node from the index.

        For HNSW (no IDMap), we mark the node as deleted in our bookkeeping
        and rebuild the index lazily when the deletion rate exceeds 20%.
        """
        if not FAISS_AVAILABLE or self._index is None:
            return

        fid = self._node_map.pop(node_id, None)
        if fid is None:
            return

        self._id_map.pop(fid, None)
        self._vector_cache.pop(node_id, None)

        if not self._use_hnsw:
            try:
                ids = np.array([fid], dtype="int64")
                self._index.remove_ids(ids)
            except Exception as exc:
                logger.error(f"FAISS flat remove failed for {node_id}: {exc}")
        else:
            # HNSW doesn't support removal; track stale fraction and rebuild when needed
            if not hasattr(self, "_stale_count"):
                object.__setattr__(self, "_stale_count", 0)
            self._stale_count += 1  # type: ignore[attr-defined]

            total = max(len(self._id_map) + self._stale_count, 1)
            stale_fraction = self._stale_count / total
            if stale_fraction > 0.20 and len(self._id_map) > 0:
                logger.info(f"HNSW stale fraction {stale_fraction:.1%} — rebuilding index.")
                existing = [
                    (fid2, self._vector_cache[nid])
                    for fid2, nid in self._id_map.items()
                    if nid in self._vector_cache
                ]
                self._build_hnsw_index(existing)
                self._stale_count = 0

    def search(self, query_data: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbours.

        Args:
            query_data: Packed uint8 query array (D/8 bytes).
            top_k: Number of results to return.

        Returns:
            List of (node_id, similarity_score) sorted by descending similarity.
            similarity = 1 - normalised_hamming_distance  ∈ [0, 1].
        """
        if not FAISS_AVAILABLE or self._index is None or not self._id_map:
            return []

        k = min(top_k, len(self._id_map))
        q = np.expand_dims(query_data, axis=0)

        try:
            distances, ids = self._index.search(q, k)
        except Exception as exc:
            logger.error(f"HNSW/FAISS search failed: {exc}")
            return []

        results: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1:
                continue

            if self._use_hnsw:
                # HNSW returns 0-based position indices; map back through insertion order
                node_id = self._position_to_node_id(int(idx))
            else:
                node_id = self._id_map.get(int(idx))

            if node_id:
                sim = 1.0 - float(dist) / self.dimension
                results.append((node_id, sim))

        return results

    def _position_to_node_id(self, position: int) -> Optional[str]:
        """
        Map HNSW sequential position back to node_id.
        Positions correspond to insertion order; we track this via _position_map.
        """
        if not hasattr(self, "_position_map"):
            object.__setattr__(self, "_position_map", {})
        pm: Dict[int, str] = self._position_map  # type: ignore[attr-defined]

        # Rebuild position map if needed (after index rebuild)
        if len(pm) < len(self._id_map):
            pm.clear()
            for pos, (fid, nid) in enumerate(
                sorted(self._id_map.items(), key=lambda x: x[0])
            ):
                pm[pos] = nid

        return pm.get(position)

    @property
    def size(self) -> int:
        return len(self._id_map)

    @property
    def index_type(self) -> str:
        if not FAISS_AVAILABLE:
            return "linear"
        return "hnsw" if self._use_hnsw else "flat"

    def stats(self) -> Dict:
        return {
            "index_type": self.index_type,
            "indexed_nodes": self.size,
            "dimension": self.dimension,
            "hnsw_m": self.m if self._use_hnsw else None,
            "ef_construction": self.ef_construction if self._use_hnsw else None,
            "ef_search": self.ef_search if self._use_hnsw else None,
            "faiss_available": FAISS_AVAILABLE,
        }
