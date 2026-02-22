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

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

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

import json
from pathlib import Path
from threading import Lock
from .config import get_config

class HNSWIndexManager:
    """
    Manages a FAISS HNSW binary ANN index for the HOT tier.
    Thread-safe singleton with disk persistence.

    Automatically switches between:
     - IndexBinaryFlat  (N < FLAT_THRESHOLD — exact, faster for small N)
     - IndexBinaryHNSW  (N ≥ FLAT_THRESHOLD — approx, faster for large N)
    """

    _instance: "HNSWIndexManager | None" = None
    _singleton_lock: Lock = Lock()

    def __new__(cls, *args, **kwargs) -> "HNSWIndexManager":
        with cls._singleton_lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._initialized = False
                cls._instance = obj
        return cls._instance

    def __init__(
        self,
        dimension: int = 16384,
        m: int = DEFAULT_HNSW_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ):
        if getattr(self, "_initialized", False):
            return

        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self._write_lock = Lock()

        self._id_map: List[Optional[str]] = []
        self._vector_store: List[np.ndarray] = []
        self._use_hnsw = False
        self._stale_count = 0
        self._index = None
        
        config = get_config()
        data_dir = Path(config.paths.data_dir if hasattr(config, 'paths') else "./data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        self.INDEX_PATH = data_dir / "mnemocore_hnsw.faiss"
        self.IDMAP_PATH = data_dir / "mnemocore_hnsw_idmap.json"
        self.VECTOR_PATH = data_dir / "mnemocore_hnsw_vectors.npy"

        if FAISS_AVAILABLE:
            self._build_flat_index()

        self._initialized = True

    # ---- Index construction -------------------------------------- #

    def _build_flat_index(self) -> None:
        """Create a fresh IndexBinaryFlat (exact Hamming ANN)."""
        self._index = faiss.IndexBinaryFlat(self.dimension)
        self._use_hnsw = False
        logger.debug(f"Built FAISS flat binary index (dim={self.dimension})")

    def _build_hnsw_index(self) -> None:
        """
        Build an HNSW binary index and optionally re-populate with existing vectors.
        """
        hnsw = faiss.IndexBinaryHNSW(self.dimension, self.m)
        hnsw.hnsw.efConstruction = self.ef_construction
        hnsw.hnsw.efSearch = self.ef_search

        if self._vector_store:
            # Compact the index to remove None entries
            compact_ids = []
            compact_vecs = []
            for i, node_id in enumerate(self._id_map):
                if node_id is not None:
                    compact_ids.append(node_id)
                    compact_vecs.append(self._vector_store[i])
            
            if compact_vecs:
                vecs = np.stack(compact_vecs)
                hnsw.add(vecs)
                
            self._id_map = compact_ids
            self._vector_store = compact_vecs
            self._stale_count = 0

        self._index = hnsw
        self._use_hnsw = True
        logger.info(
            f"Switched to FAISS HNSW index (dim={self.dimension}, M={self.m}, "
            f"efConstruction={self.ef_construction}, efSearch={self.ef_search})"
        )

    def _maybe_upgrade_to_hnsw(self) -> None:
        """Upgrade to HNSW index if HOT tier has grown large enough."""
        if not FAISS_AVAILABLE or self._use_hnsw:
            return
        active_count = len(self._id_map) - self._stale_count
        if active_count < FLAT_THRESHOLD:
            return

        logger.info(
            f"HOT tier size ({active_count}) ≥ threshold ({FLAT_THRESHOLD}) "
            "— upgrading to HNSW index."
        )

        self._build_hnsw_index()

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

        vec = np.ascontiguousarray(np.expand_dims(hdv_data, axis=0))

        with self._write_lock:
            try:
                self._index.add(vec)
                self._id_map.append(node_id)
                self._vector_store.append(hdv_data.copy())
            except Exception as exc:
                logger.error(f"HNSW/FAISS add failed for {node_id}: {repr(exc)}")
                return

            self._maybe_upgrade_to_hnsw()

    def remove(self, node_id: str) -> None:
        """
        Remove a node from the index.
        Marks node as deleted and rebuilds index lazily when the deletion rate exceeds 20%.
        """
        if not FAISS_AVAILABLE or self._index is None:
            return

        with self._write_lock:
            try:
                fid = self._id_map.index(node_id)
                self._id_map[fid] = None
                self._stale_count += 1
                
                total = max(len(self._id_map), 1)
                stale_fraction = self._stale_count / total
                
                if stale_fraction > 0.20 and len(self._id_map) > 0:
                    logger.info(f"HNSW stale fraction {stale_fraction:.1%} — rebuilding index.")
                    if self._use_hnsw:
                        self._build_hnsw_index()
                    else:
                        self._build_flat_index()
                        if self._vector_store:
                            compact_ids = []
                            compact_vecs = []
                            for i, nid in enumerate(self._id_map):
                                if nid is not None:
                                    compact_ids.append(nid)
                                    compact_vecs.append(self._vector_store[i])
                            if compact_vecs:
                                vecs = np.ascontiguousarray(np.stack(compact_vecs))
                                self._index.add(vecs)
                            self._id_map = compact_ids
                            self._vector_store = compact_vecs
                            self._stale_count = 0
                
            except ValueError:
                pass


    def search(self, query_data: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbours.

        Returns:
            List of (node_id, similarity_score) sorted by descending similarity.
            similarity = 1 - normalised_hamming_distance  ∈ [0, 1].
        """
        if not FAISS_AVAILABLE or self._index is None or not self._id_map:
            return []

        # Fetch more to account for deleted (None) entries
        k = min(top_k + self._stale_count + 50, len(self._id_map))
        if k <= 0:
            return []

        index_dimension = int(getattr(self._index, "d", self.dimension) or self.dimension)
        query_bytes = np.ascontiguousarray(query_data, dtype=np.uint8).reshape(-1)
        expected_bytes = index_dimension // 8
        if expected_bytes > 0 and query_bytes.size != expected_bytes:
            logger.warning(
                f"HNSW query dimension mismatch: index={index_dimension} bits ({expected_bytes} bytes), "
                f"query={query_bytes.size} bytes. Adjusting query to index dimension."
            )
            if query_bytes.size > expected_bytes:
                query_bytes = query_bytes[:expected_bytes]
            else:
                query_bytes = np.pad(query_bytes, (0, expected_bytes - query_bytes.size), mode="constant")

        q = np.expand_dims(query_bytes, axis=0)

        try:
            distances, ids = self._index.search(q, k)
        except Exception as exc:
            logger.error(f"HNSW/FAISS search failed: {exc}")
            return []

        results: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
                
            node_id = self._id_map[idx]
            if node_id is not None:
                sim = 1.0 - float(dist) / max(index_dimension, 1)
                sim = float(np.clip(sim, 0.0, 1.0))
                results.append((node_id, sim))
                if len(results) >= top_k:
                    break

        return results

    def _save(self):
        try:
            faiss.write_index_binary(self._index, str(self.INDEX_PATH))
            with open(self.IDMAP_PATH, "w") as f:
                json.dump({
                    "id_map": self._id_map,
                    "use_hnsw": self._use_hnsw,
                    "stale_count": self._stale_count
                }, f)
            if self._vector_store:
                np.save(str(self.VECTOR_PATH), np.stack(self._vector_store))
        except Exception as e:
            logger.error(f"Failed to save HNSW index state: {e}")

    def _load(self):
        try:
            self._index = faiss.read_index_binary(str(self.INDEX_PATH))
            index_dimension = int(getattr(self._index, "d", self.dimension) or self.dimension)
            if index_dimension != self.dimension:
                logger.warning(
                    f"HNSW index dimension mismatch on load: config={self.dimension}, index={index_dimension}. "
                    "Using index dimension."
                )
                self.dimension = index_dimension
            with open(self.IDMAP_PATH, "r") as f:
                state = json.load(f)
                self._id_map = state.get("id_map", [])
                self._use_hnsw = state.get("use_hnsw", False)
                self._stale_count = state.get("stale_count", 0)
                
            vecs = np.load(str(self.VECTOR_PATH))
            self._vector_store = list(vecs)
            logger.info("Loaded HNSW persistent state from disk")
        except Exception as e:
            logger.error(f"Failed to load HNSW index state: {e}")
            self._build_flat_index()

    @property
    def size(self) -> int:
        return len([x for x in self._id_map if x is not None])

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
            "stale_count": self._stale_count
        }
