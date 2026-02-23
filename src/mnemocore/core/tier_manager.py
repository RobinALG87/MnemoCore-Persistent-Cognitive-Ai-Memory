"""
Tiered Memory Management (Phase 6 Refactor)
============================================
Facade for tiered memory operations across HOT, WARM, and COLD tiers.

This module provides a unified interface to the tiered storage system,
coordinating between tier_storage, tier_eviction, and tier_scoring modules.

Architecture:
  - HOT (RAM): Fast access, limited capacity. Stores most relevant memories.
  - WARM (Qdrant/Disk): Larger capacity, slightly slower access.
  - COLD (Archive): Unlimited capacity, slow access. Compressed JSONL.

The TierManager facade maintains backward compatibility while delegating
to specialized modules for storage, eviction, and scoring.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.config import HAIMConfig, get_config
from mnemocore.core.exceptions import (
    MnemoCoreError,
    StorageError,
    CircuitOpenError,
)

# Phase 6 Refactor: Import specialized modules
from mnemocore.core.tier_storage import (
    HotTierStorage,
    WarmTierStorage,
    ColdTierStorage,
)
from mnemocore.core.tier_eviction import TierEvictionManager
from mnemocore.core.tier_scoring import (
    TierScoringManager,
    SearchScorer,
)

if TYPE_CHECKING:
    from mnemocore.core.node import MemoryNode
    from mnemocore.core.qdrant_store import QdrantStore

# HNSW Index Manager (Phase 4.0)
try:
    from mnemocore.core.hnsw_index import HNSWIndexManager
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

# Legacy FAISS support
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TierManager:
    """
    Facade for tiered memory management.

    Coordinates storage operations, eviction policies, and scoring
    across HOT, WARM, and COLD tiers.

    Maintains full backward compatibility with the original TierManager
    interface while delegating to specialized modules internally.
    """

    def __init__(
        self,
        config: Optional[HAIMConfig] = None,
        qdrant_store: Optional["QdrantStore"] = None,
    ):
        """
        Initialize TierManager with optional dependency injection.

        Args:
            config: Configuration object. If None, uses global get_config().
            qdrant_store: QdrantStore instance. If None, will not use Qdrant.
        """
        self.config = config or get_config()
        self._initialized: bool = False
        self.lock: asyncio.Lock = asyncio.Lock()

        # Initialize specialized modules (Phase 6 Refactor)
        self._hot_storage = HotTierStorage(
            max_memories=self.config.tiers_hot.max_memories
        )
        self._warm_storage = WarmTierStorage(self.config, qdrant_store)
        self._cold_storage = ColdTierStorage(self.config)
        self._eviction_manager = TierEvictionManager(self.config)
        self._scoring_manager = TierScoringManager(self.config)
        self._search_scorer = SearchScorer(self.config)

        # Qdrant reference (for backward compatibility)
        self.qdrant = qdrant_store
        self.use_qdrant = qdrant_store is not None
        self.warm_path = self._warm_storage.warm_path

        # Paths (for backward compatibility)
        from pathlib import Path
        self.cold_path = Path(self.config.paths.cold_archive_dir)
        if not self.use_qdrant:
            self.warm_path = Path(self.config.paths.warm_mmap_dir)

        # Phase 4.0: HNSW/FAISS Index for HOT Tier
        self._hnsw = None
        self._init_hnsw_index()

        # Legacy FAISS fields (for backward compatibility)
        self.faiss_index = None
        self.faiss_id_map: Dict[int, str] = {}
        self.node_id_to_faiss_id: Dict[str, int] = {}
        self._next_faiss_id = 1

        if not HNSW_AVAILABLE and FAISS_AVAILABLE:
            self._init_faiss()

    def _init_hnsw_index(self):
        """Initialize HNSW index for HOT tier."""
        if HNSW_AVAILABLE:
            cfg = self.config
            self._hnsw = HNSWIndexManager(
                dimension=cfg.dimensionality,
                m=getattr(cfg.qdrant, "hnsw_m", 32),
                ef_construction=getattr(cfg.qdrant, "hnsw_ef_construct", 200),
                ef_search=64,
            )
            logger.info(
                f"HNSWIndexManager initialized for HOT tier "
                f"(dim={cfg.dimensionality}, M={getattr(cfg.qdrant, 'hnsw_m', 32)})"
            )

    def _init_faiss(self):
        """Initialize FAISS binary index (legacy fallback)."""
        dim = self.config.dimensionality
        base_index = faiss.IndexBinaryFlat(dim)
        self.faiss_index = faiss.IndexBinaryIDMap(base_index)
        logger.info(f"Initialized FAISS flat binary index for HOT tier (dim={dim})")

    # =========================================================================
    # Public API - Memory Operations
    # =========================================================================

    async def initialize(self):
        """Async initialization for Qdrant collections."""
        if self._initialized:
            return

        if self.use_qdrant and self.qdrant:
            try:
                await self.qdrant.ensure_collections()
            except Exception as e:
                logger.error(f"Failed to ensure Qdrant collections: {e}")
                self.use_qdrant = False
                self.warm_path = self.cold_path  # Fallback path

        self._initialized = True

    async def add_memory(self, node: "MemoryNode"):
        """
        Add a new memory node. New memories are always HOT initially.

        Handles eviction if HOT tier is at capacity.
        """
        node.tier = "hot"

        # Ensure mutual exclusion - delete from WARM if exists
        await self._warm_storage.delete(node.id)

        # Add to HOT tier
        victim_to_evict = None
        async with self.lock:
            await self._hot_storage.save(node)
            self._add_to_faiss(node)

            # Check eviction
            hot_count = await self._hot_storage.count()
            if hot_count > self.config.tiers_hot.max_memories:
                candidates = list(self._hot_storage.get_storage_dict().values())
                victim_to_evict = self._eviction_manager.select_hot_victim(
                    candidates, exclude_ids=[node.id]
                )

        # Perform eviction outside lock
        if victim_to_evict:
            try:
                save_ok = await self._warm_storage.save(victim_to_evict)

                if save_ok:
                    async with self.lock:
                        # Double-check still qualifies
                        if victim_to_evict.tier == "warm":
                            await self._hot_storage.delete(victim_to_evict.id)
                            self._remove_from_faiss(victim_to_evict.id)
                            logger.debug(f"Confirmed eviction of {victim_to_evict.id} to WARM")
                else:
                    logger.error(f"Failed to evict {victim_to_evict.id} to WARM. Node remains in HOT.")
                    victim_to_evict.tier = "hot"
            except Exception as e:
                logger.error(f"Critical error during eviction of {victim_to_evict.id}: {e}")
                victim_to_evict.tier = "hot"

    async def get_memory(self, node_id: str) -> Optional["MemoryNode"]:
        """
        Retrieve memory by ID from any tier.

        Triggers promotion/demotion based on LTP strength and hysteresis.
        """
        # Check HOT
        demote_candidate = None
        result_node = None

        async with self.lock:
            node = await self._hot_storage.get(node_id)
            if node:
                node.access()

                if self._eviction_manager.should_demote_to_warm(node):
                    node.tier = "warm"
                    demote_candidate = node

                result_node = node

        # Handle demotion
        if demote_candidate:
            try:
                logger.info(f"Demoting {demote_candidate.id} to WARM (LTP: {demote_candidate.ltp_strength:.4f})")

                save_ok = await self._warm_storage.save(demote_candidate)

                if save_ok:
                    async with self.lock:
                        hot_node = await self._hot_storage.get(demote_candidate.id)
                        if hot_node and hot_node.tier == "warm":
                            await self._hot_storage.delete(demote_candidate.id)
                            self._remove_from_faiss(demote_candidate.id)
                else:
                    logger.error(f"Demotion of {demote_candidate.id} failed. Node remains in HOT.")
                    demote_candidate.tier = "hot"
            except Exception as e:
                logger.error(f"Demotion of {demote_candidate.id} failed: {e}")
                demote_candidate.tier = "hot"

        if result_node:
            return result_node

        # Check WARM
        warm_node = await self._warm_storage.get(node_id)
        if warm_node:
            warm_node.tier = "warm"
            warm_node.access()

            if self._eviction_manager.should_promote_to_hot(warm_node):
                await self._promote_to_hot(warm_node)

            return warm_node

        # Check COLD
        cold_node = await self._cold_storage.get(node_id)
        if cold_node:
            logger.debug(f"Retrieved {node_id} from COLD tier.")

        return cold_node

    async def get_memories_batch(self, node_ids: List[str]) -> List[Optional["MemoryNode"]]:
        """
        Retrieve multiple memories concurrently.

        Preserves input order and returns None for missing/error cases.
        """
        if not node_ids:
            return []

        unique_ids = list(dict.fromkeys(node_ids))
        tasks = [self.get_memory(nid) for nid in unique_ids]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        result_by_id: Dict[str, Optional["MemoryNode"]] = {}
        for nid, result in zip(unique_ids, raw_results):
            if isinstance(result, Exception):
                logger.error(f"Batch get_memory failed for {nid}: {result}")
                result_by_id[nid] = None
            else:
                result_by_id[nid] = result

        return [result_by_id.get(nid) for nid in node_ids]

    async def anticipate(self, node_ids: List[str]) -> None:
        """
        Phase 13.2: Anticipatory Memory
        Pre-loads specific nodes into the HOT active tier (working memory).
        """
        for nid in set(node_ids):
            in_hot = await self._hot_storage.contains(nid)

            if not in_hot:
                node = await self._warm_storage.get(nid)
                if node:
                    await self._promote_to_hot(node)

    async def delete_memory(self, node_id: str):
        """Robust delete from all tiers."""
        async with self.lock:
            if await self._hot_storage.contains(node_id):
                await self._hot_storage.delete(node_id)
                self._remove_from_faiss(node_id)
                logger.debug(f"Deleted {node_id} from HOT")

        await self._warm_storage.delete(node_id)
        # COLD is append-only, no delete needed

    # =========================================================================
    # Public API - Search
    # =========================================================================

    async def search(
        self,
        query_vec: BinaryHDV,
        top_k: int = 5,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_cold: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Global search across all tiers.
        """
        # Search HOT via FAISS/HNSW
        hot_results = self.search_hot(query_vec, top_k)

        # Apply filters to HOT results
        if time_range or metadata_filter:
            hot_results = await self._apply_filters_to_results(
                hot_results, time_range, metadata_filter
            )

        # Search WARM via Qdrant
        warm_results = []
        if self.use_qdrant:
            try:
                q_vec = np.unpackbits(query_vec.data).astype(float)

                hits = await self.qdrant.search(
                    collection=self.config.qdrant.collection_warm,
                    query_vector=q_vec,
                    limit=top_k,
                    time_range=time_range,
                    metadata_filter=metadata_filter,
                )
                warm_results = [(hit.id, hit.score) for hit in hits]
            except Exception as e:
                logger.error(f"WARM tier search failed: {e}")

        # Search COLD if requested
        cold_results: List[Tuple[str, float]] = []
        if include_cold:
            cold_results = await self.search_cold(query_vec, top_k)

        # Merge and rank
        merged = self._search_scorer.merge_and_rank(
            hot_results, warm_results, cold_results, top_k
        )

        return [(r.node_id, r.similarity) for r in merged]

    async def _apply_filters_to_results(
        self,
        results: List[Tuple[str, float]],
        time_range: Optional[Tuple[datetime, datetime]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """Apply time and metadata filters to search results."""
        filtered = []
        async with self.lock:
            storage = self._hot_storage.get_storage_dict()
            for nid, score in results:
                node = storage.get(nid)
                if not node:
                    continue

                if time_range:
                    start_ts = time_range[0].timestamp()
                    end_ts = time_range[1].timestamp()
                    if not (start_ts <= node.created_at.timestamp() <= end_ts):
                        continue

                if metadata_filter:
                    node_meta = node.metadata or {}
                    match = all(node_meta.get(k) == v for k, v in metadata_filter.items())
                    if not match:
                        continue

                filtered.append((nid, score))
        return filtered

    def search_hot(self, query_vec: BinaryHDV, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search HOT tier using HNSW or FAISS binary index."""
        if self._hnsw is not None and self._hnsw.size > 0:
            try:
                return self._hnsw.search(query_vec.data, top_k)
            except Exception as e:
                logger.error(f"HNSW search failed: {e}")

        if not self.faiss_index:
            return self._linear_search_hot(query_vec, top_k)

        try:
            q = np.expand_dims(query_vec.data, axis=0)
            distances, ids = self.faiss_index.search(q, top_k)

            results = []
            for d, fid in zip(distances[0], ids[0]):
                if fid == -1:
                    continue
                node_id = self.faiss_id_map.get(int(fid))
                if node_id:
                    sim = 1.0 - (float(d) / self.config.dimensionality)
                    results.append((node_id, sim))
            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return self._linear_search_hot(query_vec, top_k)

    def _linear_search_hot(self, query_vec: BinaryHDV, top_k: int = 5) -> List[Tuple[str, float]]:
        """Fallback linear scan for HOT tier."""
        import asyncio

        async def _scan():
            scores = []
            storage = self._hot_storage.get_storage_dict()
            for node in storage.values():
                sim = query_vec.similarity(node.hdv)
                scores.append((node.id, sim))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        # Run synchronously since we're called from sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in async context, but this is a sync method
                # Use the storage dict directly
                scores = []
                storage = self._hot_storage.get_storage_dict()
                for node in storage.values():
                    sim = query_vec.similarity(node.hdv)
                    scores.append((node.id, sim))
                scores.sort(key=lambda x: x[1], reverse=True)
                return scores[:top_k]
        except RuntimeError:
            pass

        # Fallback
        scores = []
        storage = self._hot_storage.get_storage_dict()
        for node in storage.values():
            sim = query_vec.similarity(node.hdv)
            scores.append((node.id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def search_cold(
        self, query_vec: BinaryHDV, top_k: int = 5, max_scan: int = 1000
    ) -> List[Tuple[str, float]]:
        """Linear similarity scan over COLD archive."""
        return await self._cold_storage.search(query_vec, top_k, max_scan)

    # =========================================================================
    # Public API - Statistics and Listing
    # =========================================================================

    async def get_stats(self) -> Dict:
        """Get statistics about memory distribution across tiers."""
        stats = {
            "hot_count": await self._hot_storage.count(),
            "warm_count": await self._warm_storage.count(),
            "cold_count": await self._cold_storage.count(),
            "using_qdrant": self.use_qdrant,
            "ann_index": self._hnsw.stats() if self._hnsw else {"index_type": "none"},
        }
        return stats

    async def get_hot_snapshot(self) -> List["MemoryNode"]:
        """Return a snapshot of values in HOT tier safely."""
        return await self._hot_storage.get_snapshot()

    async def get_hot_recent(self, n: int) -> List["MemoryNode"]:
        """Get the most recent n memories from HOT tier efficiently."""
        return await self._hot_storage.get_recent(n)

    async def list_warm(self, max_results: int = 500) -> List["MemoryNode"]:
        """List nodes from the WARM tier."""
        return await self._warm_storage.list_all(max_results, with_vectors=True)

    async def get_next_in_chain(self, node_id: str) -> Optional["MemoryNode"]:
        """
        Return the MemoryNode that directly follows node_id in the episodic chain.
        """
        # Check HOT tier via O(1) inverted index
        next_id = self._hot_storage.get_next_in_chain_id(node_id)
        if next_id:
            node = await self._hot_storage.get(next_id)
            if node:
                return node

        # Check WARM tier (Qdrant)
        if not self.use_qdrant or not self.qdrant:
            return None

        record = await self.qdrant.get_by_previous_id(
            self.qdrant.collection_warm, node_id
        )
        if record is None:
            return None

        return await self._warm_storage.get(str(record.id))

    async def consolidate_warm_to_cold(self):
        """Batch move from WARM to COLD based on archive criteria."""
        nodes = await self._warm_storage.list_all(limit=1000, with_vectors=True)
        candidates = self._eviction_manager.get_warm_candidates_for_archive(nodes)

        for node in candidates:
            # Save to COLD
            await self._cold_storage.save(node)
            # Delete from WARM
            await self._warm_storage.delete(node.id)

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    async def _promote_to_hot(self, node: "MemoryNode"):
        """Promote node from WARM to HOT."""
        # Step 1: Delete from WARM (I/O)
        deleted = await self._warm_storage.delete(node.id)
        if not deleted:
            logger.debug(f"Skipping promotion of {node.id}: not found in WARM")
            return

        # Step 2: Add to HOT
        victim_to_save = None
        async with self.lock:
            if await self._hot_storage.contains(node.id):
                logger.debug(f"{node.id} already in HOT, skipping duplicate promotion")
                return

            logger.info(f"Promoting {node.id} to HOT (LTP: {node.ltp_strength:.4f})")
            node.tier = "hot"
            await self._hot_storage.save(node)
            self._add_to_faiss(node)

            # Check eviction
            hot_count = await self._hot_storage.count()
            if hot_count > self.config.tiers_hot.max_memories:
                candidates = list(self._hot_storage.get_storage_dict().values())
                victim_to_save = self._eviction_manager.select_hot_victim(candidates)

        # Step 3: Evict if needed
        if victim_to_save:
            await self._warm_storage.save(victim_to_save)

    def _add_to_faiss(self, node: "MemoryNode"):
        """Add node to the ANN index (HNSW preferred, legacy flat as fallback)."""
        if self._hnsw is not None:
            self._hnsw.add(node.id, node.hdv.data)
            return

        if not self.faiss_index:
            return

        try:
            fid = self._next_faiss_id
            self._next_faiss_id += 1

            vec = np.expand_dims(node.hdv.data, axis=0)
            ids = np.array([fid], dtype='int64')
            self.faiss_index.add_with_ids(vec, ids)

            self.faiss_id_map[fid] = node.id
            self.node_id_to_faiss_id[node.id] = fid
        except Exception as e:
            logger.error(f"Failed to add node {node.id} to FAISS: {e}")

    def _remove_from_faiss(self, node_id: str):
        """Remove node from the ANN index."""
        if self._hnsw is not None:
            self._hnsw.remove(node_id)
            return

        if not self.faiss_index:
            return

        fid = self.node_id_to_faiss_id.get(node_id)
        if fid is not None:
            try:
                if hasattr(self.faiss_index, "remove_ids"):
                    ids_to_remove = np.array([fid], dtype='int64')
                    self.faiss_index.remove_ids(ids_to_remove)

                del self.faiss_id_map[fid]
                del self.node_id_to_faiss_id[node_id]
            except Exception as e:
                logger.warning(f"Failed to remove node {node_id} from index: {e}")

    # =========================================================================
    # Backward Compatibility - Legacy Properties
    # =========================================================================

    @property
    def hot(self) -> Dict[str, "MemoryNode"]:
        """Legacy access to HOT tier storage dict."""
        return self._hot_storage.get_storage_dict()

    @property
    def _next_chain(self) -> Dict[str, str]:
        """Legacy access to episodic chain index."""
        return self._hot_storage.get_next_chain_dict()


__all__ = ["TierManager"]
