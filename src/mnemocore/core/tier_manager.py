"""
Tiered Memory Management (Phase 3.5+)
=====================================
Manages memory lifecycle across HOT, WARM, and COLD tiers based on Long-Term Potentiation (LTP).

Tiers:
  - HOT (RAM): Fast access, limited capacity. Stores most relevant memories.
  - WARM (Qdrant/Disk): Larger capacity, slightly slower access.
  - COLD (Archive): Unlimited capacity, slow access. Compressed JSONL.

Logic:
  - New memories start in HOT.
  - `consolidate()` moves memories between tiers based on LTP strength and hysteresis.
  - Promote: WARM -> HOT if `ltp > threshold + delta`
  - Demote: HOT -> WARM if `ltp < threshold - delta`
  - Archive: WARM -> COLD if `ltp < archive_threshold` (or age)

All vectors use BinaryHDV (packed uint8 arrays).
"""

import gzip
from mnemocore.utils import json_compat as json
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .qdrant_store import QdrantStore
import asyncio
import functools

import numpy as np
from loguru import logger

from .binary_hdv import BinaryHDV
from .config import HAIMConfig, get_config
from .node import MemoryNode
from .exceptions import (
    MnemoCoreError,
    StorageError,
    CircuitOpenError,
    DataCorruptionError,
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Phase 4.0: HNSW index manager (replaces raw FAISS management)
try:
    from .hnsw_index import HNSWIndexManager
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False


class TierManager:
    """
    Manages memory storage across tiered hierarchy.
    Uses BinaryHDV exclusively for efficient storage and computation.
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

        # Initialization guard
        self._initialized: bool = False

        # Async lock - created eagerly; asyncio.Lock() is safe to construct
        # outside a running loop in Python 3.10+ (loop binding is deferred).
        self.lock: asyncio.Lock = asyncio.Lock()

        # HOT Tier: In-memory dictionary
        self.hot: Dict[str, MemoryNode] = {}
        # Phase 13.2: O(1) inverted index for episodic chain – previous_id → node_id.
        # Maintained in sync with self.hot so get_next_in_chain() avoids O(N) scans.
        self._next_chain: Dict[str, str] = {}

        # WARM Tier: Qdrant (injected) or fallback to filesystem
        self.qdrant = qdrant_store
        self.use_qdrant = qdrant_store is not None
        self.warm_path = None

        if not self.use_qdrant:
            self.warm_path = Path(self.config.paths.warm_mmap_dir)
            self.warm_path.mkdir(parents=True, exist_ok=True)

        # COLD Tier path
        self.cold_path = Path(self.config.paths.cold_archive_dir)
        self.cold_path.mkdir(parents=True, exist_ok=True)

        # Phase 4.0: HNSW/FAISS Index for HOT Tier (Binary)
        # HNSWIndexManager auto-selects Flat (small N) or HNSW (large N)
        cfg = self.config
        if HNSW_AVAILABLE:
            self._hnsw = HNSWIndexManager(
                dimension=cfg.dimensionality,
                m=getattr(cfg.qdrant, "hnsw_m", 32),
                ef_construction=getattr(cfg.qdrant, "hnsw_ef_construct", 200),
                ef_search=64,
            )
            logger.info(
                f"Phase 4.0 HNSWIndexManager initialised for HOT tier "
                f"(dim={cfg.dimensionality}, M={getattr(cfg.qdrant, 'hnsw_m', 32)})"
            )
        else:
            self._hnsw = None

        # Legacy FAISS fields kept for backward-compat (unused when HNSW available)
        self.faiss_index = None
        self.faiss_id_map: Dict[int, str] = {}
        self.node_id_to_faiss_id: Dict[str, int] = {}
        self._next_faiss_id = 1

        if not HNSW_AVAILABLE and FAISS_AVAILABLE:
            self._init_faiss()

    def _init_faiss(self):
        """Initialize FAISS binary index (legacy path, used when hnsw_index unavailable)."""
        dim = self.config.dimensionality
        base_index = faiss.IndexBinaryFlat(dim)
        self.faiss_index = faiss.IndexBinaryIDMap(base_index)
        logger.info(f"Initialized FAISS flat binary index for HOT tier (dim={dim})")

    async def get_hot_snapshot(self) -> List[MemoryNode]:
        """Return a snapshot of values in HOT tier safely."""
        async with self.lock:
            return list(self.hot.values())

    async def get_hot_recent(self, n: int) -> List[MemoryNode]:
        """Get the most recent n memories from HOT tier efficiently."""
        async with self.lock:
            try:
                recent_keys = list(islice(reversed(self.hot), n))
                nodes = [self.hot[k] for k in reversed(recent_keys)]
                return nodes
            except Exception:
                all_nodes = list(self.hot.values())
                return all_nodes[-n:]

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
                self.warm_path = Path(self.config.paths.warm_mmap_dir)
                self.warm_path.mkdir(parents=True, exist_ok=True)

        self._initialized = True

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run blocking function in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    async def add_memory(self, node: MemoryNode):
        """Add a new memory node. New memories are always HOT initially."""
        node.tier = "hot"

        # Delta 67.4: Ensure mutual exclusion. 
        # If the node exists in WARM, remove it before adding to HOT.
        await self._delete_from_warm(node.id)

        # Phase 1: Add to HOT tier under lock (no I/O)
        victim_to_evict = None
        async with self.lock:
            self.hot[node.id] = node
            self._add_to_faiss(node)
            # Maintain inverted chain index (Fix 7: O(1) get_next_in_chain)
            if node.previous_id:
                self._next_chain[node.previous_id] = node.id

            # Check if we need to evict - decide under lock, execute outside
            if len(self.hot) > self.config.tiers_hot.max_memories:
                # Use unified eviction logic, protecting the new node
                victim_to_evict = self._prepare_eviction_from_hot(exclude_node_id=node.id)

        # Phase 2: Perform I/O outside lock
        if victim_to_evict:
            try:
                save_ok = await self._save_to_warm(victim_to_evict)
                
                # Step 3: Remove from HOT only after successful save to WARM (either Qdrant or FS)
                if save_ok:
                    async with self.lock:
                        if victim_to_evict.id in self.hot:
                            # Double check it still qualifies for eviction (wasn't promoted back)
                            if self.hot[victim_to_evict.id].tier == "warm":
                                del self.hot[victim_to_evict.id]
                                self._remove_from_faiss(victim_to_evict.id)
                                logger.debug(f"Confirmed eviction of {victim_to_evict.id} to WARM")
                else:
                    logger.error(f"Failed to evict {victim_to_evict.id} to WARM. Node remains in HOT.")
                    victim_to_evict.tier = "hot"
            except Exception as e:
                logger.error(f"Critical error during eviction of {victim_to_evict.id}: {e}")
                victim_to_evict.tier = "hot"

    def _add_to_faiss(self, node: MemoryNode):
        """Add node to the ANN index (HNSW preferred, legacy flat as fallback)."""
        # Phase 4.0: delegate to HNSWIndexManager
        if self._hnsw is not None:
            self._hnsw.add(node.id, node.hdv.data)
            return

        # Legacy FAISS flat path
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

    async def get_memory(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory by ID from any tier."""
        # Check HOT
        demote_candidate = None
        result_node = None

        async with self.lock:
            if node_id in self.hot:
                node = self.hot[node_id]
                node.access()
                
                # check if node should be demoted
                if self._should_demote(node):
                    # Node will be demoted, mark it as warm immediately to prevent TOCTOU
                    # This ensures concurrent readers see the correct upcoming state
                    node.tier = "warm"
                    demote_candidate = node
                
                result_node = node

        # If demotion is needed, save to WARM first, then remove from HOT
        # This occurs outside the lock to allow concurrency, but the node 
        # is already marked as "warm" (graceful degradation if save fails)
        # If demotion is needed, save to WARM first, then remove from HOT
        if demote_candidate:
            try:
                logger.info(f"Demoting {demote_candidate.id} to WARM (LTP: {demote_candidate.ltp_strength:.4f})")
                
                # Step 1: Save to WARM (I/O outside lock)
                save_ok = await self._save_to_warm(demote_candidate)
                
                # Step 2: Remove from HOT (under lock)
                if save_ok:
                    async with self.lock:
                        # Double-check: it might have been accessed again (promoting it back) or removed
                        if demote_candidate.id in self.hot:
                            # Only delete if it's still marked as warm (wasn't promoted back by an 'access' call)
                            if self.hot[demote_candidate.id].tier == "warm":
                                del self.hot[demote_candidate.id]
                                self._remove_from_faiss(demote_candidate.id)
                                # Maintain inverted chain index
                                if demote_candidate.previous_id:
                                    self._next_chain.pop(demote_candidate.previous_id, None)
                else:
                    logger.error(f"Demotion of {demote_candidate.id} failed. Node remains in HOT.")
                    demote_candidate.tier = "hot"
            except Exception as e:
                import traceback
                print(f"FAILED DEMOTION EXCEPTION: {type(e).__name__}: {e}")
                traceback.print_exc()
                logger.error(f"Demotion of {demote_candidate.id} failed. Node remains in HOT.")
                demote_candidate.tier = "hot"

        if result_node:
            return result_node

        # Check WARM (Qdrant or Disk)
        warm_node = await self._load_from_warm(node_id)
        if warm_node:
            warm_node.tier = "warm"
            warm_node.access()
            # Check promotion (pure function, no lock needed)
            if self._should_promote(warm_node):
                await self._promote_to_hot(warm_node)
            return warm_node

        # Fix 1: Fall back to COLD tier (read-only archive scan).
        cold_node = await self._load_from_cold(node_id)
        if cold_node:
            logger.debug(f"Retrieved {node_id} from COLD tier.")
        return cold_node

    async def get_memories_batch(self, node_ids: List[str]) -> List[Optional[MemoryNode]]:
        """
        Retrieve multiple memories concurrently.

        Preserves input order and returns None for missing/error cases.
        """
        if not node_ids:
            return []

        unique_ids = list(dict.fromkeys(node_ids))
        tasks = [self.get_memory(nid) for nid in unique_ids]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        result_by_id: Dict[str, Optional[MemoryNode]] = {}
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
        This forces the nodes out of WARM/COLD and into RAM for near-zero latency retrieval.
        """
        for nid in set(node_ids):
            # Check if already in HOT
            in_hot = False
            async with self.lock:
                if nid in self.hot:
                    in_hot = True
            
            if not in_hot:
                # Load from WARM
                node = await self._load_from_warm(nid)
                if node:
                    # Force promote to HOT
                    await self._promote_to_hot(node)
    async def delete_memory(self, node_id: str):
        """Robust delete from all tiers."""
        async with self.lock:
            if node_id in self.hot:
                _node = self.hot[node_id]
                # Maintain inverted chain index before removal
                if _node.previous_id:
                    self._next_chain.pop(_node.previous_id, None)
                del self.hot[node_id]
                self._remove_from_faiss(node_id)
                logger.debug(f"Deleted {node_id} from HOT")

        await self._delete_from_warm(node_id)

    def _remove_from_faiss(self, node_id: str):
        """Remove node from the ANN index (HNSW preferred, legacy flat as fallback)."""
        # Phase 4.0: delegate to HNSWIndexManager
        if self._hnsw is not None:
            self._hnsw.remove(node_id)
            return

        # Legacy FAISS flat path
        if not self.faiss_index:
            return

        fid = self.node_id_to_faiss_id.get(node_id)
        if fid is not None:
            try:
                # FAISS remove_ids is only available on some index types.
                # For IndexBinaryIDMap it works. 
                if hasattr(self.faiss_index, "remove_ids"):
                    ids_to_remove = np.array([fid], dtype='int64')
                    self.faiss_index.remove_ids(ids_to_remove)
                
                del self.faiss_id_map[fid]
                del self.node_id_to_faiss_id[node_id]
            except Exception as e:
                logger.warning(f"Failed to remove node {node_id} (FAISS ID {fid}) from index: {e}")

    async def _delete_from_warm(self, node_id: str) -> bool:
        """
        Internal helper to delete from warm and return if found.

        Returns:
            True if deleted, False otherwise.

        Note:
            Errors are logged but don't propagate to allow graceful degradation.
        """
        deleted = False
        if self.use_qdrant:
            try:
                await self.qdrant.delete(self.config.qdrant.collection_warm, [node_id])
                deleted = True
            except CircuitOpenError as e:
                logger.warning(f"Cannot delete {node_id}: {e}")
            except StorageError as e:
                logger.warning(f"Storage error deleting {node_id}: {e}")
            except Exception as e:
                logger.warning(f"Qdrant delete failed for {node_id}: {e}")

        # Filesystem fallback
        if hasattr(self, 'warm_path') and self.warm_path:
            def _fs_delete():
                d = False
                npy = self.warm_path / f"{node_id}.npy"
                jsn = self.warm_path / f"{node_id}.json"
                if npy.exists() or jsn.exists():
                    try:
                        if npy.exists():
                            npy.unlink()
                        if jsn.exists():
                            jsn.unlink()
                        d = True
                    except OSError:
                        pass
                return d

            if await self._run_in_thread(_fs_delete):
                deleted = True
                logger.debug(f"Deleted {node_id} from WARM (FS)")

        return deleted

    def _prepare_eviction_from_hot(self, exclude_node_id: Optional[str] = None) -> Optional[MemoryNode]:
        """
        Prepare eviction by finding and removing the victim from HOT.
        Returns the victim node to be saved to WARM (caller must do I/O outside lock).
        Returns None if HOT tier is empty or no valid victim found.

        Args:
            exclude_node_id: Optional ID to protect from eviction (e.g., the node just added).
        """
        if not self.hot:
            return None

        candidates = self.hot.values()
        if exclude_node_id:
            candidates = [n for n in candidates if n.id != exclude_node_id]

        if not candidates:
            return None

        victim = min(candidates, key=lambda n: n.ltp_strength)
        logger.info(f"Preparing eviction of {victim.id} from HOT (LTP: {victim.ltp_strength:.4f})")

        # Prepare for removal (don't delete yet to prevent data loss if save fails)
        victim.tier = "warm"
        
        return victim

    async def _save_to_warm(self, node: MemoryNode) -> bool:
        """
        Save node to WARM tier (Qdrant + optional FS fallback).
        Returns True if successful, False otherwise.
        """
        if self.use_qdrant:
            try:
                from qdrant_client import models

                # Unpack binary vector for Qdrant storage (Bipolar Phase 4.5)
                # Pass directly as NumPy array to save massive RAM (Qdrant natively supports it)
                bits = np.unpackbits(node.hdv.data)
                vector = (bits.astype(float) * 2.0 - 1.0)

                point = models.PointStruct(
                    id=node.id,
                    vector=vector,
                    payload={
                        "content": node.content,
                        "metadata": node.metadata,
                        "created_at": node.created_at.isoformat(),
                        "last_accessed": node.last_accessed.isoformat(),
                        "ltp_strength": node.ltp_strength,
                        "access_count": node.access_count,
                        "epistemic_value": node.epistemic_value,
                        "pragmatic_value": node.pragmatic_value,
                        "dimension": node.hdv.dimension,
                        "hdv_type": "binary",
                        # Phase 4.3: Temporal metadata for time-based indexing
                        "unix_timestamp": node.unix_timestamp,
                        "iso_date": node.iso_date,
                        "previous_id": node.previous_id,
                    }
                )

                await self.qdrant.upsert(
                    collection=self.config.qdrant.collection_warm,
                    points=[point]
                )
                return True
            except CircuitOpenError as e:
                logger.warning(f"Cannot save {node.id} to Qdrant (circuit open), falling back to FS: {e}")
                # Fall through to filesystem fallback
            except StorageError as e:
                logger.error(f"Storage error saving {node.id} to Qdrant, falling back to FS: {e}")
                # Fall through to filesystem fallback
            except Exception as e:
                logger.error(f"Failed to save {node.id} to Qdrant, falling back to FS: {e}")
                # Fall through to filesystem fallback

        # Fallback (File System)
        if not hasattr(self, 'warm_path') or not self.warm_path:
            self.warm_path = Path(self.config.paths.warm_mmap_dir)
            self.warm_path.mkdir(parents=True, exist_ok=True)

        def _fs_save():
            hdv_path = self.warm_path / f"{node.id}.npy"
            np.save(hdv_path, node.hdv.data)

            meta_path = self.warm_path / f"{node.id}.json"
            data = {
                "id": node.id,
                "content": node.content,
                "metadata": node.metadata,
                "created_at": node.created_at.isoformat(),
                "last_accessed": node.last_accessed.isoformat(),
                "ltp_strength": node.ltp_strength,
                "access_count": node.access_count,
                "tier": "warm",
                "epistemic_value": node.epistemic_value,
                "pragmatic_value": node.pragmatic_value,
                "hdv_type": "binary",
                "dimension": node.hdv.dimension,
                # Phase 4.3: Temporal metadata
                "unix_timestamp": node.unix_timestamp,
                "iso_date": node.iso_date,
                "previous_id": node.previous_id,
            }
            with open(meta_path, "w") as f:
                json.dump(data, f)

        try:
            await self._run_in_thread(_fs_save)
            return True
        except Exception as e:
            logger.error(f"FS fallback failed for {node.id}: {e}")
            return False

    async def _load_from_warm(self, node_id: str) -> Optional[MemoryNode]:
        """
        Load node from WARM tier (Qdrant or FS).

        Returns:
            MemoryNode if found, None otherwise.
        """
        if self.use_qdrant:
            try:
                record = await self.qdrant.get_point(
                    self.config.qdrant.collection_warm, node_id
                )
                if record:
                    payload = record.payload
                    vec_data = record.vector

                    try:
                        # Reconstruct BinaryHDV
                        arr = np.array(vec_data) > 0.5
                        packed = np.packbits(arr.astype(np.uint8))
                        hdv = BinaryHDV(data=packed, dimension=payload["dimension"])
                    except (ValueError, KeyError, TypeError) as e:
                        logger.error(f"Data corruption for {node_id} in Qdrant: {e}")
                        return None

                    return MemoryNode(
                        id=payload.get("id", node_id),
                        hdv=hdv,
                        content=payload["content"],
                        metadata=payload["metadata"],
                        created_at=datetime.fromisoformat(payload["created_at"]),
                        last_accessed=datetime.fromisoformat(payload["last_accessed"]),
                        tier="warm",
                        access_count=payload.get("access_count", 0),
                        ltp_strength=payload.get("ltp_strength", 0.0),
                        epistemic_value=payload.get("epistemic_value", 0.0),
                        pragmatic_value=payload.get("pragmatic_value", 0.0),
                        previous_id=payload.get("previous_id"),  # Phase 4.3
                    )
                return None  # Not found
            except CircuitOpenError as e:
                logger.warning(f"Cannot load {node_id}: {e}")
                return None
            except StorageError as e:
                logger.error(f"Storage error loading {node_id}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error loading {node_id} from Qdrant: {e}")
                return None

        # Fallback (File System)
        if hasattr(self, 'warm_path') and self.warm_path:
            def _fs_load():
                hdv_path = self.warm_path / f"{node_id}.npy"
                meta_path = self.warm_path / f"{node_id}.json"

                if not hdv_path.exists() or not meta_path.exists():
                    return None  # Not found

                try:
                    with open(meta_path, "r") as f:
                        data = json.load(f)

                    hdv_data = np.load(hdv_path)
                    hdv = BinaryHDV(data=hdv_data, dimension=data["dimension"])

                    return MemoryNode(
                        id=data["id"],
                        hdv=hdv,
                        content=data["content"],
                        metadata=data["metadata"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        last_accessed=datetime.fromisoformat(data["last_accessed"]),
                        tier="warm",
                        access_count=data.get("access_count", 0),
                        ltp_strength=data.get("ltp_strength", 0.0),
                        epistemic_value=data.get("epistemic_value", 0.0),
                        pragmatic_value=data.get("pragmatic_value", 0.0),
                        previous_id=data.get("previous_id"),  # Phase 4.3
                    )
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.error(f"Data corruption in filesystem for {node_id}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Error loading {node_id} from filesystem: {e}")
                    return None

            return await self._run_in_thread(_fs_load)
        return None

    def _should_promote(self, node: MemoryNode) -> bool:
        """Pure check: return True if node qualifies for promotion (no mutation)."""
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.promote_delta
        return node.ltp_strength > (threshold + delta)

    def _should_demote(self, node: MemoryNode) -> Optional[MemoryNode]:
        """
        Pure check: return the node if it qualifies for demotion (after updating its tier).
        Returns None if no demotion needed. No I/O performed.
        """
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.demote_delta

        if node.ltp_strength < (threshold - delta):
            return node
        return None

    async def _promote_to_hot(self, node: MemoryNode):
        """Promote node from WARM to HOT (I/O first, then atomic state update).

        Order is critical:
        1. Delete from WARM (I/O) - no lock held
        2. Insert into HOT (in-memory) - under lock
        This prevents double-promotion from concurrent callers.
        """
        # Step 1: I/O outside lock (may fail gracefully)
        deleted = await self._delete_from_warm(node.id)
        if not deleted:
            logger.debug(f"Skipping promotion of {node.id}: not found in WARM (already promoted?)")
            return

        # Step 2: Atomic state transition under lock
        victim_to_save = None
        async with self.lock:
            # Double-check: another caller may have already promoted this node
            if node.id in self.hot:
                logger.debug(f"{node.id} already in HOT, skipping duplicate promotion")
                return

            logger.info(f"Promoting {node.id} to HOT (LTP: {node.ltp_strength:.4f})")
            node.tier = "hot"
            self.hot[node.id] = node
            self._add_to_faiss(node)
            # Maintain inverted chain index
            if node.previous_id:
                self._next_chain[node.previous_id] = node.id

            # Check if we need to evict - prepare under lock, execute outside
            if len(self.hot) > self.config.tiers_hot.max_memories:
                victim_to_save = self._prepare_eviction_from_hot()

        # Step 3: Perform eviction I/O outside lock
        if victim_to_save:
            await self._save_to_warm(victim_to_save)

    async def get_stats(self) -> Dict:
        """Get statistics about memory distribution across tiers."""
        stats = {
            "hot_count": len(self.hot),
            "warm_count": 0,
            "cold_count": 0,
            "using_qdrant": self.use_qdrant,
            # Phase 4.0: HNSW index stats
            "ann_index": self._hnsw.stats() if self._hnsw is not None else {"index_type": "none"},
        }

        if self.use_qdrant:
            info = await self.qdrant.get_collection_info(self.config.qdrant.collection_warm)
            if info:
                stats["warm_count"] = info.points_count
            else:
                stats["warm_count"] = -1
        else:
            if hasattr(self, 'warm_path') and self.warm_path:
                def _count():
                    return len(list(self.warm_path.glob("*.json")))
                stats["warm_count"] = await self._run_in_thread(_count)

        return stats

    async def list_warm(self, max_results: int = 500) -> List[MemoryNode]:
        """
        List nodes from the WARM tier (Phase 4.0 — used by SemanticConsolidationWorker).

        Returns up to max_results MemoryNode objects from the WARM tier.
        Falls back gracefully if Qdrant or filesystem is unavailable.
        """
        nodes: List[MemoryNode] = []

        if self.use_qdrant:
            try:
                points_result = await self.qdrant.scroll(
                    self.config.qdrant.collection_warm,
                    limit=max_results,
                    offset=None,
                    with_vectors=True,
                )
                points = points_result[0] if points_result else []
                for pt in points:
                    payload = pt.payload
                    try:
                        arr = np.array(pt.vector) > 0.5
                        packed = np.packbits(arr.astype(np.uint8))
                        hdv = BinaryHDV(data=packed, dimension=payload["dimension"])
                        node = MemoryNode(
                            id=payload.get("id", pt.id),
                            hdv=hdv,
                            content=payload["content"],
                            metadata=payload.get("metadata", {}),
                            created_at=datetime.fromisoformat(payload["created_at"]),
                            last_accessed=datetime.fromisoformat(payload["last_accessed"]),
                            tier="warm",
                            access_count=payload.get("access_count", 0),
                            ltp_strength=payload.get("ltp_strength", 0.0),
                            previous_id=payload.get("previous_id"),  # Phase 4.3: episodic chain
                        )
                        nodes.append(node)
                    except Exception as exc:
                        logger.debug(f"list_warm: could not deserialize point {pt.id}: {exc}")
            except Exception as exc:
                logger.warning(f"list_warm Qdrant failed: {exc}")

        elif hasattr(self, "warm_path") and self.warm_path:
            def _list_fs() -> List[MemoryNode]:
                result = []
                for meta_file in list(self.warm_path.glob("*.json"))[:max_results]:
                    try:
                        import json as _json
                        with open(meta_file, "r") as f:
                            data = _json.load(f)
                        hdv_path = self.warm_path / f"{data['id']}.npy"
                        if not hdv_path.exists():
                            continue
                        hdv_data = np.load(hdv_path)
                        hdv = BinaryHDV(data=hdv_data, dimension=data["dimension"])
                        result.append(
                            MemoryNode(
                                id=data["id"],
                                hdv=hdv,
                                content=data["content"],
                                metadata=data.get("metadata", {}),
                                created_at=datetime.fromisoformat(data["created_at"]),
                                last_accessed=datetime.fromisoformat(data["last_accessed"]),
                                tier="warm",
                                ltp_strength=data.get("ltp_strength", 0.0),
                                previous_id=data.get("previous_id"),  # Phase 4.3: episodic chain
                            )
                        )
                    except Exception as exc:
                        logger.debug(f"list_warm FS: skip {meta_file.name}: {exc}")
                return result

            nodes = await self._run_in_thread(_list_fs)

        return nodes

    async def get_next_in_chain(self, node_id: str) -> Optional[MemoryNode]:
        """
        Return the MemoryNode that directly follows node_id in the episodic chain.

        This is a typed wrapper around QdrantStore.get_by_previous_id() that
        returns a fully-deserialized MemoryNode instead of a raw models.Record,
        making the episodic-chain API consistent with the rest of TierManager.

        Returns:
            The next MemoryNode in the chain, or None if not found / Qdrant
            unavailable.
        """
        # 1. Check HOT tier via O(1) inverted index (Fix 7).
        async with self.lock:
            next_id = self._next_chain.get(node_id)
            if next_id and next_id in self.hot:
                return self.hot[next_id]

        # 2. Check WARM tier (Qdrant)
        if not self.use_qdrant or not self.qdrant:
            return None

        record = await self.qdrant.get_by_previous_id(
            self.qdrant.collection_warm, node_id
        )
        if record is None:
            return None

        # Resolve to a full MemoryNode via the standard warm-load path
        return await self._load_from_warm(str(record.id))

    async def consolidate_warm_to_cold(self):
        """
        Batch move from WARM to COLD based on archive criteria.
        This is an expensive operation, typically run by a background worker.
        """
        min_ltp = self.config.tiers_warm.ltp_threshold_min

        if self.use_qdrant:
            offset = None

            while True:
                points_result = await self.qdrant.scroll(
                    self.config.qdrant.collection_warm,
                    limit=100,
                    offset=offset,
                    with_vectors=True
                )
                points = points_result[0]
                next_offset = points_result[1]

                if not points:
                    break

                ids_to_delete = []
                for pt in points:
                    payload = pt.payload
                    ltp = payload.get("ltp_strength", 0.0)

                    if ltp < min_ltp:
                        vec_data = pt.vector
                        if vec_data:
                            arr = np.array(vec_data) > 0.5
                            packed = np.packbits(arr.astype(np.uint8))
                            payload["hdv_vector"] = packed

                        await self._write_to_cold(payload)
                        ids_to_delete.append(pt.id)

                if ids_to_delete:
                    await self.qdrant.delete(
                        self.config.qdrant.collection_warm, ids_to_delete
                    )

                offset = next_offset
                if offset is None:
                    break
        else:
            # Filesystem fallback
            if hasattr(self, 'warm_path') and self.warm_path:
                def _process_fs():
                    to_delete = []
                    for meta_file in self.warm_path.glob("*.json"):
                        try:
                            with open(meta_file, "r") as f:
                                meta = json.load(f)

                            if meta.get("ltp_strength", 0.0) < min_ltp:
                                to_delete.append((meta["id"], meta))
                        except Exception:
                            pass
                    return to_delete

                candidates = await self._run_in_thread(_process_fs)
                for nid, meta in candidates:
                    await self._archive_to_cold(nid, meta)

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
        Combines HNSW/FAISS (HOT), Qdrant/FS (WARM), and optionally COLD.

        Phase 4.3: time_range filters results to memories within the given datetime range.
        Fix 1: include_cold=True enables a bounded linear scan of the COLD archive.
        """
        # 1. Search HOT via FAISS (time filtering done post-hoc for in-memory)
        hot_results = self.search_hot(query_vec, top_k)

        # Apply time filter and metadata filter to HOT results if needed
        if time_range or metadata_filter:
            filtered_hot = []
            async with self.lock:
                for nid, score in hot_results:
                    node = self.hot.get(nid)
                    if not node:
                        continue
                    
                    if time_range:
                        start_ts = time_range[0].timestamp()
                        end_ts = time_range[1].timestamp()
                        if not (start_ts <= node.created_at.timestamp() <= end_ts):
                            continue
                            
                    if metadata_filter:
                        match = True
                        node_meta = node.metadata or {}
                        for k, v in metadata_filter.items():
                            if node_meta.get(k) != v:
                                match = False
                                break
                        if not match:
                            continue
                            
                    filtered_hot.append((nid, score))
            hot_results = filtered_hot

        # 2. Search WARM via Qdrant
        warm_results = []
        if self.use_qdrant:
            try:
                q_vec = np.unpackbits(query_vec.data).astype(float)

                hits = await self.qdrant.search(
                    collection=self.config.qdrant.collection_warm,
                    query_vector=q_vec,
                    limit=top_k,
                    time_range=time_range,  # Phase 4.3: Pass time filter to Qdrant
                    metadata_filter=metadata_filter, # BUG-09: Agent Isolation
                )
                warm_results = [(hit.id, hit.score) for hit in hits]
            except Exception as e:
                logger.error(f"WARM tier search failed: {e}")

        # 3. Optionally search COLD tier (Fix 1: bounded linear scan)
        cold_results: List[Tuple[str, float]] = []
        if include_cold:
            cold_results = await self.search_cold(query_vec, top_k)

        # 4. Combine and Sort (HOT scores take precedence over WARM/COLD for same ID)
        combined = {}
        for nid, score in hot_results:
            combined[nid] = score
        for nid, score in warm_results:
            combined[nid] = max(combined.get(nid, 0), score)
        for nid, score in cold_results:
            combined[nid] = max(combined.get(nid, 0), score)

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def search_hot(self, query_vec: BinaryHDV, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search HOT tier using HNSW or FAISS binary index (Phase 4.0)."""
        # Phase 4.0: use HNSWIndexManager (auto-selects flat vs HNSW)
        if self._hnsw is not None and self._hnsw.size > 0:
            try:
                return self._hnsw.search(query_vec.data, top_k)
            except Exception as e:
                logger.error(f"HNSWIndexManager search failed, falling back: {e}")

        # Legacy FAISS flat path
        if not self.faiss_index or not self.hot:
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
            logger.error(f"FAISS search failed, falling back: {e}")
            return self._linear_search_hot(query_vec, top_k)

    def _linear_search_hot(self, query_vec: BinaryHDV, top_k: int = 5) -> List[Tuple[str, float]]:
        """Fallback linear scan for HOT tier."""
        scores = []
        for node in self.hot.values():
            sim = query_vec.similarity(node.hdv)
            scores.append((node.id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def _load_from_cold(self, node_id: str) -> Optional[MemoryNode]:
        """
        Scan COLD archive files for a specific node (Fix 1: COLD read path).

        Archives are gzip JSONL, sorted newest-first for early-exit on recent data.
        Returns None if not found or on error.
        """
        def _scan():
            for archive_file in sorted(
                self.cold_path.glob("archive_*.jsonl.gz"), reverse=True
            ):
                try:
                    with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                if rec.get("id") == node_id:
                                    return rec
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue
            return None

        rec = await self._run_in_thread(_scan)
        if rec is None:
            return None

        try:
            raw_vec = rec.get("hdv_vector")
            dim = rec.get("dimension", self.config.dimensionality)
            if raw_vec:
                hdv_data = np.array(raw_vec, dtype=np.uint8)
                hdv = BinaryHDV(data=hdv_data, dimension=dim)
            else:
                hdv = BinaryHDV.zeros(dim)

            node = MemoryNode(
                id=rec["id"],
                hdv=hdv,
                content=rec.get("content", ""),
                metadata=rec.get("metadata", {}),
                tier="cold",
                ltp_strength=rec.get("ltp_strength", 0.0),
                previous_id=rec.get("previous_id"),
            )
            if "created_at" in rec:
                node.created_at = datetime.fromisoformat(rec["created_at"])
            return node
        except Exception as e:
            logger.error(f"Failed to reconstruct node {node_id} from COLD: {e}")
            return None

    async def search_cold(
        self,
        query_vec: BinaryHDV,
        top_k: int = 5,
        max_scan: int = 1000,
    ) -> List[Tuple[str, float]]:
        """
        Linear similarity scan over COLD archive (Fix 1: COLD search path).

        Bounded by max_scan records to keep latency predictable.
        Returns results sorted by descending similarity.
        """
        config_dim = self.config.dimensionality

        def _scan():
            candidates: List[Tuple[str, float]] = []
            scanned = 0
            for archive_file in sorted(
                self.cold_path.glob("archive_*.jsonl.gz"), reverse=True
            ):
                if scanned >= max_scan:
                    break
                try:
                    with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            if scanned >= max_scan:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                raw_vec = rec.get("hdv_vector")
                                if not raw_vec:
                                    continue
                                dim = rec.get("dimension", config_dim)
                                hdv = BinaryHDV(
                                    data=np.array(raw_vec, dtype=np.uint8),
                                    dimension=dim,
                                )
                                sim = query_vec.similarity(hdv)
                                candidates.append((rec["id"], sim))
                                scanned += 1
                            except Exception:
                                continue
                except Exception:
                    continue
            return candidates

        candidates = await self._run_in_thread(_scan)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    async def _write_to_cold(self, record: dict):
        """Write a record to the cold archive."""
        record["tier"] = "cold"
        record["archived_at"] = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_file = self.cold_path / f"archive_{today}.jsonl.gz"

        def _write():
            with gzip.open(archive_file, "at", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        await self._run_in_thread(_write)

    async def _archive_to_cold(self, node_id: str, meta: dict):
        """Move memory to COLD storage (File System Fallback)."""
        if not self.warm_path:
            return

        def _read_vec():
            hdv_path = self.warm_path / f"{node_id}.npy"
            if not hdv_path.exists():
                return None
            return np.load(hdv_path)

        hdv_data = await self._run_in_thread(_read_vec)
        if hdv_data is None:
            return

        record = meta.copy()
        record["hdv_vector"] = hdv_data
        await self._write_to_cold(record)
        await self._delete_from_warm(node_id)
