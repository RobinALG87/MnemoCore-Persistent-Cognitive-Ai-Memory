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
import json
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

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
        async with self.lock:
            self.hot[node.id] = node
            self._add_to_faiss(node)

            if len(self.hot) > self.config.tiers_hot.max_memories:
                await self._evict_from_hot()

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
        async with self.lock:
            if node_id in self.hot:
                node = self.hot[node_id]
                node.access()
                await self._check_demotion(node)
                return node

        # Check WARM (Qdrant or Disk)
        warm_node = await self._load_from_warm(node_id)
        if warm_node:
            warm_node.tier = "warm"
            warm_node.access()
            async with self.lock:
                await self._check_promotion(warm_node)
            return warm_node

        return None

    async def delete_memory(self, node_id: str):
        """Robust delete from all tiers."""
        async with self.lock:
            if node_id in self.hot:
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
                ids_to_remove = np.array([fid], dtype='int64')
                self.faiss_index.remove_ids(ids_to_remove)
                del self.faiss_id_map[fid]
                del self.node_id_to_faiss_id[node_id]
            except Exception as e:
                logger.error(f"Failed to remove node {node_id} from FAISS: {e}")

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

    async def _evict_from_hot(self):
        """Evict lowest-LTP memory from HOT to WARM. Assumes lock is held."""
        if not self.hot:
            return

        victim = min(self.hot.values(), key=lambda n: n.ltp_strength)

        logger.info(f"Evicting {victim.id} from HOT to WARM (LTP: {victim.ltp_strength:.4f})")
        del self.hot[victim.id]
        self._remove_from_faiss(victim.id)
        victim.tier = "warm"
        await self._save_to_warm(victim)

    async def _save_to_warm(self, node: MemoryNode):
        """
        Save memory node to WARM tier (Qdrant or fallback).

        Raises:
            StorageError: If save fails (to allow caller to handle appropriately).

        Note:
            Falls back to filesystem if Qdrant save fails.
        """
        if self.use_qdrant:
            try:
                from qdrant_client import models

                # Unpack binary vector for Qdrant storage
                bits = np.unpackbits(node.hdv.data)
                vector = bits.astype(float).tolist()

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
                        "hdv_type": "binary"
                    }
                )

                await self.qdrant.upsert(
                    collection=self.config.qdrant.collection_warm,
                    points=[point]
                )
                return
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
                "dimension": node.hdv.dimension
            }
            with open(meta_path, "w") as f:
                json.dump(data, f)

        await self._run_in_thread(_fs_save)

    async def _load_from_warm(self, node_id: str) -> Optional[MemoryNode]:
        """
        Load memory node from WARM tier.

        Returns:
            MemoryNode if found, None if not found.

        Note:
            Returns None for both "not found" and storage errors to maintain
            backward compatibility. Storage errors are logged but don't propagate
            to avoid disrupting higher-level operations.
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
                        pragmatic_value=payload.get("pragmatic_value", 0.0)
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
                        pragmatic_value=data.get("pragmatic_value", 0.0)
                    )
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.error(f"Data corruption in filesystem for {node_id}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Error loading {node_id} from filesystem: {e}")
                    return None

            return await self._run_in_thread(_fs_load)
        return None

    async def _check_promotion(self, node: MemoryNode):
        """Check if WARM node should be promoted to HOT. Assumes lock is held."""
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.promote_delta

        if node.ltp_strength > (threshold + delta):
            logger.info(f"Promoting {node.id} to HOT (LTP: {node.ltp_strength:.4f})")
            await self._delete_from_warm(node.id)
            node.tier = "hot"
            self.hot[node.id] = node
            self._add_to_faiss(node)

            if len(self.hot) > self.config.tiers_hot.max_memories:
                await self._evict_from_hot()

    async def _check_demotion(self, node: MemoryNode):
        """Check if HOT node should be demoted to WARM. Assumes lock is held."""
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.demote_delta

        if node.ltp_strength < (threshold - delta):
            logger.info(f"Demoting {node.id} to WARM (LTP: {node.ltp_strength:.4f})")
            del self.hot[node.id]
            self._remove_from_faiss(node.id)
            node.tier = "warm"
            await self._save_to_warm(node)

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
            try:
                info = await self.qdrant.client.get_collection(
                    self.config.qdrant.collection_warm
                )
                stats["warm_count"] = info.points_count
            except Exception:
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
                            )
                        )
                    except Exception as exc:
                        logger.debug(f"list_warm FS: skip {meta_file.name}: {exc}")
                return result

            nodes = await self._run_in_thread(_list_fs)

        return nodes

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
                            payload["hdv_vector"] = packed.tolist()

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

    async def search(self, query_vec: BinaryHDV, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Global search across all tiers.
        Combines FAISS (HOT) and Qdrant (WARM).
        """
        # 1. Search HOT via FAISS
        hot_results = self.search_hot(query_vec, top_k)

        # 2. Search WARM via Qdrant
        warm_results = []
        if self.use_qdrant:
            try:
                q_vec = np.unpackbits(query_vec.data).astype(float).tolist()

                hits = await self.qdrant.search(
                    collection=self.config.qdrant.collection_warm,
                    query_vector=q_vec,
                    limit=top_k
                )
                warm_results = [(hit.id, hit.score) for hit in hits]
            except Exception as e:
                logger.error(f"WARM tier search failed: {e}")

        # 3. Combine and Sort
        combined = {}
        for nid, score in hot_results:
            combined[nid] = score
        for nid, score in warm_results:
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
        record["hdv_vector"] = hdv_data.tolist()
        await self._write_to_cold(record)
        await self._delete_from_warm(node_id)
