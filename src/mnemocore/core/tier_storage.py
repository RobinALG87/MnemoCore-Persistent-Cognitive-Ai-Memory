"""
Tier Storage Module (Phase 6 Refactor)
======================================
Low-level CRUD operations for each memory tier.

This module handles the physical storage operations for:
- HOT (in-memory dictionary)
- WARM (Qdrant or filesystem fallback)
- COLD (compressed JSONL archives)

All tier-specific storage backends are isolated here.
"""

import asyncio
import gzip
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from mnemocore.utils import json_compat as json
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.config import HAIMConfig, get_config
from mnemocore.core.exceptions import CircuitOpenError, DataCorruptionError, StorageError

if TYPE_CHECKING:
    from mnemocore.core.node import MemoryNode
    from mnemocore.core.qdrant_store import QdrantStore


class TierInterface(ABC):
    """
    Abstract Base Class for all tier storage implementations.

    All tier storage backends must implement this interface to ensure
    consistent behavior across different storage mechanisms.
    """

    @abstractmethod
    async def get(self, node_id: str) -> Optional["MemoryNode"]:
        """Retrieve a memory node by ID."""
        pass

    @abstractmethod
    async def save(self, node: "MemoryNode") -> bool:
        """Save a memory node. Returns True if successful."""
        pass

    @abstractmethod
    async def delete(self, node_id: str) -> bool:
        """Delete a memory node by ID. Returns True if found and deleted."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return the number of nodes in this tier."""
        pass

    @abstractmethod
    async def list_all(
        self, limit: int = 500, with_vectors: bool = False
    ) -> List["MemoryNode"]:
        """List nodes in this tier."""
        pass


class HotTierStorage(TierInterface):
    """
    In-memory storage for HOT tier.

    Fastest access with limited capacity. Uses a simple dict
    for O(1) lookups.

    Thread-safety: This class does NOT have its own lock. All locking is
    delegated to TierManager's lock to prevent double-lock deadlock risk.
    Callers must ensure proper synchronization when accessing this storage.
    """

    def __init__(self, max_memories: int):
        self.max_memories = max_memories
        self._storage: Dict[str, "MemoryNode"] = {}
        self._next_chain: Dict[str, str] = {}  # previous_id -> node_id for O(1) chain lookup

    async def get(self, node_id: str) -> Optional["MemoryNode"]:
        return self._storage.get(node_id)

    async def save(self, node: "MemoryNode") -> bool:
        self._storage[node.id] = node
        if node.previous_id:
            self._next_chain[node.previous_id] = node.id
        return True

    async def delete(self, node_id: str) -> bool:
        node = self._storage.pop(node_id, None)
        if node and node.previous_id:
            self._next_chain.pop(node.previous_id, None)
        return node is not None

    async def count(self) -> int:
        return len(self._storage)

    async def list_all(
        self, limit: int = 500, with_vectors: bool = False
    ) -> List["MemoryNode"]:
        values = list(self._storage.values())
        return values[:limit]

    async def get_snapshot(self) -> List["MemoryNode"]:
        """Return a snapshot of all nodes in HOT tier."""
        return list(self._storage.values())

    async def get_recent(self, n: int) -> List["MemoryNode"]:
        """Get the most recent n memories from HOT tier."""
        all_nodes = list(self._storage.values())
        return all_nodes[-n:]

    def get_next_in_chain_id(self, node_id: str) -> Optional[str]:
        """Synchronous O(1) lookup for episodic chain traversal."""
        return self._next_chain.get(node_id)

    async def get_next_in_chain(
        self, node_id: str
    ) -> Optional["MemoryNode"]:
        """Get the next memory node in the episodic chain."""
        next_id = self._next_chain.get(node_id)
        if next_id:
            return self._storage.get(next_id)
        return None

    async def contains(self, node_id: str) -> bool:
        """Check if a node exists in HOT tier."""
        return node_id in self._storage

    def get_storage_dict(self) -> Dict[str, "MemoryNode"]:
        """
        Get a copy of the storage dict for iteration.

        Returns a shallow copy to prevent external mutation bypassing locks.
        For read-only iteration within a locked context.
        """
        return dict(self._storage)

    def get_next_chain_dict(self) -> Dict[str, str]:
        """
        Get a copy of the chain dict for iteration.

        Returns a copy to prevent external mutation bypassing locks.
        """
        return dict(self._next_chain)


class WarmTierStorage(TierInterface):
    """
    WARM tier storage with Qdrant primary and filesystem fallback.

    Handles both Qdrant vector database and local filesystem storage
    for warm memories that don't fit in HOT tier.
    """

    def __init__(
        self,
        config: Optional[HAIMConfig] = None,
        qdrant_store: Optional["QdrantStore"] = None,
    ):
        self.config = config or get_config()
        self.qdrant = qdrant_store
        self.use_qdrant = qdrant_store is not None
        self.warm_path: Optional[Path] = None

        if not self.use_qdrant:
            self.warm_path = Path(self.config.paths.warm_mmap_dir)
            self.warm_path.mkdir(parents=True, exist_ok=True)

    async def get(self, node_id: str) -> Optional["MemoryNode"]:
        """Load node from WARM tier (Qdrant or FS)."""
        if self.use_qdrant:
            return await self._load_from_qdrant(node_id)
        return await self._load_from_filesystem(node_id)

    async def save(self, node: "MemoryNode") -> bool:
        """Save node to WARM tier (Qdrant with FS fallback)."""
        if self.use_qdrant:
            try:
                success = await self._save_to_qdrant(node)
                if success:
                    return True
                logger.warning(f"Qdrant save failed for {node.id}, falling back to FS")
            except (CircuitOpenError, StorageError) as e:
                logger.warning(f"Qdrant unavailable for {node.id}: {e}")

        # Fallback to filesystem
        return await self._save_to_filesystem(node)

    async def delete(self, node_id: str) -> bool:
        """Delete from WARM tier (Qdrant and/or FS)."""
        deleted = False

        if self.use_qdrant:
            try:
                await self.qdrant.delete(
                    self.config.qdrant.collection_warm, [node_id]
                )
                deleted = True
            except (CircuitOpenError, StorageError) as e:
                logger.warning(f"Qdrant delete failed for {node_id}: {e}")

        # Also try filesystem deletion
        if self.warm_path:
            if await self._delete_from_filesystem(node_id):
                deleted = True

        return deleted

    async def count(self) -> int:
        """Return the count of nodes in WARM tier."""
        if self.use_qdrant:
            try:
                info = await self.qdrant.get_collection_info(
                    self.config.qdrant.collection_warm
                )
                return info.points_count if info else 0
            except Exception:
                pass

        if self.warm_path:
            return await self._count_filesystem()

        return 0

    async def list_all(
        self, limit: int = 500, with_vectors: bool = False
    ) -> List["MemoryNode"]:
        """List nodes from WARM tier."""
        if self.use_qdrant:
            return await self._list_qdrant(limit, with_vectors)
        return await self._list_filesystem(limit)

    async def _load_from_qdrant(self, node_id: str) -> Optional["MemoryNode"]:
        """Load node from Qdrant."""
        try:
            record = await self.qdrant.get_point(
                self.config.qdrant.collection_warm, node_id
            )
            if not record:
                return None

            payload = record.payload
            vec_data = record.vector

            try:
                arr = np.array(vec_data) > 0.5
                packed = np.packbits(arr.astype(np.uint8))
                hdv = BinaryHDV(data=packed, dimension=payload["dimension"])
            except (ValueError, KeyError, TypeError) as e:
                logger.error(f"Data corruption for {node_id} in Qdrant: {e}")
                return None

            from mnemocore.core.node import MemoryNode

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
                previous_id=payload.get("previous_id"),
            )
        except (CircuitOpenError, StorageError) as e:
            logger.warning(f"Cannot load {node_id} from Qdrant: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {node_id} from Qdrant: {e}")
            return None

    async def _load_from_filesystem(self, node_id: str) -> Optional["MemoryNode"]:
        """Load node from filesystem fallback."""
        if not self.warm_path:
            return None

        def _load():
            hdv_path = self.warm_path / f"{node_id}.npy"
            meta_path = self.warm_path / f"{node_id}.json"

            if not hdv_path.exists() or not meta_path.exists():
                return None

            try:
                with open(meta_path, "r") as f:
                    data = json.load(f)

                hdv_data = np.load(hdv_path)
                hdv = BinaryHDV(data=hdv_data, dimension=data["dimension"])

                from mnemocore.core.node import MemoryNode

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
                    previous_id=data.get("previous_id"),
                )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Data corruption in filesystem for {node_id}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error loading {node_id} from filesystem: {e}")
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _load)

    async def _save_to_qdrant(self, node: "MemoryNode") -> bool:
        """Save node to Qdrant."""
        try:
            from qdrant_client import models

            bits = np.unpackbits(node.hdv.data)
            vector = bits.astype(float) * 2.0 - 1.0

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
                    "unix_timestamp": node.unix_timestamp,
                    "iso_date": node.iso_date,
                    "previous_id": node.previous_id,
                }
            )

            await self.qdrant.upsert(
                collection=self.config.qdrant.collection_warm, points=[point]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save {node.id} to Qdrant: {e}")
            return False

    async def _save_to_filesystem(self, node: "MemoryNode") -> bool:
        """Save node to filesystem fallback."""
        if not self.warm_path:
            self.warm_path = Path(self.config.paths.warm_mmap_dir)
            self.warm_path.mkdir(parents=True, exist_ok=True)

        def _save():
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
                "unix_timestamp": node.unix_timestamp,
                "iso_date": node.iso_date,
                "previous_id": node.previous_id,
            }
            with open(meta_path, "w") as f:
                json.dump(data, f)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _save)
            return True
        except Exception as e:
            logger.error(f"FS save failed for {node.id}: {e}")
            return False

    async def _delete_from_filesystem(self, node_id: str) -> bool:
        """Delete from filesystem fallback."""
        if not self.warm_path:
            return False

        def _delete():
            npy = self.warm_path / f"{node_id}.npy"
            jsn = self.warm_path / f"{node_id}.json"
            deleted = False
            try:
                if npy.exists():
                    npy.unlink()
                    deleted = True
                if jsn.exists():
                    jsn.unlink()
                    deleted = True
            except OSError:
                pass
            return deleted

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _delete)

    async def _count_filesystem(self) -> int:
        """Count files in filesystem storage."""
        if not self.warm_path:
            return 0

        def _count():
            return len(list(self.warm_path.glob("*.json")))

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _count)

    async def _list_qdrant(
        self, limit: int, with_vectors: bool
    ) -> List["MemoryNode"]:
        """List nodes from Qdrant."""
        nodes = []
        try:
            points_result = await self.qdrant.scroll(
                self.config.qdrant.collection_warm,
                limit=limit,
                offset=None,
                with_vectors=with_vectors,
            )
            points = points_result[0] if points_result else []

            from mnemocore.core.node import MemoryNode

            for pt in points:
                payload = pt.payload
                try:
                    if with_vectors and pt.vector:
                        arr = np.array(pt.vector) > 0.5
                        packed = np.packbits(arr.astype(np.uint8))
                        hdv = BinaryHDV(data=packed, dimension=payload["dimension"])
                    else:
                        hdv = BinaryHDV.zeros(payload.get("dimension", self.config.dimensionality))

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
                        previous_id=payload.get("previous_id"),
                    )
                    nodes.append(node)
                except Exception as exc:
                    logger.debug(f"list_warm: could not deserialize point {pt.id}: {exc}")
        except Exception as exc:
            logger.warning(f"list_warm Qdrant failed: {exc}")

        return nodes

    async def _list_filesystem(self, limit: int) -> List["MemoryNode"]:
        """List nodes from filesystem fallback."""
        if not self.warm_path:
            return []

        def _list():
            from mnemocore.core.node import MemoryNode

            result = []
            for meta_file in list(self.warm_path.glob("*.json"))[:limit]:
                try:
                    with open(meta_file, "r") as f:
                        data = json.load(f)
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
                            previous_id=data.get("previous_id"),
                        )
                    )
                except Exception as exc:
                    logger.debug(f"list_warm FS: skip {meta_file.name}: {exc}")
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _list)


class ColdTierStorage(TierInterface):
    """
    COLD tier storage using compressed JSONL archives.

    Read-only archive with unlimited capacity. Optimized for
    batch operations and long-term retention.
    """

    def __init__(self, config: Optional[HAIMConfig] = None):
        self.config = config or get_config()
        self.cold_path = Path(self.config.paths.cold_archive_dir)
        self.cold_path.mkdir(parents=True, exist_ok=True)

    async def get(self, node_id: str) -> Optional["MemoryNode"]:
        """Load node from COLD archive by scanning archives."""
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

        loop = asyncio.get_running_loop()
        rec = await loop.run_in_executor(None, _scan)

        if rec is None:
            return None

        return self._deserialize_node(rec)

    async def save(self, node: "MemoryNode") -> bool:
        """Write a node to the cold archive."""
        record = {
            "id": node.id,
            "content": node.content,
            "metadata": node.metadata,
            "created_at": node.created_at.isoformat(),
            "last_accessed": node.last_accessed.isoformat(),
            "ltp_strength": node.ltp_strength,
            "access_count": node.access_count,
            "tier": "cold",
            "epistemic_value": node.epistemic_value,
            "pragmatic_value": node.pragmatic_value,
            "hdv_vector": node.hdv.data.tolist(),
            "dimension": node.hdv.dimension,
            "previous_id": node.previous_id,
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }

        return await self._write_record(record)

    async def delete(self, node_id: str) -> bool:
        """COLD tier is append-only; delete is a no-op."""
        logger.warning(f"COLD tier delete requested for {node_id} (no-op)")
        return False

    async def count(self) -> int:
        """Approximate count of records in COLD tier."""
        def _count():
            total = 0
            for archive_file in self.cold_path.glob("archive_*.jsonl.gz"):
                try:
                    with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                        for _ in f:
                            total += 1
                except Exception:
                    pass
            return total

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _count)

    async def list_all(
        self, limit: int = 500, with_vectors: bool = False
    ) -> List["MemoryNode"]:
        """List nodes from COLD tier (bounded scan)."""
        nodes = []
        scanned = 0

        def _scan():
            nonlocal scanned
            results = []
            for archive_file in sorted(
                self.cold_path.glob("archive_*.jsonl.gz"), reverse=True
            ):
                if scanned >= limit:
                    break
                try:
                    with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            if scanned >= limit:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                results.append(rec)
                                scanned += 1
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue
            return results

        loop = asyncio.get_running_loop()
        records = await loop.run_in_executor(None, _scan)

        for rec in records:
            node = self._deserialize_node(rec)
            if node:
                nodes.append(node)

        return nodes

    async def _write_record(self, record: dict) -> bool:
        """Write a record to the cold archive."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_file = self.cold_path / f"archive_{today}.jsonl.gz"

        def _write():
            with gzip.open(archive_file, "at", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write)
            return True
        except Exception as e:
            logger.error(f"Failed to write to COLD archive: {e}")
            return False

    def _deserialize_node(self, rec: dict) -> Optional["MemoryNode"]:
        """Deserialize a record from COLD archive to MemoryNode."""
        try:
            from mnemocore.core.node import MemoryNode

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
            logger.error(f"Failed to deserialize node from COLD: {e}")
            return None

    async def search(
        self, query_vec: BinaryHDV, top_k: int = 5, max_scan: int = 1000
    ) -> List[Tuple[str, float]]:
        """
        Linear similarity scan over COLD archive.

        Bounded by max_scan records to keep latency predictable.
        Returns results sorted by descending similarity.
        """
        config_dim = self.config.dimensionality

        def _scan():
            candidates = []
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

        loop = asyncio.get_running_loop()
        candidates = await loop.run_in_executor(None, _scan)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]


__all__ = [
    "TierInterface",
    "HotTierStorage",
    "WarmTierStorage",
    "ColdTierStorage",
]
