from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import gzip

from .node import MemoryNode
from .hdv import HDV
from .binary_hdv import BinaryHDV
from .config import get_config

logger = logging.getLogger(__name__)

class StorageBackend(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    def save(self, node: MemoryNode, tier: str) -> bool:
        pass

    @abstractmethod
    def load(self, node_id: str, tier: str) -> Optional[MemoryNode]:
        pass

    @abstractmethod
    def delete(self, node_id: str, tier: str) -> bool:
        pass

    @abstractmethod
    def count(self, tier: str) -> int:
        pass

    @abstractmethod
    def iterate(self, tier: str):
        """Yields memory nodes or metadata for iteration."""
        pass


class FileSystemBackend(StorageBackend):
    """File-system based storage (Phase 3.0 WARM/COLD)."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, node: MemoryNode, tier: str) -> bool:
        try:
            hdv_path = self.base_path / f"{node.id}.npy"
            if isinstance(node.hdv, BinaryHDV):
                np.save(hdv_path, node.hdv.data)
            elif isinstance(node.hdv, HDV):
                np.save(hdv_path, node.hdv.vector)

            meta_path = self.base_path / f"{node.id}.json"
            data = {
                "id": node.id,
                "content": node.content,
                "metadata": node.metadata,
                "created_at": node.created_at.isoformat(),
                "last_accessed": node.last_accessed.isoformat(),
                "ltp_strength": node.ltp_strength,
                "access_count": node.access_count,
                "tier": tier,
                "epistemic_value": node.epistemic_value,
                "pragmatic_value": node.pragmatic_value,
                "hdv_type": "binary" if isinstance(node.hdv, BinaryHDV) else "float",
                "dimension": node.hdv.dimension
            }
            with open(meta_path, "w") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"FS Save failed for {node.id}: {e}")
            return False

    def load(self, node_id: str, tier: str) -> Optional[MemoryNode]:
        hdv_path = self.base_path / f"{node_id}.npy"
        meta_path = self.base_path / f"{node_id}.json"

        if not hdv_path.exists() or not meta_path.exists():
            return None

        try:
            with open(meta_path, "r") as f:
                data = json.load(f)

            hdv_data = np.load(hdv_path)
            if data.get("hdv_type") == "binary":
                hdv = BinaryHDV(data=hdv_data, dimension=data["dimension"])
            else:
                hdv = HDV(dimension=data["dimension"])
                hdv.vector = hdv_data

            return MemoryNode(
                id=data["id"],
                hdv=hdv,
                content=data["content"],
                metadata=data["metadata"],
                created_at=datetime.fromisoformat(data["created_at"]),
                last_accessed=datetime.fromisoformat(data["last_accessed"]),
                tier=tier,
                access_count=data.get("access_count", 0),
                ltp_strength=data.get("ltp_strength", 0.0),
                epistemic_value=data.get("epistemic_value", 0.0),
                pragmatic_value=data.get("pragmatic_value", 0.0)
            )
        except Exception as e:
            logger.error(f"FS Load failed for {node_id}: {e}")
            return None

    def delete(self, node_id: str, tier: str) -> bool:
        npy = self.base_path / f"{node_id}.npy"
        jsn = self.base_path / f"{node_id}.json"
        deleted = False
        if npy.exists():
            npy.unlink()
            deleted = True
        if jsn.exists():
            jsn.unlink()
            deleted = True
        return deleted

    def count(self, tier: str) -> int:
        return len(list(self.base_path.glob("*.json")))

    def iterate(self, tier: str):
        for meta_file in self.base_path.glob("*.json"):
            try:
                with open(meta_file, "r") as f:
                    yield json.load(f)
            except Exception:
                continue


class QdrantBackend(StorageBackend):
    """Qdrant-based storage (Phase 3.5 WARM)."""

    def __init__(self, collection_name: str):
        self.config = get_config()
        self.collection_name = collection_name
        try:
            from .qdrant_store import QdrantStore
            self.store = QdrantStore.get_instance()
            self.store.ensure_collections()
        except ImportError:
            raise RuntimeError("Qdrant client not installed")

    def save(self, node: MemoryNode, tier: str) -> bool:
        try:
            from qdrant_client import models

            vector = []
            if isinstance(node.hdv, BinaryHDV):
                bits = np.unpackbits(node.hdv.data)
                vector = bits.astype(float).tolist()
            elif isinstance(node.hdv, HDV):
                vector = node.hdv.vector.tolist()

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
                    "hdv_type": "binary" if isinstance(node.hdv, BinaryHDV) else "float",
                    "tier": tier
                }
            )

            self.store.upsert(
                collection=self.collection_name,
                points=[point]
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant Save failed for {node.id}: {e}")
            return False

    def load(self, node_id: str, tier: str) -> Optional[MemoryNode]:
        try:
            record = self.store.get_point(self.collection_name, node_id)
            if not record:
                return None

            payload = record.payload
            vec_data = record.vector

            if payload.get("hdv_type") == "binary":
                arr = np.array(vec_data) > 0.5
                packed = np.packbits(arr.astype(np.uint8))
                hdv = BinaryHDV(data=packed, dimension=payload["dimension"])
            else:
                hdv = HDV(dimension=payload["dimension"])
                hdv.vector = np.array(vec_data)

            return MemoryNode(
                id=payload.get("id", node_id),
                hdv=hdv,
                content=payload["content"],
                metadata=payload["metadata"],
                created_at=datetime.fromisoformat(str(payload["created_at"])),
                last_accessed=datetime.fromisoformat(str(payload["last_accessed"])),
                tier=tier,
                access_count=payload.get("access_count", 0),
                ltp_strength=payload.get("ltp_strength", 0.0),
                epistemic_value=payload.get("epistemic_value", 0.0),
                pragmatic_value=payload.get("pragmatic_value", 0.0)
            )
        except Exception as e:
            logger.error(f"Qdrant Load failed for {node_id}: {e}")
            return None

    def delete(self, node_id: str, tier: str) -> bool:
        try:
            self.store.delete(self.collection_name, [node_id])
            return True
        except Exception as e:
            logger.error(f"Qdrant Delete failed for {node_id}: {e}")
            return False

    def count(self, tier: str) -> int:
        try:
            info = self.store.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return -1

    def iterate(self, tier: str):
        # Scroll implementation
        offset = None
        while True:
            points, next_offset = self.store.scroll(
                self.collection_name,
                limit=100,
                offset=offset,
                with_vectors=True
            )

            if not points:
                break

            for pt in points:
                payload = pt.payload
                # Add vector back to payload for archiving purposes if needed
                payload["vector"] = pt.vector
                payload["id"] = pt.id
                yield payload

            offset = next_offset
            if offset is None:
                break
