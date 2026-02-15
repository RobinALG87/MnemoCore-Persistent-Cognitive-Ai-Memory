"""
Tiered Memory Management (Phase 3.0)
=====================================
Manages memory lifecycle across HOT, WARM, and COLD tiers based on Long-Term Potentiation (LTP).

Tiers:
  - HOT (RAM): Fast access, limited capacity. Stores most relevant memories.
  - WARM (Disk/SSD): Larger capacity, slightly slower access. Phase 3.0: NumPy files.
  - COLD (Archive): Unlimited capacity, slow access. Compressed JSONL.

Logic:
  - New memories start in HOT.
  - `consolidate()` moves memories between tiers based on LTP strength and hysteresis.
  - Promote: WARM -> HOT if `ltp > threshold + delta`
  - Demote: HOT -> WARM if `ltp < threshold - delta`
  - Archive: WARM -> COLD if `ltp < archive_threshold` (or age)
"""

import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .binary_hdv import BinaryHDV
from .config import get_config
from .hdv import HDV
from .node import MemoryNode

logger = logging.getLogger(__name__)


class TierManager:
    """
    Manages memory storage across tiered hierarchy.
    Phase 3.0 implementation using file-based WARM/COLD tiers.
    Phase 3.5 will replace WARM with Qdrant.
    """

    def __init__(self):
        self.config = get_config()
        
        # HOT Tier: In-memory dictionary
        self.hot: Dict[str, MemoryNode] = {}
        
        # WARM Tier: Qdrant
        # Initialize store (and ensure collections exist - blocking call)
        try:
            from .qdrant_store import QdrantStore
            self.qdrant = QdrantStore.get_instance()
            self.qdrant.ensure_collections()
            self.use_qdrant = True
        except Exception as e:
            logger.warning(f"Qdrant not available, falling back to file system: {e}")
            self.use_qdrant = False
            self.warm_path = Path(self.config.paths.warm_mmap_dir)
            self.warm_path.mkdir(parents=True, exist_ok=True)
        
        # COLD Tier path
        self.cold_path = Path(self.config.paths.cold_archive_dir)
        self.cold_path.mkdir(parents=True, exist_ok=True)

    def add_memory(self, node: MemoryNode):
        """Add a new memory node. New memories are always HOT initially."""
        node.tier = "hot"
        self.hot[node.id] = node
        
        # Check capacity
        if len(self.hot) > self.config.tiers_hot.max_memories:
            self._evict_from_hot()

    def get_memory(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory by ID from any tier."""
        # Check HOT
        if node_id in self.hot:
            node = self.hot[node_id]
            node.access()
            self._check_demotion(node)
            return node
            
        # Check WARM (Qdrant or Disk)
        warm_node = self._load_from_warm(node_id)
        if warm_node:
            warm_node.tier = "warm"
            warm_node.access()
            self._check_promotion(warm_node)
            return warm_node
            
        return None

    def _evict_from_hot(self):
        """Evict lowest-LTP memory from HOT to WARM."""
        if not self.hot:
            return

        sorted_nodes = sorted(self.hot.values(), key=lambda n: n.ltp_strength)
        victim = sorted_nodes[0]
        
        del self.hot[victim.id]
        victim.tier = "warm"
        self._save_to_warm(victim)

    def _save_to_warm(self, node: MemoryNode):
        """Save memory node to WARM tier (Qdrant or fallback)."""
        if self.use_qdrant:
            try:
                from qdrant_client import models
                
                # Format vector: Qdrant expects floats.
                # If BinaryHDV, we must convert packed uint8 -> bipolar floats (-1.0, 1.0) or (0.0, 1.0)
                # Qdrant Binary Quantization works best with bipolar or un-normalized floats?
                # Actually, if we enabled BQ, Qdrant will quantize on insert.
                # We need to unpack the bits.
                
                vector = []
                if isinstance(node.hdv, BinaryHDV):
                    # Unpack bits to 0/1 integers (which pass as floats)
                    # np.unpackbits is heavy, but necessary for compatibility
                    # We convert to list of floats for the client
                    bits = np.unpackbits(node.hdv.data)
                    # Use bipolar -1/1 or 0/1?
                    # Cosine distance on 0/1 vectors is monotonic with Hamming.
                    vector = bits.astype(float).tolist()
                elif isinstance(node.hdv, HDV):
                    vector = node.hdv.vector.tolist()
                
                point = models.PointStruct(
                    id=node.id,  # UUID string is supported by Qdrant (must be valid UUID or ints)
                    # Wait, Qdrant IDs must be int or UUID. Our IDs are strings.
                    # If they are not valid UUIDs, we have to hash them to UUID?
                    # Assuming for now they are UUIDs or compatible format.
                    # If generated by uuid.uuid4(), they are fine.
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
                        "hdv_type": "binary" if isinstance(node.hdv, BinaryHDV) else "float"
                    }
                )
                
                self.qdrant.upsert(
                    collection=self.config.qdrant.collection_warm,
                    points=[point]
                )
                
                # Also saving to HOT collection if dual-tiering? No, eviction implies removal from HOT.
                # But for query consistency, we search both? 
                # WARM is separate collection.
                return
            except Exception as e:
                logger.error(f"Failed to save {node.id} to Qdrant: {e}")
                # Fallback to local file?
        
        # Fallback (File System)
        if not hasattr(self, 'warm_path'):
             self.warm_path = Path(self.config.paths.warm_mmap_dir)
             self.warm_path.mkdir(parents=True, exist_ok=True)
             
        hdv_path = self.warm_path / f"{node.id}.npy"
        if isinstance(node.hdv, BinaryHDV):
            np.save(hdv_path, node.hdv.data)
        elif isinstance(node.hdv, HDV):
             np.save(hdv_path, node.hdv.vector)
        
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
            "hdv_type": "binary" if isinstance(node.hdv, BinaryHDV) else "float",
            "dimension": node.hdv.dimension
        }
        with open(meta_path, "w") as f:
            json.dump(data, f)

    def _load_from_warm(self, node_id: str) -> Optional[MemoryNode]:
        """Load memory node from WARM tier."""
        if self.use_qdrant:
            try:
                record = self.qdrant.get_point(self.config.qdrant.collection_warm, node_id)
                if record:
                    payload = record.payload
                    vec_data = record.vector
                    
                    # Reconstruct HDV
                    if payload.get("hdv_type") == "binary":
                        # Vector is list of floats (0.0/1.0). Need to pack back to uint8.
                        # arr = np.array(vec_data, dtype=int)
                        # packed = np.packbits(arr)
                        # Actually simpler: if we get 0/1s
                        arr = np.array(vec_data) > 0.5
                        packed = np.packbits(arr.astype(np.uint8))
                        hdv = BinaryHDV(data=packed, dimension=payload["dimension"])
                    else:
                        hdv = HDV(dimension=payload["dimension"])
                        hdv.vector = np.array(vec_data)
                    
                    return MemoryNode(
                        id=payload.get("id", node_id), # Some payloads might store ID
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
            except Exception as e:
                logger.error(f"Failed to load user {node_id} from Qdrant: {e}")
        
        # Fallback (File System)
        if hasattr(self, 'warm_path'):
            hdv_path = self.warm_path / f"{node_id}.npy"
            meta_path = self.warm_path / f"{node_id}.json"
            
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
                    tier="warm",
                    access_count=data.get("access_count", 0),
                    ltp_strength=data.get("ltp_strength", 0.0),
                    epistemic_value=data.get("epistemic_value", 0.0),
                    pragmatic_value=data.get("pragmatic_value", 0.0)
                )
            except Exception:
                return None
        return None

    def _check_promotion(self, node: MemoryNode):
        """Check if WARM node should be promoted to HOT."""
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.promote_delta
        
        if node.ltp_strength > (threshold + delta):
            # Promote!
            self._delete_from_warm(node.id)
            node.tier = "hot"
            self.hot[node.id] = node
            
            # Ensure capacity
            if len(self.hot) > self.config.tiers_hot.max_memories:
                self._evict_from_hot()

    def _check_demotion(self, node: MemoryNode):
        """Check if HOT node should be demoted to WARM."""
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.demote_delta
        
        if node.ltp_strength < (threshold - delta):
            # Demote!
            del self.hot[node.id]
            node.tier = "warm"
            self._save_to_warm(node)

    def _delete_from_warm(self, node_id: str):
        """Delete from WARM tier."""
        if self.use_qdrant:
            try:
                self.qdrant.delete(self.config.qdrant.collection_warm, [node_id])
            except Exception as e:
                logger.warning(f"Could not delete {node_id} from Qdrant: {e}")
        
        # Also clean filesystem just in case of mixed usage
        if hasattr(self, 'warm_path'):
            try:
                (self.warm_path / f"{node_id}.npy").unlink(missing_ok=True)
                (self.warm_path / f"{node_id}.json").unlink(missing_ok=True)
            except Exception:
                pass

    def consolidate_warm_to_cold(self):
        """
        Batch move from WARM to COLD based on archive criteria.
        This is an expensive operation, typically run by a background worker.
        """
        min_ltp = self.config.tiers_warm.ltp_threshold_min
        
        # If using Qdrant, scroll
        if self.use_qdrant:
            # Scroll through Qdrant
            offset = None
            while True:
                points, next_offset = self.qdrant.scroll(
                    self.config.qdrant.collection_warm, 
                    limit=100, 
                    offset=offset
                )
                if not points:
                    break
                    
                for pt in points:
                    ltp = pt.payload.get("ltp_strength", 0.0)
                    if ltp < min_ltp:
                        # Archive logic using payload data + reconstruct logic
                        # Simplified: Re-use _load_from_warm logic or redundant fetch?
                        # Since scroll returns payload but not vector (unless with_vectors=True),
                        # we need to fetch vector or set with_vectors=True above.
                        # Let's assume we implement full archive properly later or fetch singly.
                        pass # TODO: Implement full consolidation with Qdrant
                
                offset = next_offset
                if offset is None:
                    break
        else:
            # Iterate over WARM tier files (Legacy / Fallback)
            if hasattr(self, 'warm_path'):
                for meta_file in self.warm_path.glob("*.json"):
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                        
                        ltp = meta.get("ltp_strength", 0.0)
                        node_id = meta["id"]
                        
                        if ltp < min_ltp:
                            self._archive_to_cold(node_id, meta)
                    except Exception as e:
                        logger.error(f"Error processing {meta_file} for consolidation: {e}")

    def _archive_to_cold(self, node_id: str, meta: dict):
        """Move memory to COLD storage (compressed JSONL)."""
        # Load the HDV first to archive it fully (need to handle Qdrant here too)
        # For Phase 3.5 MVP, let's keep COLD logic focused on file fallback for now.
        # Advancing this is tricky without vector fetch. 
        # Leaving existing file logic.
        
        hdv_path = self.warm_path / f"{node_id}.npy"
        if not hdv_path.exists():
            return
            
        hdv_data = np.load(hdv_path)
        
        record = meta.copy()
        record["hdv_vector"] = hdv_data.tolist()
        record["tier"] = "cold"
        record["archived_at"] = datetime.now(timezone.utc).isoformat()
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_file = self.cold_path / f"archive_{today}.jsonl.gz"
        
        with gzip.open(archive_file, "at", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        self._delete_from_warm(node_id)
