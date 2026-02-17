"""
Tiered Memory Management (Phase 3.0)
=====================================
Manages memory lifecycle across HOT, WARM, and COLD tiers based on Long-Term Potentiation (LTP).
"""

import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import threading
import numpy as np

from .binary_hdv import BinaryHDV
from .config import get_config
from .hdv import HDV
from .node import MemoryNode
from .storage_backends import StorageBackend, FileSystemBackend, QdrantBackend

logger = logging.getLogger(__name__)


class TierManager:
    """
    Manages memory storage across tiered hierarchy.
    Delegates storage operations to StorageBackend implementations.
    """

    def __init__(self):
        self.config = get_config()
        self.lock = threading.Lock()
        
        # HOT Tier: In-memory dictionary
        self.hot: Dict[str, MemoryNode] = {}
        
        # WARM Tier: Qdrant or File System
        try:
            self.warm_backend = QdrantBackend(self.config.qdrant.collection_warm)
            self.use_qdrant = True
        except (Exception, RuntimeError) as e:
            logger.warning(f"Qdrant not available, falling back to file system: {e}")
            self.use_qdrant = False
            self.warm_backend = FileSystemBackend(Path(self.config.paths.warm_mmap_dir))
        
        # COLD Tier path (still used for writing archive files directly)
        self.cold_path = Path(self.config.paths.cold_archive_dir)
        self.cold_path.mkdir(parents=True, exist_ok=True)
        # We could wrap COLD in a backend too, but it's append-only archive logic.

    def add_memory(self, node: MemoryNode):
        """Add a new memory node. New memories are always HOT initially."""
        node.tier = "hot"
        with self.lock:
            self.hot[node.id] = node
            
            # Check capacity
            if len(self.hot) > self.config.tiers_hot.max_memories:
                self._evict_from_hot()

    def get_memory(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory by ID from any tier."""
        # Check HOT
        with self.lock:
            if node_id in self.hot:
                node = self.hot[node_id]
                node.access()
                self._check_demotion(node)
                return node
            
        # Check WARM
        warm_node = self.warm_backend.load(node_id, "warm")
        if warm_node:
            warm_node.tier = "warm"
            warm_node.access()
            with self.lock:
                self._check_promotion(warm_node)
            return warm_node
            
        return None

    def delete_memory(self, node_id: str):
        """Robust delete from all tiers."""
        with self.lock:
            if node_id in self.hot:
                del self.hot[node_id]
                logger.debug(f"Deleted {node_id} from HOT")
        
        self.warm_backend.delete(node_id, "warm")

    def _evict_from_hot(self):
        """Evict lowest-LTP memory from HOT to WARM. Assumes lock is held."""
        if not self.hot:
            return

        victim = min(self.hot.values(), key=lambda n: n.ltp_strength)
        
        logger.info(f"Evicting {victim.id} from HOT to WARM (LTP: {victim.ltp_strength:.4f})")
        del self.hot[victim.id]
        victim.tier = "warm"
        self.warm_backend.save(victim, "warm")

    def _check_promotion(self, node: MemoryNode):
        """Check if WARM node should be promoted to HOT. Assumes lock is held."""
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.promote_delta
        
        if node.ltp_strength > (threshold + delta):
            # Promote!
            logger.info(f"Promoting {node.id} to HOT (LTP: {node.ltp_strength:.4f})")
            self.warm_backend.delete(node.id, "warm")
            node.tier = "hot"
            self.hot[node.id] = node
            
            # Ensure capacity
            if len(self.hot) > self.config.tiers_hot.max_memories:
                self._evict_from_hot()

    def _check_demotion(self, node: MemoryNode):
        """Check if HOT node should be demoted to WARM. Assumes lock is held."""
        threshold = self.config.tiers_hot.ltp_threshold_min
        delta = self.config.hysteresis.demote_delta
        
        if node.ltp_strength < (threshold - delta):
            # Demote!
            logger.info(f"Demoting {node.id} to WARM (LTP: {node.ltp_strength:.4f})")
            del self.hot[node.id]
            node.tier = "warm"
            self.warm_backend.save(node, "warm")

    def get_stats(self) -> Dict:
        """Get statistics about memory distribution across tiers."""
        return {
            "hot_count": len(self.hot),
            "warm_count": self.warm_backend.count("warm"),
            "cold_count": 0, # TODO: Count archive files?
            "using_qdrant": self.use_qdrant
        }

    def consolidate_warm_to_cold(self):
        """
        Batch move from WARM to COLD based on archive criteria.
        """
        min_ltp = self.config.tiers_warm.ltp_threshold_min
        
        # Iterate over WARM tier
        for record in self.warm_backend.iterate("warm"):
            # record is dict (metadata + maybe vector)
            ltp = record.get("ltp_strength", 0.0)
            
            if ltp < min_ltp:
                node_id = record.get("id")
                if not node_id: continue
                
                # If we have the vector in record (from Qdrant scroll), use it.
                # If from FS, iterate yields full json which doesn't have vector usually.
                # But FileSystemBackend.iterate yields json content.
                # We need to ensure we can archive it.
                
                # If record doesn't have vector, we might need to load it fully?
                # Or FS backend iterate should be smarter?
                # For FS, the old code loaded .npy.

                # Let's standardize: `iterate` yields metadata.
                # If we need to archive, we might need to call something else or `iterate` should provide everything.

                # Let's use `warm_backend.load` to get full node for archiving, to be safe.
                node = self.warm_backend.load(node_id, "warm")
                if node:
                    self._archive_to_cold(node)
                    # Delete from WARM
                    self.warm_backend.delete(node_id, "warm")

    def _archive_to_cold(self, node: MemoryNode):
        """Write node to cold archive."""
        record = {
            "id": node.id,
            "content": node.content,
            "metadata": node.metadata,
            "created_at": node.created_at.isoformat(),
            "last_accessed": node.last_accessed.isoformat(),
            "ltp_strength": node.ltp_strength,
            "access_count": node.access_count,
            "tier": "cold",
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "epistemic_value": node.epistemic_value,
            "pragmatic_value": node.pragmatic_value,
            "hdv_type": "binary" if isinstance(node.hdv, BinaryHDV) else "float",
            "dimension": node.hdv.dimension
        }

        # Vector
        if isinstance(node.hdv, BinaryHDV):
            record["hdv_vector"] = node.hdv.data.tolist()
        elif isinstance(node.hdv, HDV):
             record["hdv_vector"] = node.hdv.vector.tolist()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_file = self.cold_path / f"archive_{today}.jsonl.gz"
        with gzip.open(archive_file, "at", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
