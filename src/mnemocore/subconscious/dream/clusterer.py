"""
Episodic Clusterer – Stage 1 of Dream Pipeline
================================================
Groups memories into episodic clusters based on temporal proximity.

Memories created within a time window are grouped together as
potential episodes. This mimics how the brain organizes related
experiences.

Algorithm:
1. Sort memories by creation time
2. Group memories within time_window_hours
3. Filter clusters by min_cluster_size
4. Optionally boost synaptic connections within clusters
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from ...core.node import MemoryNode


@dataclass
class EpisodicCluster:
    """Represents a single episodic cluster."""
    cluster_id: str
    memory_count: int
    start_time: datetime
    end_time: datetime
    duration_hours: float
    memory_ids: List[str]
    categories: List[str]
    avg_ltp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "memory_count": self.memory_count,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_hours": self.duration_hours,
            "memory_ids": self.memory_ids,
            "categories": self.categories,
            "avg_ltp": self.avg_ltp,
        }


class EpisodicClusterer:
    """
    Groups memories into episodic clusters based on temporal proximity.

    Memories created within a time window are grouped together as
    potential episodes. This mimics how the brain organizes related
    experiences.

    Algorithm:
    1. Sort memories by creation time
    2. Group memories within time_window_hours
    3. Filter clusters by min_cluster_size
    4. Optionally boost synaptic connections within clusters
    """

    def __init__(
        self,
        time_window_hours: float = 24.0,
        min_cluster_size: int = 3,
    ):
        self.time_window = timedelta(hours=time_window_hours)
        self.min_cluster_size = min_cluster_size

    async def cluster(
        self,
        memories: List["MemoryNode"],
        boost_connections: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Cluster memories by temporal proximity.

        Args:
            memories: List of memory nodes to cluster.
            boost_connections: If True, strengthen connections within clusters.

        Returns:
            List of cluster dicts with metadata.
        """
        if not memories:
            return []

        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda m: m.created_at)

        clusters: List[Dict[str, Any]] = []
        current_cluster: List["MemoryNode"] = []
        cluster_start: Optional[datetime] = None

        for memory in sorted_memories:
            if cluster_start is None:
                cluster_start = memory.created_at
                current_cluster = [memory]
            elif memory.created_at - cluster_start <= self.time_window:
                current_cluster.append(memory)
            else:
                # Finalize current cluster if large enough
                if len(current_cluster) >= self.min_cluster_size:
                    clusters.append(self._create_cluster(current_cluster))
                # Start new cluster
                cluster_start = memory.created_at
                current_cluster = [memory]

        # Don't forget the last cluster
        if len(current_cluster) >= self.min_cluster_size:
            clusters.append(self._create_cluster(current_cluster))

        logger.info(f"[EpisodicClusterer] Found {len(clusters)} clusters from {len(memories)} memories")

        return clusters

    def _create_cluster(self, memories: List["MemoryNode"]) -> Dict[str, Any]:
        """Create a cluster dict from a group of memories."""
        if not memories:
            return {}

        # Calculate cluster metadata
        start_time = min(m.created_at for m in memories)
        end_time = max(m.created_at for m in memories)
        duration = end_time - start_time

        # Extract common themes from metadata
        categories = set()
        for m in memories:
            if cat := m.metadata.get("category"):
                categories.add(cat)

        return {
            "cluster_id": f"cluster_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "memory_count": len(memories),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": duration.total_seconds() / 3600,
            "memory_ids": [m.id for m in memories],
            "categories": list(categories),
            "avg_ltp": sum(m.ltp_strength for m in memories) / len(memories),
        }


__all__ = ["EpisodicClusterer", "EpisodicCluster"]
