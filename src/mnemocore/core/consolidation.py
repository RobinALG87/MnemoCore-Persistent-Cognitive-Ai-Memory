"""
Memory Consolidation Service

Handles the consolidation of memory nodes to long-term soul storage
based on age and free energy score criteria.

Phase 4.0+: SemanticConsolidator for clustering and merging similar memories.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import numpy as np
from loguru import logger

from .binary_hdv import BinaryHDV, majority_bundle
from .node import MemoryNode

if TYPE_CHECKING:
    from .tier_manager import TierManager


class SemanticConsolidator:
    """
    Semantic memory consolidation using Hamming distance clustering.

    Implements memory deduplication and semantic clustering to:
    1. Find clusters of similar memories (Hamming distance < threshold)
    2. Merge clusters by selecting a representative and updating metadata
    3. Consolidate entire tiers periodically

    Algorithm: Union-Find based connected components clustering
    - O(N^2) pairwise distance computation (vectorized via NumPy)
    - O(N * alpha(N)) union-find for cluster assembly
    - Well-suited for high-dimensional binary vectors

    Threshold motivation:
    - 0.85 similarity = 15% normalized Hamming distance
    - For 16384 dimensions: ~2456 differing bits is acceptable
    - Based on Kanerva's work: random vectors are ~0.5 similar
    - 0.85 similarity is well above random, indicating semantic kinship
    """

    def __init__(
        self,
        tier_manager: "TierManager",
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2,
    ):
        """
        Initialize SemanticConsolidator.

        Args:
            tier_manager: TierManager instance to access memories.
            similarity_threshold: Minimum similarity (0.0-1.0) to consider
                                 memories as candidates for merging.
                                 Default 0.85 = 15% Hamming distance.
            min_cluster_size: Minimum cluster size to consider for merging.
        """
        self.tier_manager = tier_manager
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def find_clusters(
        self,
        nodes: List[MemoryNode],
        threshold: float = 0.85,
    ) -> List[List[MemoryNode]]:
        """
        Find clusters of semantically similar memories using Hamming distance.

        Uses Union-Find algorithm to build connected components where
        each component contains memories with similarity >= threshold.

        Args:
            nodes: List of MemoryNode objects to cluster.
            threshold: Similarity threshold (0.0-1.0). Default 0.85.

        Returns:
            List of clusters, where each cluster is a list of MemoryNode objects.
            Clusters with size < min_cluster_size are excluded.
        """
        if len(nodes) < 2:
            return []

        # Build packed vector matrix for efficient distance computation
        n = len(nodes)
        dim_bytes = nodes[0].hdv.data.shape[0]
        dim_bits = dim_bytes * 8

        vecs = np.stack([node.hdv.data for node in nodes])  # (N, D/8)

        # Build adjacency based on similarity threshold
        # Use Union-Find for efficient cluster assembly
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Compute pairwise Hamming distances (vectorized)
        # For memory efficiency, process in batches if N is large
        batch_size = min(100, n)

        for i in range(n):
            # Compare node i with nodes i+1 to n-1
            for j in range(i + 1, n):
                # Compute Hamming distance using XOR and popcount
                xor = np.bitwise_xor(vecs[i], vecs[j])
                hamming_dist = int(np.unpackbits(xor).sum())
                similarity = 1.0 - (hamming_dist / dim_bits)

                if similarity >= threshold:
                    union(i, j)

        # Group nodes by their root
        cluster_map: Dict[int, List[MemoryNode]] = {}
        for i, node in enumerate(nodes):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(node)

        # Filter clusters by minimum size
        clusters = [
            cluster for cluster in cluster_map.values()
            if len(cluster) >= self.min_cluster_size
        ]

        logger.debug(
            f"find_clusters: {len(nodes)} nodes -> {len(clusters)} clusters "
            f"(threshold={threshold})"
        )

        return clusters

    def merge_cluster(
        self,
        cluster: List[MemoryNode],
    ) -> Tuple[MemoryNode, List[str]]:
        """
        Merge a cluster of similar memories into a single representative.

        Strategy:
        1. Select the memory with highest LTP strength as representative
        2. Create a proto-vector via majority bundling of all cluster members
        3. Update representative's HDV to the proto-vector
        4. Aggregate metadata from all members

        Args:
            cluster: List of similar MemoryNode objects to merge.

        Returns:
            Tuple of (representative MemoryNode, list of pruned node IDs).
        """
        if len(cluster) < 2:
            return cluster[0], []

        # Select representative: highest LTP strength
        representative = max(cluster, key=lambda n: n.ltp_strength)
        pruned_ids: List[str] = []

        # Compute proto-vector via majority bundling
        cluster_vectors = [node.hdv for node in cluster]
        proto_vector = majority_bundle(cluster_vectors)

        # Update representative with proto-vector
        representative.hdv = proto_vector

        # Aggregate metadata
        total_access_count = sum(n.access_count for n in cluster)
        representative.access_count = total_access_count

        # Boost LTP based on cluster size (strengthened by consolidation)
        # Preserve original highest LTP and add a boost
        original_ltp = representative.ltp_strength
        ltp_boost = 0.05 * len(cluster)  # Boost proportional to cluster size
        representative.ltp_strength = min(1.0, original_ltp + ltp_boost)

        # Update metadata with consolidation info
        if "consolidation_history" not in representative.metadata:
            representative.metadata["consolidation_history"] = []

        representative.metadata["consolidation_history"].append({
            "merged_count": len(cluster) - 1,
            "merged_ids": [n.id for n in cluster if n.id != representative.id],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Mark non-representative nodes for pruning
        for node in cluster:
            if node.id != representative.id:
                pruned_ids.append(node.id)

        logger.info(
            f"merge_cluster: {len(cluster)} nodes -> representative {representative.id[:8]}, "
            f"LTP={representative.ltp_strength:.3f}"
        )

        return representative, pruned_ids

    async def consolidate_tier(
        self,
        tier: str = "hot",
        threshold: float = 0.85,
    ) -> Dict[str, int]:
        """
        Consolidate memories in a specific tier.

        Process:
        1. Collect all nodes from the specified tier
        2. Find clusters of similar memories
        3. Merge each cluster
        4. Delete pruned nodes from storage

        Args:
            tier: Tier to consolidate ("hot" or "warm").
            threshold: Similarity threshold for clustering.

        Returns:
            Dict with consolidation statistics:
            - nodes_processed: Total nodes examined
            - clusters_found: Number of clusters identified
            - nodes_merged: Number of nodes merged into representatives
            - nodes_pruned: Number of nodes deleted
        """
        stats = {
            "nodes_processed": 0,
            "clusters_found": 0,
            "nodes_merged": 0,
            "nodes_pruned": 0,
        }

        # Collect nodes from tier
        if tier == "hot":
            nodes = await self.tier_manager.get_hot_snapshot()
        elif tier == "warm":
            nodes = await self.tier_manager.list_warm()
        else:
            logger.warning(f"Unknown tier: {tier}")
            return stats

        stats["nodes_processed"] = len(nodes)

        if len(nodes) < 2:
            logger.debug(f"consolidate_tier: Not enough nodes in {tier} tier")
            return stats

        # Find clusters
        clusters = self.find_clusters(nodes, threshold=threshold)
        stats["clusters_found"] = len(clusters)

        if not clusters:
            logger.debug(f"consolidate_tier: No clusters found in {tier} tier")
            return stats

        # Merge each cluster
        all_pruned_ids: List[str] = []
        for cluster in clusters:
            _, pruned_ids = self.merge_cluster(cluster)
            all_pruned_ids.extend(pruned_ids)
            stats["nodes_merged"] += len(cluster) - 1

        # Delete pruned nodes
        for node_id in all_pruned_ids:
            try:
                deleted = await self.tier_manager.delete_memory(node_id)
                if deleted:
                    stats["nodes_pruned"] += 1
            except Exception as e:
                logger.warning(f"Failed to delete node {node_id}: {e}")

        logger.info(
            f"consolidate_tier({tier}): processed={stats['nodes_processed']}, "
            f"clusters={stats['clusters_found']}, merged={stats['nodes_merged']}, "
            f"pruned={stats['nodes_pruned']}"
        )

        return stats


class ConsolidationService:
    """
    Service for consolidating memory nodes to soul storage.
    
    Identifies memories that are eligible for consolidation based on:
    - Age (minimum days old)
    - Free energy score (below threshold)
    """

    def consolidate_memories(
        self, engine, min_age_days: int = 7, threshold: float = 0.2
    ) -> List[str]:
        """
        Consolidate eligible memory nodes to soul storage.
        
        Args:
            engine: The HAIM engine instance containing memory_nodes
            min_age_days: Minimum age in days for a node to be consolidated
            threshold: Maximum free energy score for a node to be consolidated
            
        Returns:
            List of node IDs that were consolidated
        """
        consolidated_nodes = []
        # Use timezone-aware comparison if nodes are aware
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=min_age_days)
        
        logger.info(
            f"Starting memory consolidation: min_age={min_age_days} days, "
            f"threshold={threshold}"
        )
        
        # Iterate through memory nodes
        for node_id, node in engine.memory_nodes.items():
            try:
                # v1.7: Direct attribute access for dataclass
                node_date = node.created_at
                
                # Handle naive vs aware datetime mismatch if necessary
                if node_date.tzinfo is None:
                    node_date = node_date.replace(tzinfo=timezone.utc)
                    
                free_energy_score = node.get_free_energy_score()
                
                # Check consolidation criteria
                is_old_enough = node_date <= cutoff_date
                is_low_energy = free_energy_score < threshold
                
                if is_old_enough and is_low_energy:
                    logger.info(f"Consolidating {node_id} to Soul")
                    
                    # v1.7: Build Conceptual Hierarchy
                    # We store structural links in the Soul (ConceptualMemory)
                    year = node_date.strftime("%Y")
                    month = node_date.strftime("%Y-%m")
                    
                    # Bind to Time Hierarchy
                    engine.soul.append_to_concept(f"hierarchy:year:{year}", "member", node_id)
                    engine.soul.append_to_concept(f"hierarchy:month:{month}", "member", node_id)
                    
                    # Bind to Tag Hierarchy
                    tags = node.metadata.get("tags", [])
                    if isinstance(tags, list):
                        for tag in tags:
                            # Clean tag
                            clean_tag = str(tag).strip().lower().replace(" ", "_")
                            engine.soul.append_to_concept(f"hierarchy:tag:{clean_tag}", "member", node_id)
                    
                    consolidated_nodes.append(node_id)
                
            except Exception as e:
                logger.warning(f"Error processing node {node_id}: {e}")
                continue
        
        logger.info(
            f"Consolidation complete: {len(consolidated_nodes)} nodes moved to Soul"
        )
        
        return consolidated_nodes
