"""
Memory Consolidation Service

Handles the consolidation of memory nodes to long-term soul storage
based on age and free energy score criteria.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List

# Configure logging
logger = logging.getLogger(__name__)


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
