from typing import List, Optional
from loguru import logger

from .config import AnticipatoryConfig
from .synapse_index import SynapseIndex
from .tier_manager import TierManager
from .topic_tracker import TopicTracker

class AnticipatoryEngine:
    """
    Phase 13.2: Anticipatory Memory
    Predicts which memories the user is likely to request next based on the
    current topic trajectory and graph structure, and pre-loads them into the HOT tier.
    """
    def __init__(
        self,
        config: AnticipatoryConfig,
        synapse_index: SynapseIndex,
        tier_manager: TierManager,
        topic_tracker: TopicTracker
    ):
        self.config = config
        self.synapse_index = synapse_index
        self.tier_manager = tier_manager
        self.topic_tracker = topic_tracker

    async def predict_and_preload(self, current_node_id: str) -> List[str]:
        """
        Predicts surrounding context from the current node and ensures they are preloaded.
        Uses the multi-hop network in the SynapseIndex to find likely next nodes.
        """
        if not self.config.enabled:
            return []
            
        # Get neighbors up to predictive depth
        # We use a relatively low depth to avoid flooding the HOT tier
        neighbors = await self.synapse_index.get_multi_hop_neighbors(
            current_node_id,
            depth=self.config.predictive_depth
        )
        
        # We'll just take the top 5 highest-weighted neighbors
        # Sort by path weight (which multi-hop computes)
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:5]
        target_ids = [nid for nid, weight in sorted_neighbors if nid != current_node_id]
        
        if target_ids:
            logger.debug(f"Anticipatory engine pre-loading {len(target_ids)} predicted nodes.")
            await self.tier_manager.anticipate(target_ids)
            
        return target_ids
