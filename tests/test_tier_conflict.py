
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

from mnemocore.core.tier_manager import TierManager
from mnemocore.core.node import MemoryNode
from mnemocore.core.binary_hdv import BinaryHDV
import numpy as np

@pytest.mark.asyncio
async def test_get_memory_demotion_race_condition():
    """
    Verify that get_memory correctly handles nodes pending demotion.
    The returned node should have tier='warm' even if the async move hasn't finished.
    """
    # Setup
    tier_manager = TierManager()
    tier_manager.use_qdrant = False # Use filesystem/mock for simplicity
    tier_manager.warm_path = MagicMock() # Mock path to avoid actual IO
    
    # Mock save_to_warm to be slow (simulating IO)
    original_save = tier_manager._warm_storage.save
    save_event = asyncio.Event()
    
    async def slow_save(node):
        await asyncio.sleep(0.1) # Simulate IO delay
        save_event.set()
        return True
        
    tier_manager._warm_storage.save = AsyncMock(side_effect=slow_save)
    tier_manager._remove_from_faiss = MagicMock()
    
    # Create a HOT node
    dim = 1000
    # BinaryHDV expects packed uint8 array of size dim // 8
    # 1000 // 8 = 125 bytes
    packed_data = np.zeros(dim // 8, dtype=np.uint8)
    
    node = MemoryNode(
        id="test-node-1",
        content="test content",
        hdv=BinaryHDV(packed_data, dim),
        tier="hot"
    )
    # Set LTP to be very low so it triggers demotion
    node.ltp_strength = 0.0 
    
    # Add to manager directly
    tier_manager._hot_storage._storage[node.id] = node
    
    # Mock config thresholds to ensure demotion triggers
    # Config objects are frozen, so we must safely replace them
    import dataclasses
    from mnemocore.core.config import get_config
    
    real_config = get_config()
    # Set threshold very high (2.0) so even with access boost (LTP ~0.55) it still demotes
    # 2.0 - 0.1 = 1.9 > 0.55
    new_hot_config = dataclasses.replace(real_config.tiers_hot, ltp_threshold_min=0.99)
    new_hysteresis = dataclasses.replace(real_config.hysteresis, demote_delta=0.1)
    new_config = dataclasses.replace(real_config, tiers_hot=new_hot_config, hysteresis=new_hysteresis)
    
    tier_manager.config = new_config
    # Propagate new config to sub-managers that drive demotion decisions
    tier_manager._eviction_manager.config = new_config

    # EXECUTE
    # This call should trigger demotion logic
    returned_node = await tier_manager.get_memory("test-node-1")
    
    # ASSERTIONS
    
    # 1. The returned node MUST be marked as 'warm' immediately
    assert returned_node.tier == "warm", "Node should be marked warm immediately to prevent TOCTOU"
    
    # 2. The node should NOT be in hot anymore (eventually)
    # Wait for the background task/await to finish
    # In the current implementation, get_memory AWAITS the save, so it should be done.
    
    assert tier_manager._warm_storage.save.called, "Should have called save_to_warm"
    
    # Check that it was removed from hot
    async with tier_manager.lock:
        assert "test-node-1" not in tier_manager.hot, "Node should be removed from hot"

if __name__ == "__main__":
    asyncio.run(test_get_memory_demotion_race_condition())
