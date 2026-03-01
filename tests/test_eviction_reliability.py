import pytest
import asyncio
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from mnemocore.core.tier_manager import TierManager
from mnemocore.core.node import MemoryNode
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.config import HAIMConfig
from mnemocore.core.exceptions import StorageError

@pytest.fixture
def mock_config():
    # Setup test directory
    test_dir = Path("./data/test_eviction")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    cfg = MagicMock(spec=HAIMConfig)
    cfg.dimensionality = 16384
    cfg.tiers_hot = MagicMock()
    cfg.tiers_hot.max_memories = 2
    cfg.tiers_hot.ltp_threshold_min = 0.5
    cfg.hysteresis = MagicMock()
    cfg.hysteresis.promote_delta = 0.1
    cfg.hysteresis.demote_delta = 0.1
    cfg.ltp = MagicMock()
    cfg.ltp.decay_lambda = 0.001
    cfg.tiers_warm = MagicMock()
    cfg.tiers_warm.eviction_policy = "lru"
    cfg.paths = MagicMock()
    cfg.paths.data_dir = str(test_dir)
    cfg.paths.warm_mmap_dir = str(test_dir / "warm")
    cfg.paths.cold_archive_dir = str(test_dir / "cold")
    cfg.qdrant = MagicMock()
    cfg.qdrant.collection_warm = "test_warm"
    return cfg

@pytest.mark.asyncio
async def test_eviction_reliability_on_failure(mock_config):
    # Setup TierManager with a failing Qdrant mock AND failing FS (by making it read-only or invalid path)
    mock_qdrant = AsyncMock()
    mock_qdrant.upsert.side_effect = StorageError("Simulated write failure")
    
    # Intentionally break the FS path to force total failure
    mock_config.paths.warm_mmap_dir = "/invalid/path/that/should/fail"
    
    tm = TierManager(config=mock_config, qdrant_store=mock_qdrant)
    tm.use_qdrant = True
    
    # Add first memory
    node1 = MemoryNode(id="node1", hdv=BinaryHDV.random(16384), content="test1")
    await tm.add_memory(node1)
    
    # Add second memory
    node2 = MemoryNode(id="node2", hdv=BinaryHDV.random(16384), content="test2")
    await tm.add_memory(node2)
    
    # Assert HOT tier has 2 nodes
    assert len(tm.hot) == 2
    
    # Add third memory - should trigger eviction of node1 (lowest LTP default)
    node3 = MemoryNode(id="node3", hdv=BinaryHDV.random(16384), content="test3")
    
    # Simulate total failure (Qdrant fails, and we force _warm_storage.save to return False)
    # This verifies the logic in add_memory that checks the return value.
    with patch.object(tm._warm_storage, 'save', return_value=False):
        await tm.add_memory(node3)
    
    # Verify node1 is still in HOT because save failed
    assert "node1" in tm.hot
    assert tm.hot["node1"].tier == "hot"
    assert len(tm.hot) == 3 

@pytest.mark.asyncio
async def test_demotion_reliability_with_concurrent_access(mock_config):
    # Setup TierManager
    mock_qdrant = AsyncMock()
    tm = TierManager(config=mock_config, qdrant_store=mock_qdrant)
    tm.use_qdrant = True
    
    # Add a memory and force its LTP down to trigger demotion
    node = MemoryNode(id="node1", hdv=BinaryHDV.random(16384), content="test1")
    node.ltp_strength = 0.1 
    await tm.add_memory(node)
    
    # Mock upsert to handle any arguments (collection, points, etc.)
    async def delayed_save(*args, **kwargs):
        # Simulate concurrent access that promotes it back
        node.access() # Boosts LTP
        node.tier = "hot" 
        await asyncio.sleep(0.1)
        return True # Success
        
    mock_qdrant.upsert.side_effect = delayed_save
    
    # Now trigger demotion logic
    demote_candidate = node
    demote_candidate.tier = "warm"
    
    # Logic from tier_manager.py:252+
    await tm._warm_storage.save(demote_candidate)
    
    async with tm.lock:
        if demote_candidate.id in tm.hot:
            if tm.hot[demote_candidate.id].tier == "warm":
                del tm.hot[demote_candidate.id]
                tm._remove_from_faiss(demote_candidate.id)

    # Verify node is still in HOT
    assert "node1" in tm.hot
    assert tm.hot["node1"].tier == "hot"
