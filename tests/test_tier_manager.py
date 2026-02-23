"""
HAIM Test Suite — Tier Manager & LTP
=====================================
Tests for memory lifecycle management across HOT/WARM/COLD tiers.
"""

import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio

import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock

from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.config import get_config, reset_config
from mnemocore.core.node import MemoryNode
from mnemocore.core.tier_manager import TierManager


@pytest.fixture
def test_config(tmp_path):
    """Setup a test configuration with temp paths."""
    reset_config()
    config = get_config()
    
    # Override paths to temp directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    os.environ["HAIM_WARM_MMAP_DIR"] = str(data_dir / "warm")
    os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(data_dir / "cold")
    
    reset_config()
    yield get_config()
    
    del os.environ["HAIM_DATA_DIR"]
    del os.environ["HAIM_WARM_MMAP_DIR"]
    del os.environ["HAIM_COLD_ARCHIVE_DIR"]
    reset_config()


@pytest_asyncio.fixture
async def tier_manager(test_config):
    # Mock QdrantClient to raise error, forcing fallback to file system
    with patch("qdrant_client.QdrantClient", side_effect=Exception("Qdrant Mock Fail")):
        tm = TierManager()
        # Ensure fallback path exists since we mock away the auto-creation in __init__ if try/except fails differently?
        # Actually __init__ handles fallback.
        # But we need to await initialize?
        # TierManager.__init__ does not await.
    
    # Force fallback if needed (mock might not trigger exception in init if import succeeds but instance fails)
    tm.use_qdrant = False
    if not tm.warm_path:
        tm.warm_path = Path(test_config.paths.warm_mmap_dir)
        tm.warm_path.mkdir(parents=True, exist_ok=True)
        
    return tm


class TestLTPCalculation:
    def test_ltp_growth_with_access(self):
        # Create a node
        node = MemoryNode(
            id="test1",
            hdv=BinaryHDV.random(1024),
            content="test content"
        )
        
        initial_ltp = node.calculate_ltp()
        
        # Access it multiple times
        for _ in range(5):
            node.access()
            
        new_ltp = node.calculate_ltp()
        assert new_ltp > initial_ltp, "LTP should increase with access"

    def test_ltp_decay_with_time(self):
        node = MemoryNode(
            id="test2",
            hdv=BinaryHDV.random(1024),
            content="test content",
            created_at=datetime.now(timezone.utc) - timedelta(days=10)
        )
        
        # Calculate LTP for 10 days old
        ltp_old = node.calculate_ltp()
        
        # Compare with fresh node (same access count)
        node_fresh = MemoryNode(
            id="test3",
            hdv=BinaryHDV.random(1024),
            content="test content"
        )
        ltp_fresh = node_fresh.calculate_ltp()
        
        assert ltp_old < ltp_fresh, "LTP should decay over time"


@pytest.mark.asyncio
class TestTierManager:
    async def test_add_memory_goes_to_hot(self, tier_manager):
        node = MemoryNode(id="n1", hdv=BinaryHDV.random(1024), content="c1")
        await tier_manager.add_memory(node)
        
        # Check safely using new snapshot method or internal access
        assert "n1" in tier_manager.hot
        assert tier_manager.hot["n1"].tier == "hot"

    async def test_eviction_to_warm(self, tier_manager, test_config):
        # We can't change max_memories easily on frozen config,
        # so we fill it up to default (2000) or check logic manually.
        # Let's mock the config or just test eviction directly.

        # Add two nodes
        n1 = MemoryNode(id="n1", hdv=BinaryHDV.random(1024), content="c1")
        n1.ltp_strength = 0.1 # Low

        n2 = MemoryNode(id="n2", hdv=BinaryHDV.random(1024), content="c2")
        n2.ltp_strength = 0.9 # High

        await tier_manager.add_memory(n1)
        await tier_manager.add_memory(n2)

        # Force eviction of lowest LTP (n1) - use new two-phase method
        async with tier_manager.lock:
            victim = tier_manager._prepare_eviction_from_hot()
        if victim:
            save_ok = await tier_manager._save_to_warm(victim)
            if save_ok:
                async with tier_manager.lock:
                    if victim.id in tier_manager.hot:
                        del tier_manager.hot[victim.id]

        assert "n1" not in tier_manager.hot
        assert "n2" in tier_manager.hot

        # Check if n1 is in WARM
        warm_file = tier_manager.warm_path / "n1.json"

        # Verify metadata
        # Might need a small wait if IO is threaded?
        # But _evict awaits _save_to_warm which awaits _run_in_thread. So it should be done.

        assert warm_file.exists()

        with open(warm_file) as f:
            meta = json.load(f)
        assert meta["tier"] == "warm"
        assert meta["id"] == "n1"

    async def test_retrieval_promotes_from_warm(self, tier_manager):
        # Setup: n1 in WARM with high LTP
        n1 = MemoryNode(id="n1", hdv=BinaryHDV.random(1024), content="c1")
        n1.tier = "warm"
        n1.access_count = 10 # Ensure LTP calculation yields high value (> 0.85)
        n1.ltp_strength = 0.95 # Should trigger promotion (> 0.7 + 0.15 = 0.85)
        
        # Save to WARM manually
        await tier_manager._save_to_warm(n1)
        
        # Retrieve
        retrieved = await tier_manager.get_memory("n1")
        
        assert retrieved is not None
        assert retrieved.tier == "hot"
        assert "n1" in tier_manager.hot
        # Should be deleted from WARM
        assert not (tier_manager.warm_path / "n1.json").exists()

    async def test_consolidation_to_cold(self, tier_manager):
        # Setup: n1 in WARM with very low LTP
        n1 = MemoryNode(id="n1", hdv=BinaryHDV.random(1024), content="c1")
        n1.ltp_strength = 0.05 # < 0.3 threshold
        await tier_manager._save_to_warm(n1)
        
        # Run consolidation
        await tier_manager.consolidate_warm_to_cold()
        
        # Should be gone from WARM
        assert not (tier_manager.warm_path / "n1.json").exists()
        
        # Should be in COLD archive
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_file = tier_manager.cold_path / f"archive_{today}.jsonl.gz"
        assert archive_file.exists()
