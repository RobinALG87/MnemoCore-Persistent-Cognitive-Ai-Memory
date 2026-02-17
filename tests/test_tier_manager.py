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

import numpy as np
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.core.binary_hdv import BinaryHDV
from src.core.config import get_config, reset_config
from src.core.node import MemoryNode
from src.core.tier_manager import TierManager


@pytest.fixture
def test_config(tmp_path):
    """Setup a test configuration with temp paths."""
    reset_config()
    config = get_config()
    
    # Override paths to temp directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Use frozen dataclass replacement by tricking it or just patching
    # Since config is frozen, we can't set attributes.
    # We must reload from a test yaml or use env vars.
    # Env vars are easiest for paths.
    
    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    os.environ["HAIM_WARM_MMAP_DIR"] = str(data_dir / "warm")
    os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(data_dir / "cold")
    
    # Reload config to pick up env vars
    reset_config()
    yield get_config()
    
    # Cleanup
    del os.environ["HAIM_DATA_DIR"]
    del os.environ["HAIM_WARM_MMAP_DIR"]
    del os.environ["HAIM_COLD_ARCHIVE_DIR"]
    reset_config()


@pytest.fixture
def tier_manager(test_config):
    # Mock QdrantClient to raise error, forcing fallback to file system
    with patch("qdrant_client.QdrantClient", side_effect=Exception("Qdrant Mock Fail")):
        tm = TierManager()
    # Explicitly ensure use_qdrant is False (though exception should handle it)
    tm.use_qdrant = False
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


class TestTierManager:
    def test_add_memory_goes_to_hot(self, tier_manager):
        node = MemoryNode(id="n1", hdv=BinaryHDV.random(1024), content="c1")
        tier_manager.add_memory(node)
        
        assert "n1" in tier_manager.hot
        assert tier_manager.hot["n1"].tier == "hot"

    def test_eviction_to_warm(self, tier_manager, test_config):
        # We can't change max_memories easily on frozen config, 
        # so we fill it up to default (2000) or check logic manually.
        # Let's mock the config or just test _evict_from_hot directly.
        
        # Add two nodes
        n1 = MemoryNode(id="n1", hdv=BinaryHDV.random(1024), content="c1")
        n1.ltp_strength = 0.1 # Low
        
        n2 = MemoryNode(id="n2", hdv=BinaryHDV.random(1024), content="c2")
        n2.ltp_strength = 0.9 # High
        
        tier_manager.add_memory(n1)
        tier_manager.add_memory(n2)
        
        # Force eviction of lowest LTP (n1)
        tier_manager._evict_from_hot()
        
        assert "n1" not in tier_manager.hot
        assert "n2" in tier_manager.hot
        
        # Check if n1 is in WARM
        warm_file = tier_manager.warm_path / "n1.json"
        assert warm_file.exists()
        
        # Verify metadata
        with open(warm_file) as f:
            meta = json.load(f)
        assert meta["tier"] == "warm"
        assert meta["id"] == "n1"

    def test_retrieval_promotes_from_warm(self, tier_manager):
        # Setup: n1 in WARM with high LTP
        n1 = MemoryNode(id="n1", hdv=BinaryHDV.random(1024), content="c1")
        n1.tier = "warm"
        n1.access_count = 10 # Set high access count so calculate_ltp() yields > 0.85
        n1.calculate_ltp()
        
        # Save to WARM manually
        tier_manager._save_to_warm(n1)
        
        # Retrieve
        retrieved = tier_manager.get_memory("n1")
        
        assert retrieved is not None
        assert retrieved.tier == "hot"
        assert "n1" in tier_manager.hot
        # Should be deleted from WARM
        assert not (tier_manager.warm_path / "n1.json").exists()

    def test_consolidation_to_cold(self, tier_manager):
        # Setup: n1 in WARM with very low LTP
        n1 = MemoryNode(id="n1", hdv=BinaryHDV.random(1024), content="c1")
        n1.ltp_strength = 0.05 # < 0.3 threshold
        tier_manager._save_to_warm(n1)
        
        # Run consolidation
        tier_manager.consolidate_warm_to_cold()
        
        # Should be gone from WARM
        assert not (tier_manager.warm_path / "n1.json").exists()
        
        # Should be in COLD archive
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_file = tier_manager.cold_path / f"archive_{today}.jsonl.gz"
        assert archive_file.exists()
