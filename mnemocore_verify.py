import asyncio
import os
import shutil
import pytest
from pathlib import Path
import numpy as np

# Set dummy test config environment
os.environ["HAIM_API_KEY"] = "test-key"

from mnemocore.core.config import HAIMConfig, PathsConfig, TierConfig
from mnemocore.core.binary_hdv import TextEncoder, BinaryHDV
from mnemocore.core.hnsw_index import HNSWIndexManager
from mnemocore.core.engine import HAIMEngine
from mnemocore.core.tier_manager import TierManager
from unittest.mock import patch

@pytest.fixture(autouse=True)
def setup_test_env():
    # Force all components to use test_data_verify as their data dir
    # to prevent polluting/reading the user's real ./data folder
    test_dir = Path("./test_data_verify")
    test_dir.mkdir(exist_ok=True)
    
    cfg = HAIMConfig(
        paths=PathsConfig(
            data_dir=str(test_dir),
            warm_mmap_dir=str(test_dir / "warm"),
            cold_archive_dir=str(test_dir / "cold")
        )
    )
    
    with patch('mnemocore.core.config.get_config', return_value=cfg), \
         patch('mnemocore.core.hnsw_index.get_config', return_value=cfg), \
         patch('mnemocore.core.engine.get_config', return_value=cfg):
        yield cfg
        
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.mark.asyncio
async def test_text_encoder_normalization():
    """Verify BUG-02: Text normalization fixes identical string variances"""
    encoder = TextEncoder(dimension=1024)
    hdv1 = encoder.encode("Hello World")
    hdv2 = encoder.encode("hello, world!")
    
    assert (hdv1.data == hdv2.data).all(), "Normalization failed: Different HDVs for identical texts"

def test_hnsw_singleton():
    """Verify BUG-08: HNSWIndexManager is a thread-safe singleton"""
    HNSWIndexManager._instance = None
    idx1 = HNSWIndexManager(dimension=1024)
    idx2 = HNSWIndexManager(dimension=1024)
    assert idx1 is idx2, "HNSWIndexManager is not a singleton"

def test_hnsw_index_add_search():
    """Verify BUG-01 & BUG-03: Vector cache lost / Position mapping"""
    HNSWIndexManager._instance = None
    idx = HNSWIndexManager(dimension=1024)
    
    # Optional cleanup if it's reused
    idx._id_map = []
    idx._vector_store = []
    if idx._index:
        idx._index.reset()
        
    vec1 = BinaryHDV.random(1024)
    vec2 = BinaryHDV.random(1024)
    
    idx.add("test_node_1", vec1.data)
    idx.add("test_node_2", vec2.data)
    
    assert "test_node_1" in idx._id_map, "ID Map does not contain node 1"
    assert "test_node_2" in idx._id_map, "ID Map does not contain node 2"
    
    # The search should return test_node_1 as the top result for vec1.data
    res = idx.search(vec1.data, top_k=1)
    assert res[0][0] == "test_node_1", f"Incorrect search return: {res}"

@pytest.mark.asyncio
async def test_agent_isolation():
    """Verify BUG-09: Agent namespace isolation via engine and tier manager"""
    HNSWIndexManager._instance = None
    
    test_data_dir = Path("./test_data_verify")
    test_data_dir.mkdir(exist_ok=True)
    
    config = HAIMConfig(
        qdrant=None,
        paths=PathsConfig(
            data_dir=str(test_data_dir),
            warm_mmap_dir=str(test_data_dir / "warm"),
            cold_archive_dir=str(test_data_dir / "cold")
        ),
        tiers_hot=TierConfig(max_memories=1000, ltp_threshold_min=0.0)
    )
    # Prevent newly created memories (LTP=0.5) from being eagerly demoted
    # We run purely local/in-memory for this unit test
    
    tier_manager = TierManager(config=config, qdrant_store=None)
    engine = HAIMEngine(
        persist_path=str(test_data_dir / "memory.jsonl"),
        config=config,
        tier_manager=tier_manager
    )
    
    try:
        await engine.initialize()
        
        # Store two memories, isolated
        await engine.store("Secret logic for agent 1", metadata={"agent_id": "agent_alpha"})
        await engine.store("Public logic for agent 2", metadata={"agent_id": "agent_beta"})
        
        # Search global
        res_global = await engine.query("logic", top_k=5)
        # We expect 2 given we just pushed 2
        assert len(res_global) >= 2, f"Global search should return at least 2 memories, got {len(res_global)}"
        
        # Search isolated by agent_alpha
        res_isolated = await engine.query("logic", top_k=5, metadata_filter={"agent_id": "agent_alpha"})
        
        assert len(res_isolated) > 0, "Should find at least 1 memory for agent_alpha"
        for nid, score in res_isolated:
            node = await engine.get_memory(nid)
            assert node.metadata.get("agent_id") == "agent_alpha", "Found leaked memory from another agent namespace!"
            
    finally:
        await engine.close()
        # Clean up test dir
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
