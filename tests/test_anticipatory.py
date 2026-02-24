import pytest
import pytest_asyncio
import os
import asyncio
from mnemocore.core.config import get_config, reset_config
from mnemocore.core.engine import HAIMEngine

@pytest.fixture
def test_engine(tmp_path):
    from mnemocore.core.hnsw_index import HNSWIndexManager
    HNSWIndexManager._instance = None
    reset_config()
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    os.environ["HAIM_ENCODING_MODE"] = "binary"
    os.environ["HAIM_DIMENSIONALITY"] = "1024"
    os.environ["HAIM_ANTICIPATORY_ENABLED"] = "True"
    os.environ["HAIM_ANTICIPATORY_PREDICTIVE_DEPTH"] = "1"
    os.environ["HAIM_MEMORY_FILE"] = str(tmp_path / "memory.jsonl")
    os.environ["HAIM_LTP_INITIAL_IMPORTANCE"] = "1.0"
    
    reset_config()
    engine = HAIMEngine()
    yield engine
    
    del os.environ["HAIM_DATA_DIR"]
    del os.environ["HAIM_ENCODING_MODE"]
    del os.environ["HAIM_DIMENSIONALITY"]
    del os.environ["HAIM_ANTICIPATORY_ENABLED"]
    del os.environ["HAIM_ANTICIPATORY_PREDICTIVE_DEPTH"]
    del os.environ["HAIM_MEMORY_FILE"]
    del os.environ["HAIM_LTP_INITIAL_IMPORTANCE"]
    reset_config()

@pytest.mark.asyncio
async def test_anticipatory_memory(test_engine):
    await test_engine.initialize()
    
    node_a = await test_engine.store("I love learning about Machine Learning and AI algorithms.")
    node_b = await test_engine.store("Neural networks and deep learning models are very powerful.")
    
    # Force a connection between them by querying both
    # or just relying on their semantic similarity
    # Let's forcefully bind them
    await test_engine.bind_memories(node_a, node_b, weight=10.0)
    
    # Demote node_b to WARM to test preloading
    mem_b_obj = await test_engine.tier_manager.get_memory(node_b)
    assert mem_b_obj is not None
    # We spoof its ltp to demote it, then call promote/demote
    mem_b_obj.ltp_strength = -1.0
    mem_b_obj.tier = "hot"
    # Actually wait get_memory doesn't demote instantly unless threshold checks out, let's just forcefuly delete from hot
    async with test_engine.tier_manager.lock:
        if node_b in test_engine.tier_manager.hot:
            del test_engine.tier_manager.hot[node_b]
            test_engine.tier_manager._remove_from_faiss(node_b)
            mem_b_obj.tier = "warm"
            
    await test_engine.tier_manager._warm_storage.save(mem_b_obj)
    assert node_b not in test_engine.tier_manager.hot
    
    # Query for something exact to node_a to guarantee it ranks first
    results = await test_engine.query("I love learning about Machine Learning and AI algorithms.", top_k=2)
    # The Anticipatory Engine should spawn a background task to preload node_b
    assert len(results) > 0
    assert results[0][0] == node_a
    
    # Wait a bit for the async preloading task to complete
    await asyncio.sleep(0.5)
    
    # Check if node_b is back in HOT
    assert node_b in test_engine.tier_manager.hot, "Anticipatory engine failed to preload node_b."

    await test_engine.close()
