import pytest
import pytest_asyncio
import os
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
    os.environ["HAIM_SYNAPSE_SIMILARITY_THRESHOLD"] = "0.4"
    os.environ["HAIM_SYNAPSE_AUTO_BIND_ON_STORE"] = "True"
    os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"] = "0.01"
    os.environ["HAIM_LTP_INITIAL_IMPORTANCE"] = "0.8"
    os.environ["HAIM_MEMORY_FILE"] = str(data_dir / "memory.jsonl")
    os.environ["HAIM_CODEBOOK_FILE"] = str(data_dir / "codebook.json")
    os.environ["HAIM_SYNAPSES_FILE"] = str(data_dir / "synapses.json")
    os.environ["HAIM_WARM_MMAP_DIR"] = str(data_dir / "warm")
    os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(data_dir / "cold")
    
    reset_config()
    engine = HAIMEngine()
    yield engine
    
    del os.environ["HAIM_DATA_DIR"]
    del os.environ["HAIM_ENCODING_MODE"]
    del os.environ["HAIM_DIMENSIONALITY"]
    del os.environ["HAIM_SYNAPSE_SIMILARITY_THRESHOLD"]
    del os.environ["HAIM_SYNAPSE_AUTO_BIND_ON_STORE"]
    del os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"]
    del os.environ["HAIM_LTP_INITIAL_IMPORTANCE"]
    for key in ["HAIM_MEMORY_FILE", "HAIM_CODEBOOK_FILE", "HAIM_SYNAPSES_FILE", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR"]:
        os.environ.pop(key, None)
    reset_config()

@pytest.mark.asyncio
async def test_auto_bind_on_store(test_engine):
    await test_engine.initialize()
    
    # Store a memory
    concept = "Python Memory Management and Garbage Collection"
    mid1 = await test_engine.store(concept)
    
    # Store a very similar memory
    concept2 = "Python Garbage Collection and Memory Management"
    mid2 = await test_engine.store(concept2)
    
    # Check if a synapse was automatically formed
    async with test_engine.synapse_lock:
        syn = test_engine._synapse_index.get(mid1, mid2)
        assert syn is not None, "Synapse should be auto-created between similar concepts"
        assert syn.fire_count >= 1

    await test_engine.close()
