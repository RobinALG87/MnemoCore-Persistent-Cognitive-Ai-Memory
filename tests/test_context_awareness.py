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
    os.environ["HAIM_CONTEXT_ENABLED"] = "True"
    # Unrelated binary vectors have similarity ~0.5.
    os.environ["HAIM_CONTEXT_SHIFT_THRESHOLD"] = "0.6"
    os.environ["HAIM_CONTEXT_ROLLING_WINDOW_SIZE"] = "3"
    os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"] = "0.01"
    
    reset_config()
    engine = HAIMEngine()
    yield engine
    
    del os.environ["HAIM_DATA_DIR"]
    del os.environ["HAIM_ENCODING_MODE"]
    del os.environ["HAIM_DIMENSIONALITY"]
    del os.environ["HAIM_CONTEXT_ENABLED"]
    del os.environ["HAIM_CONTEXT_SHIFT_THRESHOLD"]
    del os.environ["HAIM_CONTEXT_ROLLING_WINDOW_SIZE"]
    del os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"]
    reset_config()

@pytest.mark.asyncio
async def test_topic_tracker_and_context(test_engine):
    await test_engine.initialize()
    
    # Store some nodes for retrieval
    await test_engine.store("I love learning about Machine Learning and AI algorithms.")
    await test_engine.store("Neural networks and deep learning models are very powerful.")
    await test_engine.store("Pineapples belong on pizza. Yes, it is true.")

    # 1. Querying related to AI multiple times to build context
    await test_engine.query("Tell me about Machine Learning.")
    await test_engine.query("How do Neural Networks work?")
    
    # Check context nodes for AI context
    ctx_nodes1 = await test_engine.get_context_nodes(top_k=2)
    assert len(ctx_nodes1) > 0, "Should retrieve context nodes"
    
    # 2. Shift topic drastically
    await test_engine.query("What about pizza toppings? Do pineapples belong?")
    
    # The history should have reset due to shift drop
    assert len(test_engine.topic_tracker.history) == 1
    
    # The retrieved context nodes should now be heavily biased toward Pizza
    ctx_nodes2 = await test_engine.get_context_nodes(top_k=1)
    
    assert ctx_nodes2[0][0] != ctx_nodes1[0][0], "Context should have decisively shifted"

    await test_engine.close()
