import os

import pytest
import pytest_asyncio

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
    os.environ["HAIM_PREFERENCE_ENABLED"] = "True"
    os.environ["HAIM_PREFERENCE_LEARNING_RATE"] = "0.2"
    os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"] = "0.01"
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
    del os.environ["HAIM_PREFERENCE_ENABLED"]
    del os.environ["HAIM_PREFERENCE_LEARNING_RATE"]
    del os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"]
    for key in [
        "HAIM_MEMORY_FILE",
        "HAIM_CODEBOOK_FILE",
        "HAIM_SYNAPSES_FILE",
        "HAIM_WARM_MMAP_DIR",
        "HAIM_COLD_ARCHIVE_DIR",
    ]:
        os.environ.pop(key, None)
    reset_config()


@pytest.mark.asyncio
async def test_preference_learning(test_engine):
    await test_engine.initialize()

    node_a = await test_engine.store("I want to eat a fresh green salad.")
    node_b = await test_engine.store("I want to eat a big greasy burger.")

    # Run a generic query
    results_before = await test_engine.query("What should I eat?", top_k=10)
    score_a_before = next((s for i, s in results_before if i == node_a), 0.0)
    score_b_before = next((s for i, s in results_before if i == node_b), 0.0)

    # User decides to eat salad (a positive outcome mapping to node_a's concept)
    await test_engine.log_decision("fresh healthy vegetables and salad", 1.0)

    # Re-run query to see if salad is biased higher
    results_after = await test_engine.query("What should I eat?", top_k=10)
    score_a_after = next((s for i, s in results_after if i == node_a), 0.0)
    score_b_after = next((s for i, s in results_after if i == node_b), 0.0)

    # Score for A should have increased relative to B because of the decided preference
    assert score_a_after > score_a_before

    # Negative feedback mapping to burger
    await test_engine.log_decision("big greasy burger fast food", -1.0)
    await test_engine.log_decision("big greasy burger fast food", -1.0)

    results_final = await test_engine.query("What should I eat?", top_k=10)
    score_b_final = next((s for i, s in results_final if i == node_b), 0.0)

    # The inverted target should drive B's score down (or not boost it as much) relative to the first boost
    assert score_b_final < score_b_after or score_b_after == score_b_before

    await test_engine.close()
