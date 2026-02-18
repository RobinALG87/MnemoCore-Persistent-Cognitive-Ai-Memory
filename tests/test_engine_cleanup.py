"""
Test HAIMEngine Synapse Cleanup
"""
import os
import pytest
from src.core.engine import HAIMEngine
from src.core.synapse import SynapticConnection
from src.core.config import reset_config

@pytest.fixture
def test_engine(tmp_path):
    reset_config()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    synapses_file = data_dir / "synapses.json"
    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    os.environ["HAIM_MEMORY_FILE"] = str(data_dir / "memory.jsonl")
    os.environ["HAIM_SYNAPSES_FILE"] = str(synapses_file)

    engine = HAIMEngine()
    yield engine

    # Cleanup
    if "HAIM_DATA_DIR" in os.environ:
        del os.environ["HAIM_DATA_DIR"]
    if "HAIM_MEMORY_FILE" in os.environ:
        del os.environ["HAIM_MEMORY_FILE"]
    if "HAIM_SYNAPSES_FILE" in os.environ:
        del os.environ["HAIM_SYNAPSES_FILE"]
    reset_config()

@pytest.mark.asyncio
async def test_cleanup_decay(test_engine):
    # Add dummy synapses
    # Synapse 1: Weak (below threshold 0.1)
    syn1 = SynapticConnection("mem_1", "mem_2", initial_strength=0.05)
    test_engine.synapses[("mem_1", "mem_2")] = syn1

    # Synapse 2: Strong (above threshold 0.1)
    syn2 = SynapticConnection("mem_3", "mem_4", initial_strength=0.2)
    test_engine.synapses[("mem_3", "mem_4")] = syn2

    # Check initial count
    assert len(test_engine.synapses) == 2

    # Run cleanup
    await test_engine.cleanup_decay(threshold=0.1)

    # Verify results
    assert len(test_engine.synapses) == 1
    assert ("mem_3", "mem_4") in test_engine.synapses
    assert ("mem_1", "mem_2") not in test_engine.synapses

    # Verify persistence
    assert os.path.exists(test_engine.synapse_path)

@pytest.mark.asyncio
async def test_cleanup_no_decay(test_engine):
    # All strong
    syn1 = SynapticConnection("mem_1", "mem_2", initial_strength=0.5)
    test_engine.synapses[("mem_1", "mem_2")] = syn1

    await test_engine.cleanup_decay(threshold=0.1)

    assert len(test_engine.synapses) == 1
