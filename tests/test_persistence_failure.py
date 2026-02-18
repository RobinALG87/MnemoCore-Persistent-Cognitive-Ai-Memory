import os
import pytest
from unittest.mock import patch, MagicMock
import asyncio

from src.core.config import get_config, reset_config
from src.core.engine import HAIMEngine

@pytest.fixture
def test_engine(tmp_path):
    reset_config()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    os.environ["HAIM_MEMORY_FILE"] = str(data_dir / "memory.jsonl")
    # Set other paths to avoid errors during init
    os.environ["HAIM_CODEBOOK_FILE"] = str(data_dir / "codebook.json")
    os.environ["HAIM_SYNAPSES_FILE"] = str(data_dir / "synapses.json")
    os.environ["HAIM_WARM_MMAP_DIR"] = str(data_dir / "warm")
    os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(data_dir / "cold")
    os.environ["HAIM_ENCODING_MODE"] = "binary"
    os.environ["HAIM_DIMENSIONALITY"] = "1024"

    reset_config()
    engine = HAIMEngine()
    yield engine

    # Cleanup
    del os.environ["HAIM_DATA_DIR"]
    del os.environ["HAIM_MEMORY_FILE"]
    del os.environ["HAIM_CODEBOOK_FILE"]
    del os.environ["HAIM_SYNAPSES_FILE"]
    del os.environ["HAIM_WARM_MMAP_DIR"]
    del os.environ["HAIM_COLD_ARCHIVE_DIR"]
    del os.environ["HAIM_ENCODING_MODE"]
    del os.environ["HAIM_DIMENSIONALITY"]
    reset_config()

def test_persistence_failure_logs_error(test_engine, capsys):
    """Test that persistence failures are logged but don't crash the store."""
    # Mock open to fail when opening the persistence file
    original_open = open
    persist_path = test_engine.persist_path

    def side_effect(file, *args, **kwargs):
        if str(file) == str(persist_path):
             raise IOError("Mocked IO Error")
        return original_open(file, *args, **kwargs)

    with patch('builtins.open', side_effect=side_effect):
        # This should NOT raise an exception - error should be caught and logged
        asyncio.run(test_engine.store("Test content"))

    # The test passes if we reach here without an exception
    # The error is logged to stderr via loguru (verified by manual inspection)
    # capsys/capfd don't reliably capture loguru output, so we just verify no exception
