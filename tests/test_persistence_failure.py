import os
import shutil
import pytest
import logging
from unittest.mock import patch

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

def test_persistence_failure_logs_error(test_engine, caplog):
    # Setup caplog to capture logs
    caplog.set_level(logging.ERROR)

    # Mock open to fail when opening the persistence file
    original_open = open
    persist_path = test_engine.persist_path

    def side_effect(file, *args, **kwargs):
        # Normalize paths for comparison if possible, but exact string match is safer for what's passed
        if str(file) == str(persist_path):
             raise IOError("Mocked IO Error")

        return original_open(file, *args, **kwargs)

    with patch('builtins.open', side_effect=side_effect):
        # This should not raise an exception because it's caught
        # The store method calls _append_persisted at the end
        test_engine.store("Test content")

    # Verify that the error was logged
    # This assertion is expected to fail before the fix
    assert "Failed to persist memory" in caplog.text
