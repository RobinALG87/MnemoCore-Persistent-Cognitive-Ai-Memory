"""
HAIM Test Suite — Binary HAIMEngine & Router
============================================
Tests integration of HAIMEngine with BinaryHDV and TierManager.
"""

import os
import shutil
import pytest
from datetime import datetime, timezone
import numpy as np
from pathlib import Path

from src.core.config import get_config, reset_config
from src.core.engine import HAIMEngine
from src.core.router import CognitiveRouter
from src.core.binary_hdv import BinaryHDV
from src.core.node import MemoryNode
from src.core.storage_backends import FileSystemBackend

@pytest.fixture
def binary_engine(tmp_path):
    reset_config()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    os.environ["HAIM_MEMORY_FILE"] = str(data_dir / "memory.jsonl")
    os.environ["HAIM_CODEBOOK_FILE"] = str(data_dir / "codebook.json")
    os.environ["HAIM_SYNAPSES_FILE"] = str(data_dir / "synapses.json")
    os.environ["HAIM_WARM_MMAP_DIR"] = str(data_dir / "warm")
    os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(data_dir / "cold")
    os.environ["HAIM_ENCODING_MODE"] = "binary"
    os.environ["HAIM_DIMENSIONALITY"] = "1024" # Small for tests
    # Prevent immediate demotion for testing storage
    os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"] = "0.0"
    
    reset_config()
    engine = HAIMEngine()

    # Force FileSystemBackend for robust testing, bypassing Qdrant Mock issues
    engine.tier_manager.warm_backend = FileSystemBackend(Path(engine.config.paths.warm_mmap_dir))
    engine.tier_manager.use_qdrant = False

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
    del os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"]
    reset_config()

class TestBinaryEngine:
    def test_initialization(self, binary_engine):
        assert binary_engine.config.encoding.mode == "binary"
        assert binary_engine.dimension == 1024
        assert isinstance(binary_engine.tier_manager, object)

    def test_store_memory_binary(self, binary_engine):
        # Ensure clean state
        assert len(binary_engine.tier_manager.hot) == 0

        mid = binary_engine.store("Hello World", metadata={"test": True})
        
        # Verify stored in HOT
        node = binary_engine.get_memory(mid)
        assert node is not None
        assert node.tier == "hot"
        assert isinstance(node.hdv, BinaryHDV)
        assert node.content == "Hello World"
        
        # Verify persistence log
        assert os.path.exists(binary_engine.persist_path)

    def test_query_memory_binary(self, binary_engine):
        # Store two distinct memories
        mid1 = binary_engine.store("The quick brown fox jumps over the lazy dog")
        mid2 = binary_engine.store("Quantum computing uses qubits and superposition")
        
        # Query for the first one (using full sentence to match position encoding)
        results = binary_engine.query("The quick brown fox jumps over the lazy dog", top_k=1)
        
        assert len(results) == 1
        top_id, score = results[0]

        assert top_id == mid1
        assert score > 0.9 # Should be very high similarity (identical)

    def test_context_vector_binary(self, binary_engine):
        binary_engine.store("Context 1")
        binary_engine.store("Context 2")
        
        ctx = binary_engine._current_context_vector()
        assert isinstance(ctx, BinaryHDV)
        assert ctx.dimension == 1024

    def test_calculate_eig_binary(self, binary_engine):
        v1 = BinaryHDV.random(1024)
        v2 = BinaryHDV.random(1024)
        
        eig = binary_engine.calculate_eig(v1, v2)
        # EIG = normalized distance. Random vectors ~0.5 distance.
        assert 0.4 < eig < 0.6


class TestRouterBinary:
    def test_router_reflex(self, binary_engine):
        router = CognitiveRouter(binary_engine)
        binary_engine.store("What is HAIM?", metadata={"answer": "Holographic memory"})
        
        response, debug = router.route("What is HAIM?")
        assert "Reflex" in response
        assert debug["system"] == "Sys1 (Fast)"

    def test_router_reasoning(self, binary_engine):
        router = CognitiveRouter(binary_engine)
        # Force complexity high with markers and uncertainty
        prompt = "Analyze the structural integrity of the quantum bridge design. I am unsure about the results."
        
        response, debug = router.route(prompt)
        assert "Reasoning" in response
        assert debug["system"] == "Sys2 (Slow)"
