"""
HAIM Test Suite — Binary HAIMEngine & Router
============================================
Tests integration of HAIMEngine with BinaryHDV and TierManager.
"""

import os
import shutil
import pytest
import pytest_asyncio
from datetime import datetime, timezone
import numpy as np

from src.core.config import get_config, reset_config
from src.core.engine import HAIMEngine
from src.core.router import CognitiveRouter
from src.core.binary_hdv import BinaryHDV
from src.core.node import MemoryNode

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
    os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"] = "0.01" # Prevent demotion
    os.environ["HAIM_LTP_INITIAL_IMPORTANCE"] = "0.8" # Higher start
    
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

@pytest.mark.asyncio
class TestBinaryEngine:
    def test_initialization(self, binary_engine):
        assert binary_engine.config.encoding.mode == "binary"
        assert binary_engine.dimension == 1024
        assert isinstance(binary_engine.tier_manager, object)

    async def test_store_memory_binary(self, binary_engine):
        mid = await binary_engine.store("Hello World", metadata={"test": True})
        
        # Verify stored in HOT
        node = await binary_engine.get_memory(mid)
        assert node is not None
        assert node.tier == "hot"
        assert isinstance(node.hdv, BinaryHDV)
        assert node.content == "Hello World"
        
        # Verify persistence log
        assert os.path.exists(binary_engine.persist_path)

    async def test_query_memory_binary(self, binary_engine):
        # Store two distinct memories
        mid1 = await binary_engine.store("The quick brown fox jumps over the lazy dog")
        mid2 = await binary_engine.store("Quantum computing uses qubits and superposition")
        
        # Query for the first one
        results = await binary_engine.query("quick brown fox", top_k=1)
        
        assert len(results) == 1
        top_id, score = results[0]
        assert top_id == mid1
        assert score > 0.5 # Should be high similarity

    async def test_context_vector_binary(self, binary_engine):
        await binary_engine.store("Context 1")
        await binary_engine.store("Context 2")
        
        ctx = await binary_engine._current_context_vector()
        assert isinstance(ctx, BinaryHDV)
        assert ctx.dimension == 1024

    def test_calculate_eig_binary(self, binary_engine):
        v1 = BinaryHDV.random(1024)
        v2 = BinaryHDV.random(1024)
        
        eig = binary_engine.calculate_eig(v1, v2)
        # EIG = normalized distance. Random vectors ~0.5 distance.
        assert 0.4 < eig < 0.6


class TestRouterBinary:
    async def test_router_reflex(self, binary_engine):
        router = CognitiveRouter(binary_engine)
        await binary_engine.store("What is HAIM?", metadata={"answer": "Holographic memory"})
        
        response, debug = await router.route("What is HAIM?")
        assert "Reflex" in response
        assert debug["system"] == "Sys1 (Fast)"

    async def test_router_reasoning(self, binary_engine):
        router = CognitiveRouter(binary_engine)
        # Force complexity high
        prompt = "Analyze the structural integrity of the quantum bridge design"
        
        response, debug = await router.route(prompt)
        assert "Reasoning" in response
        assert debug["system"] == "Sys2 (Slow)"
