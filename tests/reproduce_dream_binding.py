import asyncio
import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import json

from loguru import logger

# Adjust path to import src
import sys
# Assume we are running from project root or checks relative path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.config import get_config, SubconsciousAIConfig, HAIMConfig
from mnemocore.core.subconscious_ai import SubconsciousAIWorker, ModelClient
from mnemocore.core.node import MemoryNode
from mnemocore.core.binary_hdv import BinaryHDV

# Mock Model Client
class MockModelClient(ModelClient):
    def __init__(self, responses):
        # We don't call super().__init__ because it expects model_url
        self.responses = responses
        self.call_count = 0
        self.model_name = "mock-model"
        self.model_url = "mock-url"

    async def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        logger.info(f"MockModelClient received prompt: {prompt[:50]}...")
        
        # Simple keyword matching
        if "Categorize" in prompt:
            return self.responses.get("sorting", "{}")
        if "Analyze these memories" in prompt:
            logger.info("Generating dreaming response...")
            return self.responses.get("dreaming", "{}")
            
        return "{}"

async def run_verification():
    """Verify that _enhanced_dreaming_cycle actually binds memories."""
    
    # Setup test env
    test_dir = Path("./test_env_dream").resolve()
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    logger.info(f"Test Environment: {test_dir}")
    
    # Configure
    # We must mock config properly using replace since it is frozen
    import dataclasses
    from mnemocore.core.config import PathsConfig
    
    cfg = get_config()
    
    # Create new PathsConfig
    new_paths = dataclasses.replace(
        cfg.paths,
        data_dir=str(test_dir),
        memory_file=str(test_dir / "memories.pkl"),
        synapses_file=str(test_dir / "synapses.jsonl")
    )
    
    # Enable dreaming
    sub_cfg = SubconsciousAIConfig(
        enabled=True,
        enhanced_dreaming_enabled=True,
        pulse_interval_seconds=1,
        model_provider="mock",
        model_name="mock-model",
        dry_run=False  # IMPORTANT: Must invoke bind_memories
    )
    
    # Create new HAIMConfig with updated paths and subconscious config
    cfg = dataclasses.replace(
        cfg,
        paths=new_paths,
        subconscious_ai=sub_cfg
    )

    # Initialize Engine
    logger.info("Initializing Engine...")
    engine = HAIMEngine(config=cfg)
    await engine.initialize()
    
    # Create some dummy memories
    logger.info("Creating dummy memories...")
    
    # 1. Weak memory (low LTP)
    # create node manually to control LTP
    weak_content = "The cat sat on the mat."
    weak_vec = engine.binary_encoder.encode(weak_content)
    weak_node = MemoryNode(
        id="weak-memory-1",
        hdv=weak_vec,
        content=weak_content,
        metadata={"source": "user", "ltp_strength": 0.1}
    )
    # Ensure LTP is low
    weak_node.ltp_strength = 0.2
    # Ensure it's not analyzed yet
    weak_node.metadata.pop("dream_analyzed", None)
    
    await engine.tier_manager.add_memory(weak_node)
    
    # 2. Related memory (to be bridged)
    related_content = "Felines enjoy resting on rugs."
    related_vec = engine.binary_encoder.encode(related_content)
    related_node = MemoryNode(
        id="related-memory-2",
        hdv=related_vec,
        content=related_content,
        metadata={"source": "user"}
    )
    await engine.tier_manager.add_memory(related_node)
    
    logger.info(f"Weak Memory ID: {weak_node.id}")
    logger.info(f"Related Memory ID: {related_node.id}")
    
    # Initialize Worker
    logger.info("Initializing Subconscious Worker...")
    worker = SubconsciousAIWorker(engine, sub_cfg)
    
    # Mock LLM Response
    # The prompt asks for JSON containing "bridges".
    # The weak memory is index 0 in the prompt list (since it's recent and weak).
    # We simulate a suggestion for bridge "feline_concept".
    mock_response = {
        "bridges": {
            "1": ["feline_concept"] 
        }
    }
    
    mock_client = MockModelClient({
        "dreaming": json.dumps(mock_response)
    })
    # Inject mock client
    worker._model_client = mock_client
    
    # Mock SEARCH
    # We want engine.tier_manager.search to return 'related-memory-2' when searching for 'feline_concept'.
    original_search = engine.tier_manager.search
    
    async def mock_search(query_vec, top_k=5, time_range=None):
        logger.info(f"Mock Search invoked. Returning {related_node.id}")
        # Return list of (id, score)
        return [(related_node.id, 0.95)]
        
    # Patch the method on the INSTANCE
    engine.tier_manager.search = mock_search
    
    logger.info("Starting Dream Cycle...")
    
    try:
        # Run the cycle directly
        result = await worker._enhanced_dreaming_cycle()
        
        logger.info(f"Cycle Result: {result.output}")
        
    except Exception as e:
        logger.exception("Error during dream cycle")
        
    # VERIFICATION
    # Check if bind_memories was called -> check synapse index
    logger.info("Verifying Synapse Creation...")
    
    # The engine uses _synapse_index internally
    synapse = engine._synapse_index.get(weak_node.id, related_node.id)
    
    if synapse:
        logger.success(f"SUCCESS: Synapse found between {weak_node.id} and {related_node.id}")
        logger.info(f"Synapse Strength: {synapse.strength}")
    else:
        logger.error(f"FAILURE: No synapse found between {weak_node.id} and {related_node.id}")
        # Check if any synapse exists
        logger.info(f"Total synapses in index: {len(engine._synapse_index)}")
    
    # Verify file persistence
    if os.path.exists(cfg.paths.synapses_file):
        with open(cfg.paths.synapses_file, "r") as f:
            content = f.read()
            # The file contains JSONL lines
            if weak_node.id in content and related_node.id in content:
                logger.success("SUCCESS: Synapse persisted to file.")
            else:
                logger.warning("WARNING: Synapse file exists but IDs not found (might require a save trigger).")
                # Force save to be sure
                await engine._save_synapses()
                with open(cfg.paths.synapses_file, "r") as f2:
                     content2 = f2.read()
                     if weak_node.id in content2:
                         logger.success("SUCCESS: Synapse persisted after explicit save.")
                     else:
                         logger.error("FAILURE: Synapse still not in file.")
    else:
        logger.error("FAILURE: Synapse file not created.")

    # Cleanup
    await engine.close()
    if test_dir.exists():
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    asyncio.run(run_verification())
