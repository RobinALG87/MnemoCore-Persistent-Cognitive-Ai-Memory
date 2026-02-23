import asyncio
import os
import shutil
import time
import numpy as np
from pathlib import Path
from mnemocore.core.hnsw_index import HNSWIndexManager
from mnemocore.core.binary_hdv import TextEncoder, BinaryHDV
from mnemocore.core.config import get_config, reset_config

async def test_persistent_caching():
    print("\n--- Testing Persistent Vector Caching ---")
    data_dir = Path("./test_data_perf")
    data_dir.mkdir(exist_ok=True)
    cache_path = data_dir / "vector_cache.sqlite"
    
    if cache_path.exists():
        os.remove(cache_path)
        
    # Force config for testing
    os.environ["HAIM_PERFORMANCE_VECTOR_CACHE_ENABLED"] = "True"
    os.environ["HAIM_PERFORMANCE_VECTOR_CACHE_PATH"] = str(cache_path)
    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    reset_config()
    
    encoder = TextEncoder(dimension=1024) # Small dim for speed
    
    text = "Performance optimization is key to scalability."
    
    # First encoding (should be slow/normal and populate cache)
    start = time.time()
    vec1 = encoder.encode(text)
    duration1 = time.time() - start
    print(f"First encoding duration: {duration1*1000:.2f}ms")
    
    # Second encoding (should be fast - from in-memory cache)
    start = time.time()
    vec2 = encoder.encode(text)
    duration2 = time.time() - start
    print(f"Second encoding (in-memory) duration: {duration2*1000:.2f}ms")
    
    assert vec1 == vec2
    
    # Clear in-memory cache and re-initialize
    encoder._token_cache = {}
    
    # Third encoding (should be fast - from persistent cache)
    start = time.time()
    vec3 = encoder.encode(text)
    duration3 = time.time() - start
    print(f"Third encoding (persistent) duration: {duration3*1000:.2f}ms")
    
    assert vec1 == vec3
    assert duration3 < duration1
    print("Persistent caching verified.")

async def test_background_rebuild():
    print("\n--- Testing Background HNSW Rebuilding ---")
    data_dir = Path("./test_data_perf_hnsw")
    data_dir.mkdir(exist_ok=True)
    
    os.environ["HAIM_PERFORMANCE_BACKGROUND_REBUILD_ENABLED"] = "True"
    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    reset_config()
    
    # Use small threshold for testing
    from mnemocore.core import hnsw_index
    original_threshold = hnsw_index.FLAT_THRESHOLD
    hnsw_index.FLAT_THRESHOLD = 10
    
    try:
        mgr = HNSWIndexManager(dimension=1024)
        
        # Add nodes to trigger rebuild
        print("Adding 15 nodes to trigger HNSW upgrade...")
        for i in range(15):
            mgr.add(f"node_{i}", np.random.randint(0, 256, (1024//8,), dtype=np.uint8))
            
        print("Waiting for background rebuild to complete...")
        max_wait = 10
        start = time.time()
        while not mgr._use_hnsw and time.time() - start < max_wait:
            await asyncio.sleep(0.5)
            
        if mgr._use_hnsw:
            print(f"HNSW rebuild confirmed in {time.time() - start:.2f}s")
        else:
            print("FAILED: HNSW rebuild did not complete in time.")
            
        # Test remove trigger
        print("Removing nodes to trigger stale-fraction rebuild...")
        # Add more to make sure we have enough
        for i in range(20, 40):
            mgr.add(f"node_{i}", np.random.randint(0, 256, (1024//8,), dtype=np.uint8))
            
        # Remove many
        for i in range(15):
            mgr.remove(f"node_{i}")
            
        # Should be pending
        if mgr._rebuild_pending:
            print("Stale rebuild pending as expected.")
            
        # Wait for completion
        start = time.time()
        while mgr._rebuild_pending and time.time() - start < max_wait:
            await asyncio.sleep(0.5)
            
        print(f"Stale rebuild finished in {time.time() - start:.2f}s")
        
    finally:
        hnsw_index.FLAT_THRESHOLD = original_threshold
    
    print("Background HNSW rebuild verified.")

async def main():
    try:
        await test_persistent_caching()
        await test_background_rebuild()
    finally:
        # Cleanup
        for d in ["./test_data_perf", "./test_data_perf_hnsw"]:
            if os.path.exists(d):
                shutil.rmtree(d)

if __name__ == "__main__":
    asyncio.run(main())
