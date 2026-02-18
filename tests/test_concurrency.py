
import asyncio
import random
import time
from mnemocore.core.engine import HAIMEngine
from unittest.mock import patch, MagicMock
from pathlib import Path
from mnemocore.core.config import get_config

async def worker_task(engine, worker_id, num_ops=50):
    for i in range(num_ops):
        # Alternate between store and query
        if random.random() > 0.5:
             content = f"Worker {worker_id} - Operation {i} - {random.random()}"
             await engine.store(content, metadata={"worker": worker_id})
        else:
             await engine.query(f"something about worker {worker_id}", top_k=2)
        
        # Small delay to increase likelihood of interleaving
        await asyncio.sleep(random.uniform(0.001, 0.01))
    
    snap = await engine.tier_manager.get_hot_snapshot()
    # print(f"Worker {worker_id} done. Partial HOT: {len(snap)}")

async def main():
    print("Initializing HAIMEngine for concurrency test...")
    # Mock Qdrant for this test to avoid needing a real server
    with patch("qdrant_client.AsyncQdrantClient"):
        engine = HAIMEngine()
        # Ensure fallback
        engine.tier_manager.use_qdrant = False
        if not engine.tier_manager.warm_path:
             config = get_config()
             engine.tier_manager.warm_path = Path(config.paths.warm_mmap_dir)
             engine.tier_manager.warm_path.mkdir(parents=True, exist_ok=True)
             
        await engine.initialize()
        
        num_workers = 10
        ops_per_worker = 50 
        
        print(f"Starting {num_workers} workers, each doing {ops_per_worker} operations...")
        start_time = time.time()
        
        tasks = []
        for i in range(num_workers):
            tasks.append(worker_task(engine, i, ops_per_worker))
            
        await asyncio.gather(*tasks)
            
        end_time = time.time()
        print(f"Concurrency test completed in {end_time - start_time:.2f} seconds.")
        
        # Snapshot for metrics
        hot_snap = await engine.tier_manager.get_hot_snapshot()
        print(f"Final HOT tier size: {len(hot_snap)}")
        
        await engine.close()

if __name__ == "__main__":
    asyncio.run(main())
