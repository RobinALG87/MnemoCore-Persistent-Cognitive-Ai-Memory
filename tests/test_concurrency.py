import threading
import time
import random
from src.core.engine import HAIMEngine

def worker_task(engine, worker_id, num_ops=50):
    for i in range(num_ops):
        # Alternate between store and query
        if random.random() > 0.5:
            content = f"Worker {worker_id} - Operation {i} - {random.random()}"
            engine.store(content, metadata={"worker": worker_id})
        else:
            engine.query(f"something about worker {worker_id}", top_k=2)
        
        # Small delay to increase likelihood of interleaving
        time.sleep(random.uniform(0.001, 0.01))

def test_concurrency():
    print("Initializing HAIMEngine for concurrency test...")
    engine = HAIMEngine()
    
    num_workers = 10
    ops_per_worker = 100
    threads = []
    
    print(f"Starting {num_workers} workers, each doing {ops_per_worker} operations...")
    start_time = time.time()
    
    for i in range(num_workers):
        t = threading.Thread(target=worker_task, args=(engine, i, ops_per_worker))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    end_time = time.time()
    print(f"Concurrency test completed in {end_time - start_time:.2f} seconds.")
    print(f"Final HOT tier size: {len(engine.tier_manager.hot)}")
    engine.close()

if __name__ == "__main__":
    test_concurrency()
