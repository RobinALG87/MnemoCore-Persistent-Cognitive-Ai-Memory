import os
import sys
import time
import uuid

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import asyncio
import os

from core.binary_hdv import BinaryHDV
from core.config import get_config, reset_config
from core.engine import HAIMEngine
from core.node import MemoryNode


async def benchmark_query():
    print("Initializing Engine...")
    os.environ["HAIM_ENCODING_MODE"] = "binary"
    reset_config()
    engine = HAIMEngine()
    await engine.initialize()

    # Mock data generation
    count = 2000
    print(f"Generating {count} dummy memories...")

    engine.tier_manager.hot.clear()

    dim = engine.dimension

    start_gen = time.time()
    for i in range(count):
        # Create random binary vector
        data = np.random.randint(0, 2, size=(dim // 8), dtype=np.uint8)
        hdv = BinaryHDV(data=data, dimension=dim)

        node = MemoryNode(
            id=f"mem_{i}", hdv=hdv, content=f"Dummy content {i}", metadata={}
        )
        node.tier = "hot"
        engine.tier_manager.hot[node.id] = node
        # Add to FAISS too
        engine.tier_manager._add_to_faiss(node)

    print(f"Generation took {time.time() - start_gen:.4f}s")

    # Benchmark Query
    print("Benchmarking Query...")
    query_text = "test query"

    # Warmup
    await engine.query(query_text, top_k=5)

    start_time = time.time()
    iterations = 50
    for _ in range(iterations):
        await engine.query(query_text, top_k=5)

    total_time = time.time() - start_time
    avg_time = total_time / iterations

    print(f"Average Query Time over {iterations} runs: {avg_time * 1000:.2f} ms")
    print(f"Total Time: {total_time:.4f}s")


if __name__ == "__main__":
    asyncio.run(benchmark_query())
