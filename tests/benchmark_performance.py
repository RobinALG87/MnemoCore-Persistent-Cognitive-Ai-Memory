import asyncio
import time
import tracemalloc

import numpy as np

from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.engine import HAIMEngine
from mnemocore.core.node import MemoryNode
from mnemocore.core.synapse import SynapticConnection


async def benchmark_synapse_lookup():
    print("\n--- Benchmarking Synapse Lookup ---")
    engine = HAIMEngine(dimension=4096)  # Smaller dim for speed

    # Create artificial synapses
    print("Generating 10,000 synapses...")
    node_id = "test_node_center"

    # 100 connections for the center node
    for i in range(100):
        neighbor = f"neighbor_{i}"
        key = tuple(sorted([node_id, neighbor]))
        syn = SynapticConnection(node_id, neighbor, initial_strength=0.5)
        engine.synapses[key] = syn

        # Populate adjacency manually for benchmark
        if node_id not in engine.synapse_adjacency:
            engine.synapse_adjacency[node_id] = []
        if neighbor not in engine.synapse_adjacency:
            engine.synapse_adjacency[neighbor] = []
        engine.synapse_adjacency[node_id].append(syn)
        engine.synapse_adjacency[neighbor].append(syn)

    # 9,900 irrelevant connections
    for i in range(9900):
        id_a = f"noise_a_{i}"
        id_b = f"noise_b_{i}"
        key = tuple(sorted([id_a, id_b]))
        syn = SynapticConnection(id_a, id_b, initial_strength=0.1)
        engine.synapses[key] = syn

        if id_a not in engine.synapse_adjacency:
            engine.synapse_adjacency[id_a] = []
        if id_b not in engine.synapse_adjacency:
            engine.synapse_adjacency[id_b] = []
        engine.synapse_adjacency[id_a].append(syn)
        engine.synapse_adjacency[id_b].append(syn)

    print(f"Total synapses: {len(engine.synapses)}")

    start_time = time.time()
    iterations = 1000
    for _ in range(iterations):
        await engine.get_node_boost(node_id)

    total_time = time.time() - start_time
    print(f"Time for {iterations} lookups: {total_time:.4f}s")
    print(f"Avg time per lookup: {total_time / iterations * 1000:.4f}ms")
    with open("benchmark_results.txt", "a") as f:
        f.write(f"Synapse Lookup Time: {total_time:.4f}s\n")
        f.write(f"Avg time per lookup: {total_time / iterations * 1000:.4f}ms\n")
    return total_time


async def benchmark_vector_allocations():
    print("\n--- Benchmarking Vector Allocations ---")
    engine = HAIMEngine(dimension=16384)
    # Populate HOT tier slightly

    print("Populating HOT tier with 50 nodes...")
    vectors = []
    for i in range(50):
        vec = BinaryHDV.random(engine.dimension)
        node = MemoryNode(id=f"node_{i}", hdv=vec, content=f"content_{i}")
        await engine.tier_manager.add_memory(node)

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        await engine._current_context_vector(sample_n=50)

    total_time = time.time() - start_time
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = end_snapshot.compare_to(start_snapshot, "lineno")
    print(f"Time for {iterations} context calcs: {total_time:.4f}s")

    with open("benchmark_results.txt", "a") as f:
        f.write(f"Vector Context Time: {total_time:.4f}s\n")
        f.write("Top memory allocations:\n")
        for stat in stats[:3]:
            f.write(str(stat) + "\n")


async def main():
    # Clear file
    with open("benchmark_results.txt", "w") as f:
        f.write("Benchmark Results:\n")

    await benchmark_synapse_lookup()
    await benchmark_vector_allocations()


if __name__ == "__main__":
    asyncio.run(main())
