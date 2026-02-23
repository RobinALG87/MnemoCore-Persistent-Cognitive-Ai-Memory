"""
Benchmark for MnemoCore with up to 100k memories.

LEGACY: This file is kept for backward compatibility.
New benchmarks should use the comprehensive suite in benchmarks/.

Measures:
- actual HAIMEngine.store() latency (P50, P95, P99)
- actual HAIMEngine.query() latency (P50, P95, P99)
- HDV primitive latency (P99)
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.engine import HAIMEngine
from mnemocore.core.config import reset_config


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(int(len(sorted_values) * pct), len(sorted_values) - 1)
    return sorted_values[idx]


def _ms_stats(samples: List[float]) -> Dict[str, float]:
    return {
        "count": float(len(samples)),
        "mean_ms": mean(samples) if samples else 0.0,
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
    }


def generate_contents(count: int) -> List[str]:
    print(f"Generating {count:,} memory payloads...")
    return [f"benchmark memory #{i:06d} with signal {i % 97}" for i in range(count)]


async def measure_store_latency(engine: HAIMEngine, contents: List[str]) -> Dict[str, float]:
    print(f"Measuring store() latency on {len(contents):,} real calls...")
    latencies_ms: List[float] = []
    for i, content in enumerate(contents):
        start = time.perf_counter()
        await engine.store(content, metadata={"benchmark": True, "index": i})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return _ms_stats(latencies_ms)


async def measure_query_latency(
    engine: HAIMEngine, queries: List[str], top_k: int = 5
) -> Dict[str, float]:
    print(f"Measuring query() latency on {len(queries):,} real calls...")
    latencies_ms: List[float] = []
    for query_text in queries:
        start = time.perf_counter()
        await engine.query(query_text, top_k=top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return _ms_stats(latencies_ms)


def measure_hdv_operations(dimension: int, n_samples: int = 10000) -> Dict[str, Dict[str, float]]:
    print(f"Measuring HDV operations ({n_samples:,} samples)...")
    v1 = BinaryHDV.random(dimension)
    v2 = BinaryHDV.random(dimension)

    bind_times = []
    permute_times = []
    distance_times = []

    for _ in range(n_samples):
        start = time.perf_counter()
        v1.xor_bind(v2)
        bind_times.append((time.perf_counter() - start) * 1_000_000)

        start = time.perf_counter()
        v1.permute(1)
        permute_times.append((time.perf_counter() - start) * 1_000_000)

        start = time.perf_counter()
        v1.hamming_distance(v2)
        distance_times.append((time.perf_counter() - start) * 1_000_000)

    return {
        "bind": {"p99_us": _percentile(bind_times, 0.99), "mean_us": mean(bind_times)},
        "permute": {"p99_us": _percentile(permute_times, 0.99), "mean_us": mean(permute_times)},
        "distance": {"p99_us": _percentile(distance_times, 0.99), "mean_us": mean(distance_times)},
    }


async def run_benchmark(args: argparse.Namespace) -> None:
    os.environ["HAIM_DIMENSIONALITY"] = str(args.dimension)
    reset_config()

    engine = HAIMEngine()
    await engine.initialize()
    try:
        contents = generate_contents(args.n_memories)

        print()
        print("=" * 72)
        print("HAIMEngine store/query benchmark")
        print("=" * 72)

        store_sample = contents[: args.store_samples]
        store_stats = await measure_store_latency(engine, store_sample)

        query_count = min(args.query_samples, len(store_sample))
        query_inputs = [f"signal {(i * 7) % 97}" for i in range(query_count)]
        query_stats = await measure_query_latency(engine, query_inputs, top_k=args.top_k)

        hdv_stats = measure_hdv_operations(args.dimension, args.hdv_samples)

        print()
        print(f"{'Metric':<32} {'Mean':<14} {'P50':<14} {'P95':<14} {'P99':<14}")
        print("-" * 90)
        print(
            f"{'store() latency (ms)':<32} "
            f"{store_stats['mean_ms']:<14.3f} {store_stats['p50_ms']:<14.3f} "
            f"{store_stats['p95_ms']:<14.3f} {store_stats['p99_ms']:<14.3f}"
        )
        print(
            f"{'query() latency (ms)':<32} "
            f"{query_stats['mean_ms']:<14.3f} {query_stats['p50_ms']:<14.3f} "
            f"{query_stats['p95_ms']:<14.3f} {query_stats['p99_ms']:<14.3f}"
        )

        print()
        print(f"{'HDV op':<20} {'Mean (us)':<16} {'P99 (us)':<16}")
        print("-" * 54)
        for op, stats in hdv_stats.items():
            print(f"{op:<20} {stats['mean_us']:<16.2f} {stats['p99_us']:<16.2f}")

        print()
        print("=" * 72)
        print("SLO Check")
        print("=" * 72)
        print(
            f"store() P99 < 50ms: {'PASS' if store_stats['p99_ms'] < 50 else 'FAIL'} "
            f"({store_stats['p99_ms']:.3f}ms)"
        )
        print(
            f"query() P99 < 50ms: {'PASS' if query_stats['p99_ms'] < 50 else 'FAIL'} "
            f"({query_stats['p99_ms']:.3f}ms)"
        )
    finally:
        await engine.close()
        reset_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HAIMEngine store/query performance")
    parser.add_argument("--dimension", type=int, default=1024, help="HDV dimensionality")
    parser.add_argument("--n-memories", type=int, default=100000, help="Dataset size label")
    parser.add_argument(
        "--store-samples", type=int, default=5000, help="Number of real store() calls"
    )
    parser.add_argument(
        "--query-samples", type=int, default=1000, help="Number of real query() calls"
    )
    parser.add_argument("--hdv-samples", type=int, default=10000, help="HDV primitive sample count")
    parser.add_argument("--top-k", type=int, default=5, help="top_k for query() benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
