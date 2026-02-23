"""
Throughput benchmarks for MnemoCore.

Measures operations per second under concurrent load.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import (
    BenchmarkBase,
    BenchmarkConfig,
    BenchmarkResult,
    compute_percentiles,
    save_result,
)


@dataclass
class ThroughputSample:
    """Single throughput measurement sample."""

    duration_sec: float
    operations: int
    ops_per_sec: float


@dataclass
class ConcurrentThroughputResult:
    """Result from concurrent throughput benchmark."""

    workers: int
    duration_sec: float
    total_operations: int
    ops_per_sec: float
    avg_latency_ms: float
    p99_latency_ms: float
    errors: int = 0


class ThroughputBenchmark(BenchmarkBase):
    """
    Benchmark throughput under concurrent load.

    Measures:
    - Sequential store throughput
    - Concurrent store throughput (varying worker counts)
    - Query throughput under load
    - Mixed workload throughput
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.throughput_results: Dict[str, ConcurrentThroughputResult] = {}

    async def run(self) -> BenchmarkResult:
        """Run the full throughput benchmark suite."""
        await self.setup()

        try:
            await self.warmup()

            # Test different concurrency levels
            worker_counts = [1, 5, 10, 20, 50]

            all_latencies = []
            total_ops = 0
            start_time = time.time()

            for workers in worker_counts:
                logger.info(f"Testing with {workers} concurrent workers...")
                result = await self._benchmark_concurrent_stores(workers)
                self.throughput_results[f"{workers}_workers"] = result
                all_latencies.extend(result.latencies)
                total_ops += result.total_operations

            total_duration = time.time() - start_time

            # Compile result
            result = self._compile_results(total_ops, total_duration, all_latencies)

            if self.config.save_results:
                save_result(result, self.config.output_dir)

            return result

        finally:
            await self.teardown()

    async def _benchmark_concurrent_stores(
        self,
        n_workers: int,
    ) -> ConcurrentThroughputResult:
        """Benchmark concurrent store operations."""
        batch_size = self.config.batch_size
        total_ops = n_workers * batch_size

        # Generate memories
        memories = self.generate_memories(total_ops, f"concurrent_{n_workers}")

        # Worker function
        async def store_worker(worker_id: int, start_idx: int, count: int) -> List[float]:
            latencies = []
            for i in range(count):
                content = memories[start_idx + i]
                t0 = time.perf_counter()
                try:
                    await self.engine.store(
                        content,
                        metadata={"worker": worker_id, "index": i}
                    )
                except Exception as e:
                    logger.warning(f"Worker {worker_id} store {i} failed: {e}")
                latencies.append((time.perf_counter() - t0) * 1000.0)
            return latencies

        # Run workers
        start_time = time.time()
        tasks = []
        for i in range(n_workers):
            start_idx = i * batch_size
            task = asyncio.create_task(store_worker(i, start_idx, batch_size))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.time() - start_time

        # Collect latencies
        all_latencies: List[float] = []
        errors = 0

        for r in results:
            if isinstance(r, Exception):
                errors += 1
            elif isinstance(r, list):
                all_latencies.extend(r)

        # Compute stats
        stats = compute_percentiles(all_latencies)

        throughput_result = ConcurrentThroughputResult(
            workers=n_workers,
            duration_sec=duration,
            total_operations=total_ops,
            ops_per_sec=total_ops / duration if duration > 0 else 0.0,
            avg_latency_ms=stats["mean"],
            p99_latency_ms=stats["p99"],
            errors=errors,
        )
        throughput_result.latencies = all_latencies

        logger.info(
            f"{n_workers} workers: {throughput_result.ops_per_sec:.2f} ops/sec, "
            f"P99 latency: {throughput_result.p99_latency_ms:.3f}ms"
        )

        return throughput_result

    async def _benchmark_concurrent_queries(
        self,
        n_workers: int,
        n_queries: int,
    ) -> ConcurrentThroughputResult:
        """Benchmark concurrent query operations."""
        queries = self.generate_queries(n_queries)

        async def query_worker(worker_id: int, queries: List[str]) -> List[float]:
            latencies = []
            for q in queries:
                t0 = time.perf_counter()
                try:
                    await self.engine.query(q, top_k=5)
                except Exception as e:
                    logger.debug(f"Query failed: {e}")
                latencies.append((time.perf_counter() - t0) * 1000.0)
            return latencies

        # Split queries among workers
        queries_per_worker = n_queries // n_workers
        tasks = []
        for i in range(n_workers):
            start = i * queries_per_worker
            end = start + queries_per_worker if i < n_workers - 1 else n_queries
            worker_queries = queries[start:end]
            tasks.append(query_worker(i, worker_queries))

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        all_latencies: List[float] = []
        for r in results:
            all_latencies.extend(r)

        stats = compute_percentiles(all_latencies)

        return ConcurrentThroughputResult(
            workers=n_workers,
            duration_sec=duration,
            total_operations=n_queries,
            ops_per_sec=n_queries / duration if duration > 0 else 0.0,
            avg_latency_ms=stats["mean"],
            p99_latency_ms=stats["p99"],
        )

    def _compile_results(
        self,
        total_ops: int,
        total_duration: float,
        all_latencies: List[float],
    ) -> BenchmarkResult:
        """Compile results into a BenchmarkResult."""
        stats = compute_percentiles(all_latencies)

        # Find best throughput
        best_throughput = 0.0
        best_config = None
        for config, result in self.throughput_results.items():
            if result.ops_per_sec > best_throughput:
                best_throughput = result.ops_per_sec
                best_config = config

        metadata = {
            "dimension": self.config.dimension,
            "worker_counts_tested": list(self.throughput_results.keys()),
            "best_throughput_config": best_config,
            "best_throughput_ops_per_sec": best_throughput,
        }

        throughput_by_workers = {}
        for config, result in self.throughput_results.items():
            throughput_by_workers[config] = {
                "ops_per_sec": result.ops_per_sec,
                "avg_latency_ms": result.avg_latency_ms,
                "p99_latency_ms": result.p99_latency_ms,
            }

        result = BenchmarkResult(
            name="concurrent_throughput",
            category="throughput",
            timestamp=time.time(),
            duration_ms=total_duration * 1000,
            samples=total_ops,
            mean_ms=stats["mean"],
            median_ms=stats["median"],
            p50_ms=stats["p50"],
            p95_ms=stats["p95"],
            p99_ms=stats["p99"],
            min_ms=stats["min"],
            max_ms=stats["max"],
            stdev_ms=stats["stdev"],
            throughput_ops=total_ops / total_duration if total_duration > 0 else 0.0,
            metadata=metadata,
            tier_stats={"throughput_by_workers": throughput_by_workers},
        )

        return result


async def benchmark_throughput_quick(
    dimension: int = 4096,
    batch_size: int = 100,
    output_dir: str = "benchmark_results",
) -> BenchmarkResult:
    """Quick throughput benchmark for CI/CD."""
    config = BenchmarkConfig(
        dimension=dimension,
        batch_size=batch_size,
        n_warmup=50,
        output_dir=output_dir,
    )
    benchmark = ThroughputBenchmark(config)
    return await benchmark.run()


__all__ = [
    "ThroughputBenchmark",
    "ThroughputSample",
    "ConcurrentThroughputResult",
    "benchmark_throughput_quick",
]
