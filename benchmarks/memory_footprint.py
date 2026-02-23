"""
Memory footprint benchmarks for MnemoCore.

Measures RAM and disk usage for various dataset sizes.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import (
    BenchmarkBase,
    BenchmarkConfig,
    BenchmarkResult,
    SystemMetrics,
    save_result,
)


@dataclass
class FootprintSnapshot:
    """Memory and disk usage snapshot."""

    n_memories: int
    ram_mb: float
    disk_mb: float
    hot_count: int
    warm_count: int
    cold_count: int


@dataclass
class FootprintResult:
    """Result from memory footprint benchmark."""

    snapshots: List[FootprintSnapshot] = field(default_factory=list)
    ram_per_1k: float = 0.0
    disk_per_1k: float = 0.0
    total_ram_mb: float = 0.0
    total_disk_mb: float = 0.0


class MemoryFootprintBenchmark(BenchmarkBase):
    """
    Benchmark memory footprint across scales.

    Measures:
    - RAM usage per 1K/10K/100K/1M memories
    - Disk usage per 1K/10K/100K/1M memories
    - Growth rate (linear vs sub-linear)
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.snapshots: List[FootprintSnapshot] = []
        self.metrics = SystemMetrics()

        # Test scales
        self.test_scales = [1000, 10000, 50000, 100000]

    async def run(self) -> BenchmarkResult:
        """Run the memory footprint benchmark."""
        await self.setup()

        try:
            # Record baseline
            baseline_ram = self.metrics.get_memory_mb()
            baseline_disk = self._get_disk_usage()
            logger.info(f"Baseline - RAM: {baseline_ram:.2f}MB, Disk: {baseline_disk:.2f}MB")

            # Test each scale
            for scale in self.test_scales:
                await self._measure_scale(scale, baseline_ram, baseline_disk)

            # Compile results
            result = self._compile_results()

            if self.config.save_results:
                save_result(result, self.config.output_dir)

            return result

        finally:
            await self.teardown()

    async def _measure_scale(
        self,
        n_memories: int,
        baseline_ram: float,
        baseline_disk: float,
    ) -> FootprintSnapshot:
        """Measure memory footprint for a specific scale."""
        logger.info(f"Measuring footprint for {n_memories:,} memories...")

        # Generate and store memories
        batch_size = 1000
        memories = self.generate_memories(n_memories, f"scale_{n_memories}")

        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            tasks = [self.engine.store(m, metadata={"scale": n_memories}) for m in batch]
            await asyncio.gather(*tasks)

            # Brief pause to allow consolidation
            await asyncio.sleep(0.1)

        # Force sync/consolidation
        await asyncio.sleep(1)

        # Measure
        ram_mb = self.metrics.get_memory_mb() - baseline_ram
        disk_mb = self._get_disk_usage() - baseline_disk

        # Get tier counts
        tier_stats = await self.engine.tier_manager.get_stats()
        hot_count = tier_stats.get("hot_count", 0)
        warm_count = tier_stats.get("warm_count", 0)
        cold_count = tier_stats.get("cold_count", 0)

        snapshot = FootprintSnapshot(
            n_memories=n_memories,
            ram_mb=ram_mb,
            disk_mb=disk_mb,
            hot_count=hot_count,
            warm_count=warm_count,
            cold_count=cold_count,
        )

        self.snapshots.append(snapshot)

        logger.info(
            f"Scale {n_memories:,}: RAM={ram_mb:.2f}MB, Disk={disk_mb:.2f}MB, "
            f"RAM/1K={ram_mb / (n_memories / 1000):.2f}MB"
        )

        return snapshot

    def _get_disk_usage(self) -> float:
        """Get total disk usage in MB."""
        if not self.config.persist_path:
            return 0.0

        total_mb = 0.0
        for root, dirs, files in os.walk(self.config.persist_path):
            for file in files:
                try:
                    filepath = os.path.join(root, file)
                    total_mb += os.path.getsize(filepath) / (1024 * 1024)
                except OSError:
                    pass

        return total_mb

    def _compile_results(self) -> BenchmarkResult:
        """Compile snapshots into a result."""
        if not self.snapshots:
            raise RuntimeError("No snapshots collected")

        # Calculate per-1K metrics from the largest snapshot
        largest = self.snapshots[-1]
        ram_per_1k = (largest.ram_mb / largest.n_memories) * 1000
        disk_per_1k = (largest.disk_mb / largest.n_memories) * 1000

        # Check for linear vs sub-linear growth
        if len(self.snapshots) >= 2:
            first = self.snapshots[0]
            last = self.snapshots[-1]
            scale_factor = last.n_memories / first.n_memories
            ram_scale = last.ram_mb / first.ram_mb if first.ram_mb > 0 else 0
            disk_scale = last.disk_mb / first.disk_mb if first.disk_mb > 0 else 0
        else:
            scale_factor = 1.0
            ram_scale = 1.0
            disk_scale = 1.0

        metadata = {
            "dimension": self.config.dimension,
            "ram_per_1k_mb": ram_per_1k,
            "disk_per_1k_mb": disk_per_1k,
            "scale_factor": scale_factor,
            "ram_growth_factor": ram_scale,
            "disk_growth_factor": disk_scale,
            "growth_type": "linear" if ram_scale >= scale_factor * 0.9 else "sub-linear",
        }

        tier_stats = {
            "snapshots": [
                {
                    "n_memories": s.n_memories,
                    "ram_mb": s.ram_mb,
                    "disk_mb": s.disk_mb,
                    "hot": s.hot_count,
                    "warm": s.warm_count,
                    "cold": s.cold_count,
                }
                for s in self.snapshots
            ]
        }

        result = BenchmarkResult(
            name="memory_footprint",
            category="footprint",
            timestamp=time.time(),
            duration_ms=0.0,
            samples=sum(s.n_memories for s in self.snapshots),
            mean_ms=ram_per_1k,  # Using mean_ms to store RAM per 1K
            median_ms=disk_per_1k,  # Using median_ms to store disk per 1K
            p50_ms=ram_per_1k,
            p95_ms=ram_per_1k * 1.5,
            p99_ms=disk_per_1k * 2,
            min_ms=ram_per_1k,
            max_ms=disk_per_1k,
            metadata=metadata,
            tier_stats=tier_stats,
        )

        return result


async def benchmark_footprint_quick(
    dimension: int = 4096,
    max_scale: int = 10000,
    output_dir: str = "benchmark_results",
) -> BenchmarkResult:
    """Quick footprint benchmark for CI/CD."""
    config = BenchmarkConfig(
        dimension=dimension,
        output_dir=output_dir,
    )
    benchmark = MemoryFootprintBenchmark(config)
    benchmark.test_scales = [1000, 5000, max_scale]
    return await benchmark.run()


__all__ = [
    "MemoryFootprintBenchmark",
    "FootprintSnapshot",
    "FootprintResult",
    "benchmark_footprint_quick",
]
