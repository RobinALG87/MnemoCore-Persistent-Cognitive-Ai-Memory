"""
Latency benchmarks for MnemoCore tiered operations.

Measures store, recall, and synthesize latency for each tier (HOT, WARM, COLD).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import (
    BenchmarkBase,
    BenchmarkConfig,
    BenchmarkResult,
    SystemMetrics,
    compute_percentiles,
    save_result,
)


@dataclass
class TierLatencyStats:
    """Latency statistics for a single tier."""

    tier: str
    store_mean_ms: float
    store_p99_ms: float
    recall_mean_ms: float
    recall_p99_ms: float
    synthesize_mean_ms: Optional[float] = None
    synthesize_p99_ms: Optional[float] = None


class LatencyBenchmark(BenchmarkBase):
    """
    Benchmark latency operations per tier.

    Measures:
    - store() latency per tier
    - query() / recall latency per tier
    - synthesize (reconstructive recall) latency
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.tier_latencies: Dict[str, TierLatencyStats] = {}

    async def run(self) -> BenchmarkResult:
        """Run the full latency benchmark suite."""
        await self.setup()

        try:
            # Warmup
            await self.warmup()

            # Measure HOT tier latency
            hot_stats = await self._measure_tier_latency("hot", self.config.hot_size)
            self.tier_latencies["hot"] = hot_stats

            # Measure WARM tier latency (if Qdrant available)
            if self.engine.tier_manager.use_qdrant:
                warm_stats = await self._measure_tier_latency("warm", self.config.warm_size)
                self.tier_latencies["warm"] = warm_stats
            else:
                logger.warning("Qdrant not available, skipping WARM tier benchmark")

            # Measure COLD tier latency
            cold_stats = await self._measure_tier_latency("cold", self.config.cold_size)
            self.tier_latencies["cold"] = cold_stats

            # Compile results
            result = self._compile_results()

            # Save if configured
            if self.config.save_results:
                save_result(result, self.config.output_dir)

            return result

        finally:
            await self.teardown()

    async def _measure_tier_latency(
        self,
        tier: str,
        target_size: int,
    ) -> TierLatencyStats:
        """Measure latency for a specific tier."""
        logger.info(f"Measuring {tier.upper()} tier latency (target size: {target_size})")

        n_samples = min(self.config.n_samples, 1000)
        store_times = []
        recall_times = []

        # Generate data
        memories = self.generate_memories(n_samples, f"{tier}_store")
        queries = self.generate_queries(min(n_samples // 2, 500))

        # Store operations
        for i, content in enumerate(memories):
            start = time.perf_counter()
            await self.engine.store(
                content,
                metadata={"tier": tier, "index": i}
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            store_times.append(elapsed_ms)

        # Force tier alignment if needed
        await self._align_to_tier(tier, target_size)

        # Query operations
        for query in queries:
            start = time.perf_counter()
            await self.engine.query(query, top_k=5)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            recall_times.append(elapsed_ms)

        # Synthesize operations (reconstructive recall)
        synthesize_times = []
        if tier in ["hot", "warm"]:  # Only for faster tiers
            synthesize_queries = queries[:min(50, len(queries))]
            for query in synthesize_queries:
                start = time.perf_counter()
                try:
                    await self.engine.reconstructive_recall(
                        query,
                        top_k=10,
                        enable_synthesis=True
                    )
                except Exception as e:
                    logger.debug(f"Synthesize failed for query: {e}")
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                synthesize_times.append(elapsed_ms)

        # Compute statistics
        store_stats = compute_percentiles(store_times)
        recall_stats = compute_percentiles(recall_times)

        stats = TierLatencyStats(
            tier=tier,
            store_mean_ms=store_stats["mean"],
            store_p99_ms=store_stats["p99"],
            recall_mean_ms=recall_stats["mean"],
            recall_p99_ms=recall_stats["p99"],
        )

        if synthesize_times:
            synth_stats = compute_percentiles(synthesize_times)
            stats.synthesize_mean_ms = synth_stats["mean"]
            stats.synthesize_p99_ms = synth_stats["p99"]

        logger.info(
            f"{tier.upper()} - Store: {stats.store_mean_ms:.3f}ms (P99: {stats.store_p99_ms:.3f}ms), "
            f"Recall: {stats.recall_mean_ms:.3f}ms (P99: {stats.recall_p99_ms:.3f}ms)"
        )

        return stats

    async def _align_to_tier(self, tier: str, target_size: int) -> None:
        """Ensure memories are in the correct tier for measurement."""
        tier_manager = self.engine.tier_manager

        if tier == "hot":
            # Add enough memories to fill HOT tier
            current_count = await tier_manager._hot_storage.count()
            if current_count < target_size:
                needed = min(target_size - current_count, 10000)
                for i in range(needed):
                    await self.engine.store(
                        f"hot_fill_{i}",
                        metadata={"fill": True}
                    )
            # Trigger eviction to WARM to get some in WARM
            if target_size < 10000:
                extra = 2000
                for i in range(extra):
                    await self.engine.store(
                        f"hot_overflow_{i}",
                        metadata={"overflow": True}
                    )

        elif tier == "warm":
            # Ensure HOT is full to trigger WARM usage
            hot_max = tier_manager.config.tiers_hot.max_memories
            for i in range(hot_max + 500):
                await self.engine.store(
                    f"warm_trigger_{i}",
                    metadata={"trigger": True}
                )

        elif tier == "cold":
            # Force consolidation to COLD
            await tier_manager.consolidate_warm_to_cold()

    def _compile_results(self) -> BenchmarkResult:
        """Compile all tier statistics into a result."""
        # Aggregate stats
        all_store_means = [s.store_mean_ms for s in self.tier_latencies.values()]
        all_recall_means = [s.recall_mean_ms for s in self.tier_latencies.values()]

        metadata = {
            "dimension": self.config.dimension,
            "tiers_tested": list(self.tier_latencies.keys()),
            "samples_per_tier": self.config.n_samples,
        }

        tier_stats = {}
        for tier, stats in self.tier_latencies.items():
            tier_stats[tier] = {
                "store_mean_ms": stats.store_mean_ms,
                "store_p99_ms": stats.store_p99_ms,
                "recall_mean_ms": stats.recall_mean_ms,
                "recall_p99_ms": stats.recall_p99_ms,
            }
            if stats.synthesize_mean_ms:
                tier_stats[tier]["synthesize_mean_ms"] = stats.synthesize_mean_ms
                tier_stats[tier]["synthesize_p99_ms"] = stats.synthesize_p99_ms

        result = BenchmarkResult(
            name="tier_latency",
            category="latency",
            timestamp=time.time(),
            duration_ms=0.0,  # Not tracking overall duration here
            samples=self.config.n_samples * len(self.tier_latencies),
            mean_ms=sum(all_store_means + all_recall_means) / len(all_store_means + all_recall_means),
            median_ms=0.0,  # Computed from percentiles
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=max(s.store_p99_ms for s in self.tier_latencies.values()),
            min_ms=min(s.store_mean_ms for s in self.tier_latencies.values()),
            max_ms=max(s.store_p99_ms for s in self.tier_latencies.values()),
            metadata=metadata,
            tier_stats=tier_stats,
        )

        # Compute percentiles properly
        all_times = []
        for stats in self.tier_latencies.values():
            all_times.extend([
                stats.store_mean_ms,
                stats.store_p99_ms,
                stats.recall_mean_ms,
                stats.recall_p99_ms,
            ])

        if all_times:
            pct = compute_percentiles(all_times)
            result.median_ms = pct["median"]
            result.p50_ms = pct["p50"]
            result.p95_ms = pct["p95"]

        return result


async def benchmark_latency_quick(
    dimension: int = 4096,
    n_samples: int = 500,
    output_dir: str = "benchmark_results",
) -> BenchmarkResult:
    """Quick latency benchmark for CI/CD."""
    config = BenchmarkConfig(
        dimension=dimension,
        n_samples=n_samples,
        n_warmup=50,
        hot_size=1000,
        warm_size=5000,
        cold_size=10000,
        output_dir=output_dir,
    )
    benchmark = LatencyBenchmark(config)
    return await benchmark.run()


__all__ = [
    "LatencyBenchmark",
    "TierLatencyStats",
    "benchmark_latency_quick",
]
