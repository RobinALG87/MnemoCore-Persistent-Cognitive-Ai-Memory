"""
Comparison benchmark framework for MnemoCore vs alternatives.

Prepares comparison tests against MemGPT, Zep, LangMem, etc.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
class ComparisonResult:
    """Result from comparing two memory systems."""

    system_a: str
    system_b: str
    metric_name: str
    value_a: float
    value_b: float
    ratio: float  # A/B
    winner: str  # "A", "B", or "tie"


@dataclass
class ComparisonReport:
    """Report comparing MnemoCore to alternatives."""

    mnemocore_results: Dict[str, float]
    competitor_results: Dict[str, float]
    comparisons: List[ComparisonResult]
    summary: str


class MemorySystemInterface(ABC):
    """
    Abstract interface for memory system comparison.

    All systems to be compared must implement this interface.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the memory system."""
        pass

    @abstractmethod
    async def store(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Store a memory. Returns memory ID."""
        pass

    @abstractmethod
    async def recall(self, query: str, top_k: int = 5) -> List[Any]:
        """Recall memories by query."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """System name."""
        pass


class MnemoCoreAdapter(MemorySystemInterface):
    """Adapter for MnemoCore to the comparison interface."""

    def __init__(self, config: BenchmarkConfig):
        from .base import BenchmarkBase

        self.config = config
        self._engine = None

    async def initialize(self) -> None:
        from mnemocore.core.engine import HAIMEngine
        from mnemocore.core.config import reset_config
        import os

        os.environ["HAIM_DIMENSIONALITY"] = str(self.config.dimension)
        reset_config()

        self._engine = HAIMEngine(persist_path=self.config.persist_path)
        await self._engine.initialize()

    async def store(self, content: str, metadata: Optional[Dict] = None) -> str:
        node = await self._engine.store(content, metadata=metadata or {})
        return node.id

    async def recall(self, query: str, top_k: int = 5) -> List[Any]:
        results = await self._engine.query(query, top_k=top_k)
        return results

    async def close(self) -> None:
        if self._engine:
            await self._engine.close()
            from mnemocore.core.config import reset_config
            reset_config()

    @property
    def name(self) -> str:
        return "MnemoCore"


class ComparisonBenchmark(BenchmarkBase):
    """
    Benchmark framework for comparing MnemoCore to alternatives.

    Supports:
    - Side-by-side comparison of operations
    - Statistical comparison (mean, p95, p99)
    - Preparation for MemGPT, Zep, LangMem integration
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.competitors: List[MemorySystemInterface] = []
        self.comparison_results: List[ComparisonResult] = []

    def add_competitor(self, system: MemorySystemInterface) -> None:
        """Add a competitor system to compare against."""
        self.competitors.append(system)

    async def run(self) -> BenchmarkResult:
        """Run comparison benchmarks against all competitors."""
        # First, benchmark MnemoCore
        mnemocore_adapter = MnemoCoreAdapter(self.config)
        await mnemocore_adapter.initialize()

        try:
            mnemocore_metrics = await self._benchmark_system(mnemocore_adapter)

            # Benchmark each competitor
            competitor_metrics = {}
            for competitor in self.competitors:
                await competitor.initialize()
                try:
                    metrics = await self._benchmark_system(competitor)
                    competitor_metrics[competitor.name] = metrics
                finally:
                    await competitor.close()

            # Generate comparisons
            self.comparison_results = self._compare_metrics(
                mnemocore_metrics,
                competitor_metrics,
            )

            # Compile result
            result = self._compile_results(
                mnemocore_metrics,
                competitor_metrics,
            )

            if self.config.save_results:
                save_result(result, self.config.output_dir)

            return result

        finally:
            await mnemocore_adapter.close()

    async def _benchmark_system(
        self,
        system: MemorySystemInterface,
    ) -> Dict[str, float]:
        """Benchmark a single system."""
        logger.info(f"Benchmarking {system.name}...")

        n_samples = self.config.n_samples
        memories = self.generate_memories(n_samples, system.name)
        queries = self.generate_queries(min(n_samples // 2, 500))

        # Measure store latency
        store_times = []
        for i, content in enumerate(memories):
            start = time.perf_counter()
            await system.store(content, metadata={"index": i})
            store_times.append((time.perf_counter() - start) * 1000.0)

        # Measure recall latency
        recall_times = []
        for query in queries:
            start = time.perf_counter()
            await system.recall(query, top_k=5)
            recall_times.append((time.perf_counter() - start) * 1000.0)

        # Compute stats
        store_stats = compute_percentiles(store_times)
        recall_stats = compute_percentiles(recall_times)

        metrics = {
            "store_mean_ms": store_stats["mean"],
            "store_p99_ms": store_stats["p99"],
            "recall_mean_ms": recall_stats["mean"],
            "recall_p99_ms": recall_stats["p99"],
            "throughput_ops_per_sec": n_samples / (sum(store_times) / 1000.0),
        }

        logger.info(
            f"{system.name}: Store={metrics['store_mean_ms']:.3f}ms, "
            f"Recall={metrics['recall_mean_ms']:.3f}ms"
        )

        return metrics

    def _compare_metrics(
        self,
        mnemocore: Dict[str, float],
        competitors: Dict[str, List[Dict[str, float]]],
    ) -> List[ComparisonResult]:
        """Compare MnemoCore metrics against competitors."""
        results = []

        for competitor_name, competitor_metrics in competitors.items():
            for metric_name, mnemocore_value in mnemocore.items():
                competitor_value = competitor_metrics.get(metric_name, 0.0)

                if competitor_value == 0:
                    continue

                # For latency, lower is better
                # For throughput, higher is better
                if "latency" in metric_name or "ms" in metric_name:
                    winner = "MnemoCore" if mnemocore_value < competitor_value else competitor_name
                    ratio = competitor_value / mnemocore_value  # How many times faster
                else:
                    winner = "MnemoCore" if mnemocore_value > competitor_value else competitor_name
                    ratio = mnemocore_value / competitor_value

                results.append(ComparisonResult(
                    system_a="MnemoCore",
                    system_b=competitor_name,
                    metric_name=metric_name,
                    value_a=mnemocore_value,
                    value_b=competitor_value,
                    ratio=ratio,
                    winner=winner,
                ))

        return results

    def _compile_results(
        self,
        mnemocore_metrics: Dict[str, float],
        competitor_metrics: Dict[str, Dict[str, float]],
    ) -> BenchmarkResult:
        """Compile comparison results into a BenchmarkResult."""
        # Flatten all metrics for statistics
        all_values = list(mnemocore_metrics.values())
        for comp_metrics in competitor_metrics.values():
            all_values.extend(comp_metrics.values())

        stats = compute_percentiles(all_values)

        metadata = {
            "competitors_tested": list(competitor_metrics.keys()),
            "mnemocore_metrics": mnemocore_metrics,
            "competitor_metrics": competitor_metrics,
            "comparisons": [
                {
                    "vs": r.system_b,
                    "metric": r.metric_name,
                    "mnemocore": r.value_a,
                    "competitor": r.value_b,
                    "ratio": r.ratio,
                    "winner": r.winner,
                }
                for r in self.comparison_results
            ],
        }

        # Generate summary
        wins = sum(1 for r in self.comparison_results if r.winner == "MnemoCore")
        total = len(self.comparison_results)
        summary = f"MnemoCore wins {wins}/{total} comparisons"

        result = BenchmarkResult(
            name="system_comparison",
            category="comparison",
            timestamp=time.time(),
            duration_ms=0.0,
            samples=self.config.n_samples * (1 + len(competitor_metrics)),
            mean_ms=stats["mean"],
            median_ms=stats["median"],
            p50_ms=stats["p50"],
            p95_ms=stats["p95"],
            p99_ms=stats["p99"],
            min_ms=stats["min"],
            max_ms=stats["max"],
            stdev_ms=stats["stdev"],
            metadata=metadata,
            tier_stats={"summary": summary},
        )

        return result


# Placeholder adapters for future implementation
class MemGPTAdapter(MemorySystemInterface):
    """
    Placeholder adapter for MemGPT.

    TODO: Implement once MemGPT client is available.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    async def initialize(self) -> None:
        logger.warning("MemGPT adapter not yet implemented")
        raise NotImplementedError("MemGPT adapter not yet implemented")

    async def store(self, content: str, metadata: Optional[Dict] = None) -> str:
        raise NotImplementedError()

    async def recall(self, query: str, top_k: int = 5) -> List[Any]:
        raise NotImplementedError()

    async def close(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "MemGPT"


class ZepAdapter(MemorySystemInterface):
    """
    Placeholder adapter for Zep.

    TODO: Implement once Zep client is available.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    async def initialize(self) -> None:
        logger.warning("Zep adapter not yet implemented")
        raise NotImplementedError("Zep adapter not yet implemented")

    async def store(self, content: str, metadata: Optional[Dict] = None) -> str:
        raise NotImplementedError()

    async def recall(self, query: str, top_k: int = 5) -> List[Any]:
        raise NotImplementedError()

    async def close(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "Zep"


class LangMemAdapter(MemorySystemInterface):
    """
    Placeholder adapter for LangMem.

    TODO: Implement once LangMem client is available.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    async def initialize(self) -> None:
        logger.warning("LangMem adapter not yet implemented")
        raise NotImplementedError("LangMem adapter not yet implemented")

    async def store(self, content: str, metadata: Optional[Dict] = None) -> str:
        raise NotImplementedError()

    async def recall(self, query: str, top_k: int = 5) -> List[Any]:
        raise NotImplementedError()

    async def close(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "LangMem"


__all__ = [
    "ComparisonBenchmark",
    "ComparisonResult",
    "ComparisonReport",
    "MemorySystemInterface",
    "MnemoCoreAdapter",
    "MemGPTAdapter",
    "ZepAdapter",
    "LangMemAdapter",
]
