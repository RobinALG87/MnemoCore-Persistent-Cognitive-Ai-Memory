"""
Benchmark runner for MnemoCore.

Orchestrates running all benchmarks and generating reports.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import click
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BenchmarkConfig, BenchmarkResult, save_result
from .latency import LatencyBenchmark, benchmark_latency_quick
from .throughput import ThroughputBenchmark, benchmark_throughput_quick
from .memory_footprint import MemoryFootprintBenchmark, benchmark_footprint_quick
from .comparison import ComparisonBenchmark
from .regression import RegressionDetector, RegressionReport


@dataclass
class BenchmarkSuiteResult:
    """Result from running a complete benchmark suite."""

    name: str
    timestamp: str
    duration_sec: float
    results: List[BenchmarkResult] = field(default_factory=list)
    regression_reports: List[RegressionReport] = field(default_factory=list)
    passed: bool = True
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "duration_sec": self.duration_sec,
            "results": [r.to_dict() for r in self.results],
            "regression_reports": [
                {
                    "passed": r.passed,
                    "summary": r.summary,
                    "timestamp": r.timestamp,
                    "alerts_count": len(r.alerts),
                }
                for r in self.regression_reports
            ],
            "passed": self.passed,
            "summary": self.summary,
        }


class BenchmarkRunner:
    """
    Main benchmark runner for MnemoCore.

    Orchestrates:
    - Running individual benchmarks
    - Collecting results
    - Regression detection
    - Report generation
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        enable_regression: bool = True,
        baseline_dir: str = "benchmark_results/baselines",
    ):
        self.config = config
        self.enable_regression = enable_regression
        self.regression_detector = RegressionDetector(baseline_dir=baseline_dir)
        if enable_regression:
            self.regression_detector.load_baselines()

    async def run_all(self) -> BenchmarkSuiteResult:
        """Run all benchmarks in the suite."""
        logger.info("Starting full benchmark suite...")
        start_time = time.time()

        results: List[BenchmarkResult] = []
        regression_reports: List[RegressionReport] = []

        # Latency benchmark
        logger.info("-" * 72)
        logger.info("Running Latency Benchmark...")
        logger.info("-" * 72)
        try:
            latency_bench = LatencyBenchmark(self.config)
            latency_result = await latency_bench.run()
            results.append(latency_result)

            if self.enable_regression:
                report = self.regression_detector.detect_regression(latency_result)
                regression_reports.append(report)
                self.regression_detector.print_report(report)
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")

        # Throughput benchmark
        logger.info("-" * 72)
        logger.info("Running Throughput Benchmark...")
        logger.info("-" * 72)
        try:
            throughput_bench = ThroughputBenchmark(self.config)
            throughput_result = await throughput_bench.run()
            results.append(throughput_result)

            if self.enable_regression:
                report = self.regression_detector.detect_regression(throughput_result)
                regression_reports.append(report)
                self.regression_detector.print_report(report)
        except Exception as e:
            logger.error(f"Throughput benchmark failed: {e}")

        # Memory footprint benchmark
        logger.info("-" * 72)
        logger.info("Running Memory Footprint Benchmark...")
        logger.info("-" * 72)
        try:
            footprint_bench = MemoryFootprintBenchmark(self.config)
            footprint_result = await footprint_bench.run()
            results.append(footprint_result)

            if self.enable_regression:
                report = self.regression_detector.detect_regression(footprint_result)
                regression_reports.append(report)
                self.regression_detector.print_report(report)
        except Exception as e:
            logger.error(f"Memory footprint benchmark failed: {e}")

        # Compile suite result
        duration = time.time() - start_time
        passed = all(r.passed for r in regression_reports) if regression_reports else True

        summary = self._generate_summary(results, regression_reports)

        suite_result = BenchmarkSuiteResult(
            name="full_suite",
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_sec=duration,
            results=results,
            regression_reports=regression_reports,
            passed=passed,
            summary=summary,
        )

        # Save suite result
        self._save_suite_result(suite_result)

        return suite_result

    async def run_latency(self) -> BenchmarkResult:
        """Run only latency benchmark."""
        logger.info("Running latency benchmark...")
        bench = LatencyBenchmark(self.config)
        result = await bench.run()

        if self.enable_regression:
            report = self.regression_detector.detect_regression(result)
            self.regression_detector.print_report(report)

        return result

    async def run_throughput(self) -> BenchmarkResult:
        """Run only throughput benchmark."""
        logger.info("Running throughput benchmark...")
        bench = ThroughputBenchmark(self.config)
        result = await bench.run()

        if self.enable_regression:
            report = self.regression_detector.detect_regression(result)
            self.regression_detector.print_report(report)

        return result

    async def run_footprint(self) -> BenchmarkResult:
        """Run only memory footprint benchmark."""
        logger.info("Running memory footprint benchmark...")
        bench = MemoryFootprintBenchmark(self.config)
        result = await bench.run()

        if self.enable_regression:
            report = self.regression_detector.detect_regression(result)
            self.regression_detector.print_report(report)

        return result

    async def run_quick_ci(self) -> BenchmarkSuiteResult:
        """
        Run quick benchmarks suitable for CI/CD.

        Uses reduced dataset sizes for faster execution.
        """
        logger.info("Running quick CI benchmark suite...")
        start_time = time.time()

        # Create CI-optimized config
        ci_config = BenchmarkConfig(
            dimension=4096,
            n_memories=5000,
            n_samples=500,
            n_warmup=50,
            hot_size=1000,
            warm_size=5000,
            cold_size=10000,
            output_dir=self.config.output_dir,
        )

        results: List[BenchmarkResult] = []
        regression_reports: List[RegressionReport] = []

        # Quick latency
        try:
            latency_result = await benchmark_latency_quick(
                dimension=ci_config.dimension,
                n_samples=ci_config.n_samples,
                output_dir=ci_config.output_dir,
            )
            results.append(latency_result)

            if self.enable_regression:
                report = self.regression_detector.detect_regression(latency_result)
                regression_reports.append(report)
                self.regression_detector.print_report(report)
        except Exception as e:
            logger.error(f"Quick latency benchmark failed: {e}")

        # Quick throughput
        try:
            throughput_result = await benchmark_throughput_quick(
                dimension=ci_config.dimension,
                batch_size=50,
                output_dir=ci_config.output_dir,
            )
            results.append(throughput_result)

            if self.enable_regression:
                report = self.regression_detector.detect_regression(throughput_result)
                regression_reports.append(report)
                self.regression_detector.print_report(report)
        except Exception as e:
            logger.error(f"Quick throughput benchmark failed: {e}")

        # Quick footprint
        try:
            footprint_result = await benchmark_footprint_quick(
                dimension=ci_config.dimension,
                max_scale=5000,
                output_dir=ci_config.output_dir,
            )
            results.append(footprint_result)

            if self.enable_regression:
                report = self.regression_detector.detect_regression(footprint_result)
                regression_reports.append(report)
                self.regression_detector.print_report(report)
        except Exception as e:
            logger.error(f"Quick footprint benchmark failed: {e}")

        duration = time.time() - start_time
        passed = all(r.passed for r in regression_reports) if regression_reports else True

        summary = f"CI Quick Suite: {len(results)} benchmarks completed in {duration:.2f}s"

        suite_result = BenchmarkSuiteResult(
            name="ci_quick_suite",
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_sec=duration,
            results=results,
            regression_reports=regression_reports,
            passed=passed,
            summary=summary,
        )

        self._save_suite_result(suite_result)

        return suite_result

    def _generate_summary(
        self,
        results: List[BenchmarkResult],
        regression_reports: List[RegressionReport],
    ) -> str:
        """Generate a summary of the benchmark run."""
        lines = [
            "Benchmark Suite Summary",
            "=" * 72,
        ]

        for result in results:
            lines.append(f"\n{result.category}: {result.name}")
            if result.throughput_ops:
                lines.append(f"  Throughput: {result.throughput_ops:.2f} ops/sec")
            lines.append(f"  P99 Latency: {result.p99_ms:.4f}ms")
            lines.append(f"  Mean Latency: {result.mean_ms:.4f}ms")

        if regression_reports:
            lines.append("\n" + "=" * 72)
            lines.append("Regression Detection:")
            passed_count = sum(1 for r in regression_reports if r.passed)
            lines.append(f"  {passed_count}/{len(regression_reports)} checks passed")

        return "\n".join(lines)

    def _save_suite_result(self, suite_result: BenchmarkSuiteResult) -> None:
        """Save suite result to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"suite_{suite_result.name}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(suite_result.to_dict(), f, indent=2)

        logger.info(f"Saved suite result to {filepath}")


# CLI interface
@click.group()
def cli():
    """MnemoCore Benchmark Suite CLI."""
    pass


@cli.command()
@click.option("--dimension", default=16384, help="HDV dimensionality")
@click.option("--samples", default=1000, help="Number of samples")
@click.option("--output", default="benchmark_results", help="Output directory")
@click.option("--quick", is_flag=True, help="Run quick CI benchmarks")
async def run(dimension: int, samples: int, output: str, quick: bool):
    """Run the full benchmark suite."""
    config = BenchmarkConfig(
        dimension=dimension,
        n_samples=samples,
        output_dir=output,
    )

    runner = BenchmarkRunner(config)

    if quick:
        result = await runner.run_quick_ci()
    else:
        result = await runner.run_all()

    click.echo(result.summary)

    # Exit with error code if regression detected
    if not result.passed:
        sys.exit(1)


@cli.command()
@click.option("--dimension", default=16384, help="HDV dimensionality")
@click.option("--samples", default=1000, help="Number of samples")
@click.option("--output", default="benchmark_results", help="Output directory")
async def latency(dimension: int, samples: int, output: str):
    """Run latency benchmark only."""
    config = BenchmarkConfig(
        dimension=dimension,
        n_samples=samples,
        output_dir=output,
    )
    runner = BenchmarkRunner(config)
    result = await runner.run_latency()
    click.echo(result.get_summary())


@cli.command()
@click.option("--dimension", default=16384, help="HDV dimensionality")
@click.option("--batch-size", default=100, help="Batch size for concurrent ops")
@click.option("--output", default="benchmark_results", help="Output directory")
async def throughput(dimension: int, batch_size: int, output: str):
    """Run throughput benchmark only."""
    config = BenchmarkConfig(
        dimension=dimension,
        batch_size=batch_size,
        output_dir=output,
    )
    runner = BenchmarkRunner(config)
    result = await runner.run_throughput()
    click.echo(result.get_summary())


@cli.command()
@click.option("--dimension", default=16384, help="HDV dimensionality")
@click.option("--max-scale", default=10000, help="Maximum scale to test")
@click.option("--output", default="benchmark_results", help="Output directory")
async def footprint(dimension: int, max_scale: int, output: str):
    """Run memory footprint benchmark only."""
    config = BenchmarkConfig(
        dimension=dimension,
        output_dir=output,
    )

    result = await benchmark_footprint_quick(
        dimension=dimension,
        max_scale=max_scale,
        output_dir=output,
    )
    click.echo(result.get_summary())


@cli.command()
@click.option("--baseline-dir", default="benchmark_results/baselines", help="Baseline directory")
@click.option("--result-file", required=True, help="Current result JSON file")
def detect(baseline_dir: str, result_file: str):
    """Detect regression from a result file."""
    detector = RegressionDetector(baseline_dir=baseline_dir)
    detector.load_baselines()

    from .base import load_result
    current = load_result(result_file)
    report = detector.detect_regression(current)

    detector.print_report(report)

    if not report.passed:
        sys.exit(1)


@cli.command()
@click.option("--baseline-dir", default="benchmark_results/baselines", help="Baseline directory")
@click.option("--result-dir", default="benchmark_results", help="Results directory")
def init_baselines(baseline_dir: str, result_dir: str):
    """Initialize baselines from existing results."""
    detector = RegressionDetector(baseline_dir=baseline_dir)

    result_path = Path(result_dir)
    count = 0

    for json_file in result_path.glob("*.json"):
        if "suite" in json_file.name or "baseline" in json_file.name:
            continue

        try:
            result = load_result(str(json_file))
            detector.set_baseline(result)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    click.echo(f"Created {count} baseline(s) in {baseline_dir}")


if __name__ == "__main__":
    # For direct Python execution
    import asyncio

    async def main():
        config = BenchmarkConfig(
            dimension=4096,
            n_samples=500,
            output_dir="benchmark_results",
        )
        runner = BenchmarkRunner(config)
        result = await runner.run_quick_ci()
        print(result.summary)

    asyncio.run(main())


__all__ = [
    "BenchmarkRunner",
    "BenchmarkSuiteResult",
]
