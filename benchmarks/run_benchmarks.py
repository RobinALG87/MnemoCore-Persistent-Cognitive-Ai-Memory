#!/usr/bin/env python3
"""
MnemoCore Benchmark Runner

Standalone script to run MnemoCore benchmarks.

Usage:
    python run_benchmarks.py                    # Run full suite
    python run_benchmarks.py --quick            # Run quick CI suite
    python run_benchmarks.py latency            # Run latency only
    python run_benchmarks.py throughput         # Run throughput only
    python run_benchmarks.py footprint          # Run memory footprint only
    python run_benchmarks.py init-baselines     # Create baselines from results
    python run_benchmarks.py detect-regression  # Check for regressions
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path so 'mnemocore' is importable
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from benchmarks.runner import BenchmarkRunner, BenchmarkConfig
from benchmarks.latency import LatencyBenchmark, benchmark_latency_quick
from benchmarks.throughput import ThroughputBenchmark, benchmark_throughput_quick
from benchmarks.memory_footprint import MemoryFootprintBenchmark, benchmark_footprint_quick
from benchmarks.regression import RegressionDetector, create_baselines_from_results
from benchmarks.base import load_result, BenchmarkResult
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="MnemoCore Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Run full benchmark suite
  %(prog)s --quick            Run quick CI benchmark suite
  %(prog)s latency            Run latency benchmark only
  %(prog)s throughput         Run throughput benchmark only
  %(prog)s footprint          Run memory footprint benchmark only
  %(prog)s init-baselines     Create baselines from current results
  %(prog)s detect             Check for regressions vs baselines
        """
    )

    # Global options
    parser.add_argument(
        "--dimension", "-d",
        type=int,
        default=16384,
        help="HDV dimensionality (default: 16384)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=1000,
        help="Number of samples per benchmark (default: 1000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    parser.add_argument(
        "--no-regression",
        action="store_true",
        help="Disable regression detection"
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="benchmark_results/baselines",
        help="Directory for baseline files (default: benchmark_results/baselines)"
    )

    # Quick mode
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick CI benchmark suite (reduced dataset)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Available benchmark commands"
    )

    # Latency subcommand
    latency_parser = subparsers.add_parser(
        "latency",
        help="Run latency benchmark only"
    )
    latency_parser.add_argument(
        "--tier",
        choices=["hot", "warm", "cold", "all"],
        default="all",
        help="Specific tier to benchmark"
    )

    # Throughput subcommand
    throughput_parser = subparsers.add_parser(
        "throughput",
        help="Run throughput benchmark only"
    )
    throughput_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of concurrent workers (default: 10)"
    )

    # Footprint subcommand
    footprint_parser = subparsers.add_parser(
        "footprint",
        help="Run memory footprint benchmark only"
    )
    footprint_parser.add_argument(
        "--max-scale",
        type=int,
        default=10000,
        help="Maximum scale to test (default: 10000)"
    )

    # Init baselines subcommand
    init_parser = subparsers.add_parser(
        "init-baselines",
        help="Create baselines from current results"
    )
    init_parser.add_argument(
        "--result-dir",
        type=str,
        default="benchmark_results",
        help="Directory containing result files"
    )

    # Detect regression subcommand
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect regressions vs baselines"
    )
    detect_parser.add_argument(
        "result_file",
        type=str,
        help="Result JSON file to check"
    )

    # Compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare MnemoCore to alternatives (placeholder)"
    )

    return parser.parse_args()


async def run_full_suite(args):
    """Run the complete benchmark suite."""
    config = BenchmarkConfig(
        dimension=args.dimension,
        n_samples=args.samples,
        output_dir=args.output,
    )

    runner = BenchmarkRunner(
        config,
        enable_regression=not args.no_regression,
        baseline_dir=args.baseline_dir,
    )

    result = await runner.run_all()

    print("\n" + "=" * 72)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 72)
    print(f"Duration: {result.duration_sec:.2f} seconds")
    print(f"Passed: {result.passed}")
    print("\n" + result.summary)

    return 0 if result.passed else 1


async def run_latency_benchmark(args):
    """Run latency benchmark."""
    print("Running latency benchmark...")
    print(f"Dimension: {args.dimension}, Samples: {args.samples}")

    config = BenchmarkConfig(
        dimension=args.dimension,
        n_samples=args.samples,
        output_dir=args.output,
    )

    runner = BenchmarkRunner(
        config,
        enable_regression=not args.no_regression,
        baseline_dir=args.baseline_dir,
    )

    result = await runner.run_latency()
    print("\n" + result.get_summary())

    return 0


async def run_throughput_benchmark(args):
    """Run throughput benchmark."""
    print("Running throughput benchmark...")
    print(f"Dimension: {args.dimension}, Workers: {args.workers}")

    config = BenchmarkConfig(
        dimension=args.dimension,
        n_samples=args.samples,
        concurrent_workers=args.workers,
        output_dir=args.output,
    )

    runner = BenchmarkRunner(
        config,
        enable_regression=not args.no_regression,
        baseline_dir=args.baseline_dir,
    )

    result = await runner.run_throughput()
    print("\n" + result.get_summary())

    return 0


async def run_footprint_benchmark(args):
    """Run memory footprint benchmark."""
    print("Running memory footprint benchmark...")
    print(f"Dimension: {args.dimension}, Max Scale: {args.max_scale}")

    result = await benchmark_footprint_quick(
        dimension=args.dimension,
        max_scale=args.max_scale,
        output_dir=args.output,
    )

    print("\n" + result.get_summary())

    return 0


def init_baselines(args):
    """Initialize baselines from existing results."""
    print(f"Creating baselines from {args.result_dir}...")

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return 1

    # Load all result files
    results = []
    for json_file in result_dir.glob("*.json"):
        if "suite" in json_file.name or "baseline" in json_file.name:
            continue
        try:
            result = load_result(str(json_file))
            results.append(result)
            print(f"  Loaded: {result.category}/{result.name}")
        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")

    if not results:
        print("No valid results found")
        return 1

    # Create baselines
    baseline_dir = Path(args.baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    from benchmarks.regression import RegressionDetector
    detector = RegressionDetector(baseline_dir=str(baseline_dir))

    for result in results:
        detector.set_baseline(result)

    print(f"\nCreated {len(results)} baseline(s) in {baseline_dir}")

    return 0


def detect_regression(args):
    """Detect regressions from a result file."""
    print(f"Checking for regressions in {args.result_file}...")

    detector = RegressionDetector(baseline_dir=args.baseline_dir)
    detector.load_baselines()

    current = load_result(args.result_file)
    report = detector.detect_regression(current)

    detector.print_report(report)

    return 0 if report.passed else 1


async def main():
    """Main entry point."""
    args = parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("=" * 72)
    print("MnemoCore Benchmark Suite")
    print("=" * 72)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    try:
        if args.quick:
            # Quick CI mode
            config = BenchmarkConfig(
                dimension=args.dimension,
                n_samples=args.samples,
                output_dir=args.output,
            )
            runner = BenchmarkRunner(
                config,
                enable_regression=not args.no_regression,
                baseline_dir=args.baseline_dir,
            )
            result = await runner.run_quick_ci()
            print("\n" + result.summary)
            return 0 if result.passed else 1

        elif args.command == "latency":
            return await run_latency_benchmark(args)

        elif args.command == "throughput":
            return await run_throughput_benchmark(args)

        elif args.command == "footprint":
            return await run_footprint_benchmark(args)

        elif args.command == "init-baselines":
            return init_baselines(args)

        elif args.command == "detect":
            return detect_regression(args)

        elif args.command == "compare":
            print("Comparison benchmarks are not yet implemented.")
            print("This will compare MnemoCore against MemGPT, Zep, and LangMem.")
            return 0

        else:
            # No subcommand, run full suite
            return await run_full_suite(args)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Benchmark failed with error")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
