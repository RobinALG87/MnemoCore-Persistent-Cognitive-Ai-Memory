"""
Regression detection for MnemoCore benchmarks.

Detects performance regressions by comparing current results against historical baselines.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .base import BenchmarkResult, BenchmarkConfig, load_result, save_result


@dataclass
class RegressionAlert:
    """Alert for detected regression."""

    metric_name: str
    baseline_value: float
    current_value: float
    percent_change: float
    threshold_percent: float
    severity: str  # "warning", "critical"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RegressionReport:
    """Report from regression detection."""

    passed: bool
    alerts: List[RegressionAlert]
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RegressionDetector:
    """
    Detect performance regressions by comparing against baseline.

    Features:
    - Load historical baselines from JSON files
    - Compare key metrics (latency p99, throughput, memory footprint)
    - Configurable thresholds (default 10%)
    - Generate alerts and reports
    """

    DEFAULT_THRESHOLDS = {
        "latency_p99_ms": 0.10,  # 10% increase = regression
        "latency_mean_ms": 0.15,  # 15% increase
        "throughput_ops": -0.10,  # 10% decrease = regression
        "memory_per_1k_mb": 0.20,  # 20% increase
    }

    def __init__(
        self,
        baseline_dir: str = "benchmark_results/baselines",
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.baseline_dir = Path(baseline_dir)
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.baselines: Dict[str, BenchmarkResult] = {}

        # Ensure baseline directory exists
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def load_baselines(self) -> None:
        """Load all baseline files from directory."""
        self.baselines.clear()

        if not self.baseline_dir.exists():
            logger.warning(f"Baseline directory not found: {self.baseline_dir}")
            return

        for filepath in self.baseline_dir.glob("*.json"):
            try:
                result = load_result(str(filepath))
                key = f"{result.category}_{result.name}"
                self.baselines[key] = result
                logger.info(f"Loaded baseline: {key}")
            except Exception as e:
                logger.warning(f"Failed to load baseline {filepath}: {e}")

    def set_baseline(self, result: BenchmarkResult, name: Optional[str] = None) -> None:
        """Save a result as a new baseline."""
        name = name or f"{result.category}_{result.name}"
        self.baselines[name] = result

        # Save to file
        filename = f"{name}.json"
        filepath = self.baseline_dir / filename
        with open(filepath, "w") as f:
            f.write(result.to_json())

        logger.info(f"Saved baseline: {name} to {filepath}")

    def detect_regression(
        self,
        current: BenchmarkResult,
        baseline: Optional[BenchmarkResult] = None,
    ) -> RegressionReport:
        """
        Detect regression in current result vs baseline.

        If baseline is None, uses the loaded baseline with matching category/name.
        """
        if baseline is None:
            key = f"{current.category}_{current.name}"
            baseline = self.baselines.get(key)

        if baseline is None:
            logger.warning(f"No baseline found for {key}, skipping regression check")
            return RegressionReport(
                passed=True,
                alerts=[],
                summary=f"No baseline available for {current.name}",
            )

        alerts = []

        # Check P99 latency regression
        if current.p99_ms > 0 and baseline.p99_ms > 0:
            change_pct = (current.p99_ms - baseline.p99_ms) / baseline.p99_ms
            threshold = self.thresholds.get("latency_p99_ms", 0.10)

            if change_pct > threshold:
                alerts.append(RegressionAlert(
                    metric_name="p99_latency_ms",
                    baseline_value=baseline.p99_ms,
                    current_value=current.p99_ms,
                    percent_change=change_pct * 100,
                    threshold_percent=threshold * 100,
                    severity="critical" if change_pct > threshold * 2 else "warning",
                ))

        # Check mean latency regression
        if current.mean_ms > 0 and baseline.mean_ms > 0:
            change_pct = (current.mean_ms - baseline.mean_ms) / baseline.mean_ms
            threshold = self.thresholds.get("latency_mean_ms", 0.15)

            if change_pct > threshold:
                alerts.append(RegressionAlert(
                    metric_name="mean_latency_ms",
                    baseline_value=baseline.mean_ms,
                    current_value=current.mean_ms,
                    percent_change=change_pct * 100,
                    threshold_percent=threshold * 100,
                    severity="warning",
                ))

        # Check throughput regression
        if current.throughput_ops and baseline.throughput_ops:
            change_pct = (current.throughput_ops - baseline.throughput_ops) / baseline.throughput_ops
            threshold = self.thresholds.get("throughput_ops", -0.10)

            if change_pct < threshold:  # Negative = decrease = bad
                alerts.append(RegressionAlert(
                    metric_name="throughput_ops_per_sec",
                    baseline_value=baseline.throughput_ops,
                    current_value=current.throughput_ops,
                    percent_change=change_pct * 100,
                    threshold_percent=threshold * 100,
                    severity="critical" if change_pct < threshold * 2 else "warning",
                ))

        # Check memory footprint regression
        current_mem = current.metadata.get("ram_per_1k_mb")
        baseline_mem = baseline.metadata.get("ram_per_1k_mb")
        if current_mem and baseline_mem:
            change_pct = (current_mem - baseline_mem) / baseline_mem
            threshold = self.thresholds.get("memory_per_1k_mb", 0.20)

            if change_pct > threshold:
                alerts.append(RegressionAlert(
                    metric_name="ram_per_1k_mb",
                    baseline_value=baseline_mem,
                    current_value=current_mem,
                    percent_change=change_pct * 100,
                    threshold_percent=threshold * 100,
                    severity="warning",
                ))

        # Generate summary
        passed = len(alerts) == 0

        if passed:
            summary = f"No regressions detected for {current.name}. All metrics within thresholds."
        else:
            critical_count = sum(1 for a in alerts if a.severity == "critical")
            summary = (
                f"Detected {len(alerts)} regression(s) for {current.name}: "
                f"{critical_count} critical, {len(alerts) - critical_count} warnings"
            )

        return RegressionReport(
            passed=passed,
            alerts=alerts,
            summary=summary,
        )

    def detect_all(
        self,
        current_results: List[BenchmarkResult],
    ) -> List[RegressionReport]:
        """Check all current results against their baselines."""
        reports = []
        for result in current_results:
            report = self.detect_regression(result)
            reports.append(report)
        return reports

    def print_report(self, report: RegressionReport) -> None:
        """Print regression report to console."""
        print("\n" + "=" * 72)
        print(f"REGRESSION REPORT: {report.timestamp}")
        print("=" * 72)
        print(f"Status: {'PASS' if report.passed else 'FAIL'}")
        print(f"Summary: {report.summary}")
        print()

        if report.alerts:
            print("Alerts:")
            for alert in report.alerts:
                symbol = "!!!" if alert.severity == "critical" else "!"
                print(
                    f"  {symbol} [{alert.severity.upper()}] {alert.metric_name}: "
                    f"{alert.baseline_value:.4f} -> {alert.current_value:.4f} "
                    f"({alert.percent_change:+.2f}%)"
                )
        print("=" * 72 + "\n")

    def save_report(self, report: RegressionReport, filepath: str) -> None:
        """Save regression report to JSON file."""
        data = {
            "passed": report.passed,
            "summary": report.summary,
            "timestamp": report.timestamp,
            "alerts": [
                {
                    "metric": a.metric_name,
                    "baseline": a.baseline_value,
                    "current": a.current_value,
                    "percent_change": a.percent_change,
                    "threshold": a.threshold_percent,
                    "severity": a.severity,
                }
                for a in report.alerts
            ],
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved regression report to {filepath}")


def create_baselines_from_results(
    results: List[BenchmarkResult],
    baseline_dir: str = "benchmark_results/baselines",
) -> None:
    """Create baseline files from a list of results."""
    detector = RegressionDetector(baseline_dir=baseline_dir)
    for result in results:
        detector.set_baseline(result)
    logger.info(f"Created {len(results)} baseline(s) in {baseline_dir}")


__all__ = [
    "RegressionDetector",
    "RegressionAlert",
    "RegressionReport",
    "create_baselines_from_results",
]
