"""Reproducible, local-only performance baseline for AgentMemory's SQLite API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import psutil

from mnemocore.agent_memory import AgentMemory, MemoryKind, MemoryScope

SEED = 1729
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

_WORKER_ENV_ALLOWLIST = frozenset({
    "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC",
    "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
    "LANG", "LC_ALL", "LC_CTYPE",
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
})


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    repetitions: int = 5
    corpus_size: int = 128
    warmup_operations: int = 5
    remember_operations: int = 32
    recall_operations: int = 10
    context_operations: int = 5
    seed: int = SEED

    def __post_init__(self) -> None:
        for name in (
            "repetitions",
            "corpus_size",
            "remember_operations",
            "recall_operations",
            "context_operations",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.warmup_operations < 0:
            raise ValueError("warmup_operations must be non-negative")
        if self.remember_operations > self.corpus_size:
            raise ValueError("remember_operations must not exceed corpus_size")


def build_corpus(count: int, seed: int = SEED) -> list[str]:
    """Return fixed generic text with stable hit and selectivity signals."""
    return [
        f"Generic record {seed}-{index:04d} sharedsignal "
        f"{'selectiveeven' if index % 2 == 0 else 'selectiveodd'} "
        f"unique{index:04d}"
        for index in range(count)
    ]


def corpus_sha256(corpus: list[str]) -> str:
    encoded = json.dumps(corpus, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_NETWORK_DENIAL_ENABLED = False
_NETWORK_AUDIT_INSTALLED = False
_NETWORK_AUDIT_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.sendto",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
    }
)
_PROCESS_AUDIT_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.spawn",
        "os.posix_spawn",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.startfile",
        "os.startfile/2",
    }
)


def _network_audit_hook(event: str, _args: tuple[Any, ...]) -> None:
    if not _NETWORK_DENIAL_ENABLED:
        return
    if event in _NETWORK_AUDIT_EVENTS:
        raise RuntimeError("Network access is disabled for AgentMemory benchmarks")
    if event in _PROCESS_AUDIT_EVENTS:
        raise RuntimeError("Process creation is disabled inside benchmark workers")


def _set_network_denial(enabled: bool) -> None:
    """Deny network and nested processes without blocking asyncio internals."""
    global _NETWORK_AUDIT_INSTALLED, _NETWORK_DENIAL_ENABLED
    if not _NETWORK_AUDIT_INSTALLED:
        sys.addaudithook(_network_audit_hook)
        _NETWORK_AUDIT_INSTALLED = True
    _NETWORK_DENIAL_ENABLED = enabled


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        if path.exists():
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _manifest(config: BenchmarkConfig, corpus: list[str]) -> dict[str, Any]:
    dependencies = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    dependency_files = [
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "requirements.txt",
        REPOSITORY_ROOT / "requirements-dev.txt",
    ]
    return {
        "engine": "AgentMemory",
        "storage": "SQLite",
        "telemetry": "disabled",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "cpu_count": os.cpu_count(),
        "sqlite_version": sqlite3.sqlite_version,
        "dependency_manifest_sha256": hashlib.sha256(
            "\n".join(dependencies).encode("utf-8")
        ).hexdigest(),
        "dependency_files_sha256": _hash_files(dependency_files),
        "seed": config.seed,
        "corpus_sha256": corpus_sha256(corpus),
        "corpus_count": len(corpus),
        "warmup_operations": config.warmup_operations,
        "counts": {
            "remember": config.remember_operations,
            "recall_per_scenario": config.recall_operations,
            "context": config.context_operations,
        },
    }


class _RSSSampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.002):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> _RSSSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join()


async def _timed(operation: Any) -> tuple[float, Any]:
    started = time.perf_counter_ns()
    result = await operation
    return (time.perf_counter_ns() - started) / 1_000_000, result


def _validate_checkpoint(checkpoint: dict[str, int]) -> None:
    if checkpoint["busy"] != 0 or checkpoint["checkpointed"] != checkpoint["log"]:
        raise RuntimeError(f"Incomplete SQLite WAL checkpoint: {checkpoint}")


def _checkpoint_and_pragmas(path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        names = ("journal_mode", "synchronous", "foreign_keys", "page_size", "cache_size")
        values = {name: connection.execute(f"PRAGMA {name}").fetchone()[0] for name in names}
        checkpoint_row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    checkpoint = dict(zip(("busy", "log", "checkpointed"), map(int, checkpoint_row)))
    _validate_checkpoint(checkpoint)
    return values, checkpoint


def _sqlite_sizes(path: Path) -> dict[str, int]:
    sizes = {
        "main": path.stat().st_size,
        "wal": Path(f"{path}-wal").stat().st_size if Path(f"{path}-wal").exists() else 0,
        "shm": Path(f"{path}-shm").stat().st_size if Path(f"{path}-shm").exists() else 0,
    }
    sizes["total"] = sum(sizes.values())
    return sizes


async def _worker(config: BenchmarkConfig) -> dict[str, Any]:
    corpus = build_corpus(config.corpus_size, config.seed)
    scope = MemoryScope(
        tenant_id="benchmark-tenant",
        user_id="benchmark-user",
        agent_id="benchmark-agent",
        project_id="benchmark-project",
    )
    latencies: dict[str, list[float]] = {
        name: []
        for name in (
            "remember",
            "lexical_hit",
            "lexical_miss",
            "lexical_selectivity",
            "timeline_before",
            "timeline_after",
            "context_compile",
        )
    }
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="mnemocore-agent-benchmark-") as directory:
        path = Path(directory) / "memory.db"
        memory = await AgentMemory.open(path, scope=scope)
        with _RSSSampler() as rss:
            for index in range(config.warmup_operations):
                await memory.remember(
                    f"Warmup generic record {index} warmupsignal",
                    kind=MemoryKind.OBSERVATION,
                )
            stored = []
            for content in corpus[: config.remember_operations]:
                elapsed, record = await _timed(
                    memory.remember(content, kind=MemoryKind.OBSERVATION)
                )
                latencies["remember"].append(elapsed)
                stored.append(record)

            scenario_results: dict[str, list[Any]] = {}
            for _ in range(config.recall_operations):
                elapsed, results = await _timed(memory.recall("unique0000", limit=10))
                latencies["lexical_hit"].append(elapsed)
                scenario_results["hit"] = results
                elapsed, results = await _timed(memory.recall("absenttokenzz", limit=10))
                latencies["lexical_miss"].append(elapsed)
                scenario_results["miss"] = results
                elapsed, results = await _timed(memory.recall("selectiveeven", limit=10))
                latencies["lexical_selectivity"].append(elapsed)
                scenario_results["selectivity"] = results

            source = await memory.remember(
                "Generic timeline target timelinesignal",
                kind=MemoryKind.FACT,
                valid_from="2026-01-01T00:00:00Z",
            )
            replacement = await memory.supersede(
                source.id,
                "Generic timeline replacement timelinesignal",
                effective_at="2026-02-01T00:00:00Z",
            )
            known_after = replacement.updated_at.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            )
            timeline_ok = True
            for _ in range(config.recall_operations):
                elapsed, before = await _timed(
                    memory.recall(
                        "timelinesignal",
                        valid_at="2026-01-15T00:00:00Z",
                        known_at=known_after,
                    )
                )
                latencies["timeline_before"].append(elapsed)
                elapsed, after = await _timed(
                    memory.recall(
                        "timelinesignal",
                        valid_at="2026-02-15T00:00:00Z",
                        known_at=known_after,
                    )
                )
                latencies["timeline_after"].append(elapsed)
                timeline_ok &= [item.memory.id for item in before] == [source.id]
                timeline_ok &= [item.memory.id for item in after] == [replacement.id]

            packs = []
            for _ in range(config.context_operations):
                elapsed, pack = await _timed(
                    memory.compile_context("sharedsignal", token_budget=512)
                )
                latencies["context_compile"].append(elapsed)
                packs.append(pack)

            checks = {
                "lexical_hit": bool(scenario_results["hit"])
                and scenario_results["hit"][0].memory.id == stored[0].id,
                "lexical_miss": scenario_results["miss"] == [],
                "lexical_selectivity": 1 < len(scenario_results["selectivity"]) <= 10,
                "timeline_valid_known": timeline_ok,
                "context_receipted": all(
                    any(
                        item.receipt.evidence_ids
                        for level in (pack.core, pack.working, pack.episodic, pack.semantic, pack.procedural)
                        for item in level
                    )
                    for pack in packs
                ),
            }
        await memory.close()
        pragmas, checkpoint = _checkpoint_and_pragmas(path)
        sizes = _sqlite_sizes(path)
        await asyncio.get_running_loop().shutdown_default_executor()
        return {
            "latency_ms": latencies,
            "semantic_checks": checks,
            "result_counts": {
                "lexical_hit": len(scenario_results["hit"]),
                "lexical_miss": len(scenario_results["miss"]),
                "lexical_selectivity": len(scenario_results["selectivity"]),
            },
            "sqlite_bytes": sizes,
            "rss_bytes": {
                "baseline": rss.baseline,
                "peak": rss.peak,
                "delta": max(0, rss.peak - rss.baseline),
            },
            "sqlite_pragmas": pragmas,
            "sqlite_checkpoint": checkpoint,
        }


async def _network_denied_worker(config: BenchmarkConfig) -> dict[str, Any]:
    """Install denial after asyncio has created its private wakeup socket."""
    _set_network_denial(True)
    return await _worker(config)


def _nearest_rank(values: list[float] | list[int], percentile: float) -> float | int:
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percentile) - 1]


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario in runs[0]["latency_ms"]:
        values = [sample for run in runs for sample in run["latency_ms"][scenario]]
        summary[scenario] = {
            "samples": len(values),
            "mean_ms": mean(values),
            "median_ms": median(values),
            "p95_ms": _nearest_rank(values, 0.95),
            "p99_ms": _nearest_rank(values, 0.99),
        }
    return summary


def _resource_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("rss_bytes", "sqlite_bytes"):
        result[family] = {}
        for metric in runs[0][family]:
            values = [run[family][metric] for run in runs]
            result[family][metric] = {
                "samples": len(values),
                "min": min(values),
                "mean": mean(values),
                "median": median(values),
                "max": max(values),
            }
    return result


def run_baseline(config: BenchmarkConfig, *, output_path: Path | None = None) -> dict[str, Any]:
    corpus = build_corpus(config.corpus_size, config.seed)
    runs = []
    worker_environment = {
        key: value
        for key, value in os.environ.items()
        if key in _WORKER_ENV_ALLOWLIST or key.startswith("PYTHON")
    }
    worker_environment["MNEMOCORE_TELEMETRY"] = "0"
    worker_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_environment["PYTHONIOENCODING"] = "utf-8"
    for _ in range(config.repetitions):
        completed = subprocess.run(
            [sys.executable, "-m", "benchmarks.agent_memory_baseline", "--worker"],
            cwd=REPOSITORY_ROOT,
            input=json.dumps(asdict(config)),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=worker_environment,
            check=True,
        )
        runs.append(json.loads(completed.stdout))
    manifest = _manifest(config, corpus)
    manifest["sqlite_pragmas"] = runs[0]["sqlite_pragmas"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "agent_memory_sqlite",
        "config": asdict(config),
        "manifest": manifest,
        "runs": runs,
        "summary": _summary(runs),
        "resource_summary": _resource_summary(runs),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/agent_memory.json"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        config = BenchmarkConfig(**json.loads(sys.stdin.read()))
        print(json.dumps(asyncio.run(_network_denied_worker(config)), separators=(",", ":")))
        return 0
    config = (
        BenchmarkConfig(
            repetitions=1,
            corpus_size=8,
            warmup_operations=1,
            remember_operations=3,
            recall_operations=2,
            context_operations=1,
        )
        if args.smoke
        else BenchmarkConfig(repetitions=args.repetitions)
    )
    result = run_baseline(config, output_path=args.output)
    print(json.dumps({"output": str(args.output), "runs": len(result["runs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
