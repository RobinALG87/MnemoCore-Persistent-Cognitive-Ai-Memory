import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.agent_memory_baseline import (
    BenchmarkConfig,
    _set_network_denial,
    _nearest_rank,
    _validate_checkpoint,
    build_corpus,
    corpus_sha256,
    run_baseline,
)


def test_corpus_is_stable_generic_and_hash_addressed():
    first = build_corpus(12, seed=1729)
    second = build_corpus(12, seed=1729)

    assert first == second
    assert len(first) == 12
    assert corpus_sha256(first) == corpus_sha256(second)
    assert len(corpus_sha256(first)) == 64
    assert all(item.startswith("Generic record 1729-") for item in first)


def test_baseline_defaults_to_at_least_five_fresh_repetitions():
    config = BenchmarkConfig()

    assert config.repetitions >= 5
    assert config.seed == 1729


@pytest.mark.parametrize(
    ("values", "percentile", "expected"),
    [
        ([1, 2, 3, 4], 0.50, 2),
        ([1, 2, 3, 4], 0.95, 4),
        (list(range(1, 101)), 0.95, 95),
        (list(range(1, 101)), 0.99, 99),
    ],
)
def test_nearest_rank_uses_ceil_n_times_p_minus_one(values, percentile, expected):
    assert _nearest_rank(values, percentile) == expected


def test_worker_network_denial_rejects_outbound_socket_connect():
    _set_network_denial(True)
    try:
        with socket.socket() as client:
            with pytest.raises(RuntimeError, match="Network access is disabled"):
                client.connect(("127.0.0.1", 9))
    finally:
        _set_network_denial(False)


def test_worker_network_denial_rejects_udp_sendto():
    _set_network_denial(True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            with pytest.raises(RuntimeError, match="Network access is disabled"):
                client.sendto(b"probe", ("127.0.0.1", 9))
    finally:
        _set_network_denial(False)


@pytest.mark.parametrize(
    "resolver",
    [
        lambda: socket.gethostbyname("localhost"),
        lambda: socket.gethostbyaddr("127.0.0.1"),
        lambda: socket.getnameinfo(("127.0.0.1", 80), 0),
    ],
)
def test_worker_network_denial_rejects_dns_resolution(resolver):
    _set_network_denial(True)
    try:
        with pytest.raises(RuntimeError, match="Network access is disabled"):
            resolver()
    finally:
        _set_network_denial(False)


def test_worker_containment_rejects_nested_subprocess():
    _set_network_denial(True)
    try:
        with pytest.raises(RuntimeError, match="Process creation is disabled"):
            subprocess.run([sys.executable, "-c", "pass"], check=True)
    finally:
        _set_network_denial(False)


def test_worker_containment_rejects_os_system():
    _set_network_denial(True)
    try:
        with pytest.raises(RuntimeError, match="Process creation is disabled"):
            os.system("exit 0")
    finally:
        _set_network_denial(False)


def test_worker_containment_rejects_spawn_without_starting_a_process():
    _set_network_denial(True)
    try:
        with pytest.raises(RuntimeError, match="Process creation is disabled"):
            os.spawnv(
                os.P_NOWAIT,
                r"Z:\does-not-exist\python.exe",
                ["python.exe", "-c", "pass"],
            )
    finally:
        _set_network_denial(False)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"busy": 1, "log": 4, "checkpointed": 4},
        {"busy": 0, "log": 4, "checkpointed": 3},
    ],
)
def test_checkpoint_validation_fails_busy_or_incomplete(checkpoint):
    with pytest.raises(RuntimeError, match="Incomplete SQLite WAL checkpoint"):
        _validate_checkpoint(checkpoint)


def test_smoke_result_has_reproducible_schema_and_raw_samples(tmp_path):
    output = tmp_path / "agent-memory-smoke.json"
    config = BenchmarkConfig(
        repetitions=1,
        corpus_size=8,
        warmup_operations=1,
        remember_operations=3,
        recall_operations=2,
        context_operations=1,
    )

    result = run_baseline(config, output_path=output)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["schema_version"] == 1
    assert result["benchmark"] == "agent_memory_sqlite"
    assert result["config"]["repetitions"] == 1
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert set(run["latency_ms"]) == {
        "remember",
        "lexical_hit",
        "lexical_miss",
        "lexical_selectivity",
        "timeline_before",
        "timeline_after",
        "context_compile",
    }
    assert all(run["latency_ms"][name] for name in run["latency_ms"])
    assert run["semantic_checks"] == {
        "lexical_hit": True,
        "lexical_miss": True,
        "lexical_selectivity": True,
        "timeline_valid_known": True,
        "context_receipted": True,
    }
    assert run["result_counts"]["lexical_hit"] == 1
    assert run["result_counts"]["lexical_miss"] == 0
    assert run["result_counts"]["lexical_selectivity"] == 2
    assert set(run["sqlite_bytes"]) == {"main", "wal", "shm", "total"}
    assert run["sqlite_bytes"]["main"] > 0
    assert set(run["rss_bytes"]) == {"baseline", "peak", "delta"}
    assert run["rss_bytes"]["peak"] >= run["rss_bytes"]["baseline"]
    assert result["manifest"]["corpus_sha256"] == corpus_sha256(build_corpus(8, 1729))
    assert result["manifest"]["telemetry"] == "disabled"
    assert result["manifest"]["engine"] == "AgentMemory"
    assert result["manifest"]["sqlite_pragmas"] == run["sqlite_pragmas"]
    assert run["sqlite_pragmas"]["foreign_keys"] == 1
    assert run["sqlite_checkpoint"] == {"busy": 0, "log": 0, "checkpointed": 0}
    assert set(result["resource_summary"]) == {"rss_bytes", "sqlite_bytes"}
    assert set(result["resource_summary"]["rss_bytes"]) == {"baseline", "peak", "delta"}
    assert set(result["resource_summary"]["sqlite_bytes"]) == {
        "main",
        "wal",
        "shm",
        "total",
    }
    for family in result["resource_summary"].values():
        for metric in family.values():
            assert set(metric) == {"samples", "min", "mean", "median", "max"}
