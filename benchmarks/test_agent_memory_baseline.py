import json
from pathlib import Path

from benchmarks.agent_memory_baseline import (
    BenchmarkConfig,
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
        "timeline_recall",
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
