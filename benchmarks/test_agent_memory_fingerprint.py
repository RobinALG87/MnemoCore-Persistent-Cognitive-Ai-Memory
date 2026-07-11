import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemocore.agent_memory import MemoryKind, MemoryRecord, MemoryScope
from mnemocore.agent_memory.fingerprint import _FINGERPRINT_BYTES, fingerprint_similarity
from mnemocore.agent_memory import sqlite_store


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def test_fingerprint_similarity_is_exact_bounded_and_repeatable():
    exact = fingerprint_similarity("stable text", "stable text")
    first = fingerprint_similarity("stable text", "different text")
    second = fingerprint_similarity("stable text", "different text")

    assert exact == 1.0
    assert _FINGERPRINT_BYTES * 8 == 16_384
    assert 0.0 <= first <= 1.0
    assert first == second


def test_fingerprint_is_deterministic_across_fresh_subprocesses():
    script = (
        "from mnemocore.agent_memory.fingerprint import fingerprint_similarity; "
        "print(fingerprint_similarity('stable text', 'different text'))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    values = [
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        for _ in range(2)
    ]

    assert values[0] == values[1]


def test_two_rerank_invocations_return_the_same_order():
    rows = [
        {"id": "a", "content": "shared alpha"},
        {"id": "b", "content": "shared beta"},
        {"id": "c", "content": "shared gamma"},
    ]

    first = sqlite_store._apply_hdv_rerank("shared alpha", rows)
    second = sqlite_store._apply_hdv_rerank("shared alpha", rows)

    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_fingerprint_failure_preserves_fts_order_and_lexical_components(monkeypatch):
    rows = [
        {"id": "first", "content": "sharedterm first", "bm25_raw": -2.0},
        {"id": "second", "content": "sharedterm second", "bm25_raw": -1.0},
    ]
    scope = MemoryScope(user_id="test-user", agent_id="test-agent")
    now = datetime.now(timezone.utc)
    records = {
        row["id"]: MemoryRecord(
            id=row["id"],
            scope=scope,
            kind=MemoryKind.OBSERVATION,
            content=row["content"],
            observed_at=now,
            created_at=now,
            updated_at=now,
        )
        for row in rows
    }

    def fail_fingerprint(_left, _right):
        raise RuntimeError("fingerprint unavailable")

    monkeypatch.setattr(sqlite_store, "fingerprint_similarity", fail_fingerprint)
    monkeypatch.setattr(
        sqlite_store,
        "_row_to_lifecycle_record",
        lambda _path, row: records[row["id"]],
    )

    reranked = sqlite_store._apply_hdv_rerank("sharedterm", rows)
    results = sqlite_store._rows_to_recall_results(
        Path("memory.db"),
        reranked,
        {"first": ("event-first",), "second": ("event-second",)},
        2,
        query="sharedterm",
        use_hdv_rerank=True,
    )

    assert reranked is rows
    assert [result.memory.id for result in results] == ["first", "second"]
    assert all("hdv" not in result.score_components for result in results)
    assert all("fusion" not in result.score_components for result in results)
