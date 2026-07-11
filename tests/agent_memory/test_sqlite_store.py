import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import timezone
from types import MappingProxyType

import pytest

from mnemocore.agent_memory import (
    MemoryKind,
    MemoryNotFoundError,
    MemoryScope,
    MemoryStatus,
    StorageError,
    ValidationError,
)
from mnemocore.agent_memory.sqlite_store import SQLiteMemoryStore


def _memory_history_schema(**overrides):
    columns = {
        "id": "id TEXT PRIMARY KEY",
        "memory_id": "memory_id TEXT NOT NULL",
        "event_id": "event_id TEXT NOT NULL",
        "action": "action TEXT NOT NULL",
        "status": "status TEXT NOT NULL",
        "details_json": "details_json TEXT NOT NULL",
        "created_at": "created_at TEXT NOT NULL",
    }
    columns.update(overrides)
    declarations = ",\n".join(f"                {value}" for value in columns.values())
    return f"""
        CREATE TABLE memory_history (
{declarations},
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (event_id) REFERENCES memory_events(id)
        );
    """


FINGERPRINT_MISMATCHES = [
    pytest.param({"id": "id TEXT"}, id="primary-key"),
    pytest.param({"memory_id": "memory_id TEXT"}, id="nullability"),
    pytest.param(
        {"details_json": "details_json BLOB NOT NULL"},
        id="declared-type",
    ),
    pytest.param(
        {"details_json": "details_json TEXT NOT NULL DEFAULT '{}'"},
        id="default",
    ),
]


@pytest.fixture
def scope():
    return MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore")


@pytest.mark.asyncio
async def test_open_creates_versioned_schema(tmp_path):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {
        "memory_events",
        "memories",
        "memory_evidence",
        "memory_relations",
        "memory_history",
        "memory_fts",
    } <= tables


@pytest.mark.asyncio
async def test_open_releases_schema_connection(tmp_path):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    path.unlink()

    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE memory_events SET payload_json = '{\"changed\":true}' WHERE id = 'event-1'",
        "DELETE FROM memory_events WHERE id = 'event-1'",
    ],
    ids=["update", "delete"],
)
async def test_memory_events_ledger_rejects_mutation(tmp_path, mutation):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            INSERT INTO memory_events (
                id, scope_key, tenant_id, user_id, agent_id, event_type,
                payload_json, occurred_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "event-1",
                "scope-1",
                "local",
                "robin",
                "codex",
                "remembered",
                '{"original":true}',
                "2026-07-10T00:00:00Z",
                "2026-07-10T00:00:00Z",
            ),
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="memory_events is immutable"):
            conn.execute(mutation)

        event = conn.execute(
            "SELECT id, payload_json FROM memory_events WHERE id = 'event-1'"
        ).fetchone()

    assert event == ("event-1", '{"original":true}')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper_sql", "expected_problem"),
    [
        (
            "ALTER TABLE memories DROP COLUMN updated_at",
            "memories columns",
        ),
        (
            "DROP INDEX ux_memory_events_scope_idempotency",
            "ux_memory_events_scope_idempotency",
        ),
        (
            """
            DROP INDEX ux_memory_events_scope_idempotency;
            CREATE UNIQUE INDEX ux_memory_events_scope_idempotency
                ON memory_events(scope_key, idempotency_key);
            """,
            "ux_memory_events_scope_idempotency",
        ),
        (
            """
            DROP TABLE memory_history;
            CREATE TABLE memory_history (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """,
            "memory_history foreign keys",
        ),
        (
            "DROP TRIGGER IF EXISTS trg_memory_events_immutable_update",
            "trg_memory_events_immutable_update",
        ),
        (
            """
            DROP TABLE memory_fts;
            CREATE TABLE memory_fts (memory_id TEXT, content TEXT);
            """,
            "memory_fts must be an FTS5 virtual table",
        ),
        (
            """
            DROP TABLE memory_fts;
            CREATE VIRTUAL TABLE memory_fts USING fts5(
                memory_id UNINDEXED, content, tokenize='porter'
            );
            """,
            "memory_fts must use unicode61",
        ),
    ],
    ids=[
        "columns",
        "missing-index",
        "non-partial-idempotency-index",
        "foreign-keys",
        "immutable-trigger",
        "fts5-module",
        "fts5-tokenizer",
    ],
)
async def test_open_rejects_malformed_version_one_schema(
    tmp_path, tamper_sql, expected_problem
):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(tamper_sql)

    with pytest.raises(StorageError, match=expected_problem) as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)


@pytest.mark.asyncio
async def test_version_zero_conflict_rolls_back_without_blessing_schema(tmp_path):
    path = tmp_path / "memory.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE memory_events (id TEXT PRIMARY KEY)")
        conn.commit()

    with pytest.raises(StorageError, match="memory_events columns") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert tables == {"memory_events"}


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", FINGERPRINT_MISMATCHES)
async def test_open_rejects_version_one_table_fingerprint_mismatch(
    tmp_path, overrides
):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("DROP TABLE memory_history;")
        conn.executescript(_memory_history_schema(**overrides))

    with pytest.raises(StorageError, match="memory_history table_info") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", FINGERPRINT_MISMATCHES)
async def test_version_zero_table_fingerprint_conflict_rolls_back(
    tmp_path, overrides
):
    path = tmp_path / "memory.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(_memory_history_schema(**overrides))

    with pytest.raises(StorageError, match="memory_history table_info") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert tables == {"memory_history"}


@pytest.mark.asyncio
async def test_open_rejects_fts_with_indexed_memory_id(tmp_path):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            DROP TABLE memory_fts;
            CREATE VIRTUAL TABLE memory_fts USING fts5(
                memory_id, content, tokenize='unicode61'
            );
            """
        )

    with pytest.raises(StorageError, match="memory_id must be UNINDEXED") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)


@pytest.mark.asyncio
async def test_remember_is_persistent_and_idempotent(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    first = await store.remember(
        scope,
        "Use BM25 candidate union",
        idempotency_key="decision-1",
    )
    second = await store.remember(
        scope,
        "Use BM25 candidate union",
        idempotency_key="decision-1",
    )
    assert second.id == first.id
    assert len(await store.list(scope)) == 1
    await store.close()

    reopened = await SQLiteMemoryStore.open(path)
    assert (await reopened.get(scope, first.id)).content == "Use BM25 candidate union"
    history = await reopened.history(scope, first.id)
    assert [entry.action for entry in history] == ["remembered"]
    await reopened.close()


@pytest.mark.asyncio
async def test_remember_returns_same_canonical_metadata_on_create_retry_and_reopen(
    tmp_path, scope
):
    path = tmp_path / "memory.db"
    metadata = MappingProxyType(
        {"nested": MappingProxyType({"items": ("first", {"answer": 42})})}
    )
    store = await SQLiteMemoryStore.open(path)

    created = await store.remember(
        scope,
        "Canonical return value",
        metadata=metadata,
        idempotency_key="canonical-return",
    )
    retried = await store.remember(
        scope,
        "Ignored retry payload",
        idempotency_key="canonical-return",
    )
    await store.close()

    reopened = await SQLiteMemoryStore.open(path)
    restored = await reopened.get(scope, created.id)
    await reopened.close()

    assert created == retried == restored
    assert created.metadata == {"nested": {"items": ("first", {"answer": 42})}}


@pytest.mark.asyncio
async def test_crud_is_exact_scope_and_idempotency_is_scope_local(tmp_path, scope):
    store = await SQLiteMemoryStore.open(tmp_path / "memory.db")
    foreign_scope = MemoryScope(
        user_id="other",
        agent_id=scope.agent_id,
        project_id=scope.project_id,
    )
    local = await store.remember(scope, "Local memory", idempotency_key="same-key")
    foreign = await store.remember(
        foreign_scope,
        "Foreign memory",
        idempotency_key="same-key",
    )

    assert local.id != foreign.id
    assert [record.id for record in await store.list(scope)] == [local.id]
    assert [record.id for record in await store.list(foreign_scope)] == [foreign.id]
    with pytest.raises(MemoryNotFoundError):
        await store.get(foreign_scope, local.id)
    with pytest.raises(MemoryNotFoundError):
        await store.history(foreign_scope, local.id)
    await store.close()


@pytest.mark.asyncio
async def test_forget_tombstones_and_removes_from_recall(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(scope, "Never repeat this failed migration")

    forgotten = await store.forget(scope, record.id, reason="incorrect")

    assert forgotten.status is MemoryStatus.FORGOTTEN
    assert forgotten.updated_at > record.updated_at
    assert await store.recall(scope, "failed migration") == []
    with pytest.raises(MemoryNotFoundError):
        await store.get(scope, record.id)
    assert await store.get(scope, record.id, include_forgotten=True) == forgotten
    history = await store.history(scope, record.id)
    assert [entry.action.value for entry in history] == ["remembered", "forgotten"]
    assert history[-1].details == {"reason": "incorrect"}
    await store.close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_forget_is_idempotent_for_an_already_forgotten_memory(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(scope, "Forget exactly once")

    first = await store.forget(scope, record.id, reason="first reason")
    repeated = await store.forget(scope, record.id, reason="replacement reason")

    assert repeated == first
    assert [entry.details for entry in await store.history(scope, record.id)] == [
        {},
        {"reason": "first reason"},
    ]
    await store.close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT count(*) FROM memory_events WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT count(*) FROM memory_history WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT count(*) FROM memory_lifecycle WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 2


@pytest.mark.asyncio
async def test_forget_requires_exact_scope_without_side_effects(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(scope, "Private scoped memory")
    foreign_scope = MemoryScope(
        user_id="other",
        agent_id=scope.agent_id,
        project_id=scope.project_id,
    )

    with pytest.raises(MemoryNotFoundError):
        await store.forget(foreign_scope, record.id, reason="not authorized")

    assert (await store.get(scope, record.id)).status is MemoryStatus.ACTIVE
    assert [result.memory.id for result in await store.recall(scope, "private scoped")] == [
        record.id
    ]
    await store.close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT count(*) FROM memory_events WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM memory_history WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 1


@pytest.mark.asyncio
async def test_forget_rolls_back_ledger_and_projections_on_failure(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(scope, "Remain searchable after rollback")
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            CREATE TRIGGER fail_forgotten_history_insert
            BEFORE INSERT ON memory_history
            WHEN NEW.action = 'forgotten'
            BEGIN
                SELECT RAISE(ABORT, 'forced forget failure');
            END;
            """
        )

    with pytest.raises(StorageError, match="forced forget failure") as raised:
        await store.forget(scope, record.id, reason="must roll back")
    assert isinstance(raised.value.__cause__, sqlite3.Error)
    assert (await store.get(scope, record.id)).status is MemoryStatus.ACTIVE
    assert [result.memory.id for result in await store.recall(scope, "remain searchable")] == [
        record.id
    ]
    await store.close()

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT count(*) FROM memory_events WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM memory_history WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_id = ?",
            (record.id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            """
            SELECT status, known_to FROM memory_lifecycle
            WHERE memory_id = ? ORDER BY known_from
            """,
            (record.id,),
        ).fetchall() == [("active", None)]


@pytest.mark.asyncio
async def test_remember_and_forget_maintain_lifecycle_immediately(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(
        scope,
        "Lifecycle projection",
        valid_from="2026-07-01T00:00:00Z",
        valid_to="2026-08-01T00:00:00Z",
    )
    with closing(sqlite3.connect(path)) as conn:
        remembered = conn.execute(
            """
            SELECT status, known_from, known_to, valid_from, valid_to, event_id,
                   created_at
            FROM memory_lifecycle WHERE memory_id = ?
            """,
            (record.id,),
        ).fetchone()
        remembered_event = conn.execute(
            """
            SELECT occurred_at, id, created_at FROM memory_events
            WHERE memory_id = ? AND event_type = 'remembered'
            """,
            (record.id,),
        ).fetchone()
    assert remembered == (
        "active",
        remembered_event[0],
        None,
        "2026-07-01T00:00:00.000000Z",
        "2026-08-01T00:00:00.000000Z",
        remembered_event[1],
        remembered_event[2],
    )

    await store.forget(scope, record.id, reason="expired")
    with closing(sqlite3.connect(path)) as conn:
        lifecycle = conn.execute(
            """
            SELECT status, known_from, known_to, valid_from, valid_to, event_id,
                   created_at
            FROM memory_lifecycle WHERE memory_id = ? ORDER BY known_from
            """,
            (record.id,),
        ).fetchall()
        forgotten_event = conn.execute(
            """
            SELECT occurred_at, id, created_at FROM memory_events
            WHERE memory_id = ? AND event_type = 'forgotten'
            """,
            (record.id,),
        ).fetchone()
    assert lifecycle == [
        (
            "active",
            remembered_event[0],
            forgotten_event[0],
            "2026-07-01T00:00:00.000000Z",
            "2026-08-01T00:00:00.000000Z",
            remembered_event[1],
            remembered_event[2],
        ),
        (
            "forgotten",
            forgotten_event[0],
            None,
            "2026-07-01T00:00:00.000000Z",
            "2026-08-01T00:00:00.000000Z",
            forgotten_event[1],
            forgotten_event[2],
        ),
    ]
    before_rebuild = lifecycle
    assert await store.rebuild(scope) == 1
    with closing(sqlite3.connect(path)) as conn:
        after_rebuild = conn.execute(
            """
            SELECT status, known_from, known_to, valid_from, valid_to, event_id,
                   created_at
            FROM memory_lifecycle WHERE memory_id = ? ORDER BY known_from
            """,
            (record.id,),
        ).fetchall()
    assert after_rebuild == before_rebuild
    await store.close()


@pytest.mark.asyncio
async def test_remember_serializes_immutable_json_and_utc_timestamps(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    metadata = MappingProxyType(
        {
            "outer": MappingProxyType(
                {"items": ("first", MappingProxyType({"answer": 42}))}
            )
        }
    )
    record = await store.remember(
        scope,
        "Canonical metadata",
        metadata=metadata,
        observed_at="2026-07-10T12:00:00+02:00",
        valid_from="2026-07-10T09:00:00Z",
        valid_to="2026-07-11T09:00:00+00:00",
    )
    await store.close()

    reopened = await SQLiteMemoryStore.open(path)
    restored = await reopened.get(scope, record.id)
    assert restored.metadata == {"outer": {"items": ("first", {"answer": 42})}}
    assert restored.observed_at.tzinfo is timezone.utc
    assert restored.observed_at.isoformat() == "2026-07-10T10:00:00+00:00"
    await reopened.close()

    with closing(sqlite3.connect(path)) as conn:
        stored = conn.execute(
            """
            SELECT memories.metadata_json, memories.observed_at,
                   memory_events.payload_json
            FROM memories
            JOIN memory_events ON memory_events.memory_id = memories.id
            WHERE memories.id = ?
            """,
            (record.id,),
        ).fetchone()
    assert stored[:2] == (
        '{"outer":{"items":["first",{"answer":42}]}}',
        "2026-07-10T10:00:00.000000Z",
    )
    payload = json.loads(stored[2])
    assert payload["memory_id"] == record.id
    assert payload["observation"] == {
        "kind": "observation",
        "content": "Canonical metadata",
        "metadata": {"outer": {"items": ["first", {"answer": 42}]}},
        "status": "active",
        "confidence": 1.0,
        "observed_at": "2026-07-10T10:00:00.000000Z",
        "valid_from": "2026-07-10T09:00:00.000000Z",
        "valid_to": "2026-07-11T09:00:00.000000Z",
        "created_at": record.created_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "updated_at": record.updated_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
    }


@pytest.mark.asyncio
async def test_list_filters_orders_and_validates_limit(tmp_path, scope):
    store = await SQLiteMemoryStore.open(tmp_path / "memory.db")
    first = await store.remember(scope, "First", kind=MemoryKind.FACT)
    second = await store.remember(scope, "Second", kind=MemoryKind.PROCEDURE)

    assert [record.id for record in await store.list(scope)] == [second.id, first.id]
    assert [
        record.id
        for record in await store.list(scope, kind=MemoryKind.FACT)
    ] == [first.id]
    assert [
        record.id
        for record in await store.list(scope, status=MemoryStatus.ACTIVE, limit=1)
    ] == [second.id]
    for invalid_limit in (0, 1001, True):
        with pytest.raises(ValidationError, match="limit"):
            await store.list(scope, limit=invalid_limit)
    await store.close()


@pytest.mark.asyncio
async def test_recall_is_relevant_scope_safe_and_temporal(tmp_path, scope):
    store = await SQLiteMemoryStore.open(tmp_path / "memory.db")
    await store.remember(scope, "BM25 must introduce lexical candidates")
    await store.remember(scope, "Redis stores transient queue signals")
    foreign = MemoryScope(
        user_id="other",
        agent_id="codex",
        project_id="mnemocore",
    )
    await store.remember(foreign, "BM25 private foreign memory")

    results = await store.recall(scope, "BM25 lexical", limit=5)

    assert [result.memory.content for result in results] == [
        "BM25 must introduce lexical candidates"
    ]
    assert results[0].score == 1.0
    assert results[0].score_components["bm25_rank"] == 1.0
    assert isinstance(results[0].score_components["bm25_raw"], float)
    assert results[0].reason == "Matched lexical terms in the authorized scope"
    await store.close()


@pytest.mark.asyncio
async def test_recall_applies_point_in_time_validity_boundaries(tmp_path, scope):
    store = await SQLiteMemoryStore.open(tmp_path / "memory.db")
    await store.remember(
        scope,
        "Temporal fact expired",
        kind=MemoryKind.FACT,
        valid_to="2026-07-10T12:00:00Z",
    )
    await store.remember(
        scope,
        "Temporal fact future",
        kind=MemoryKind.FACT,
        valid_from="2026-07-10T12:00:01Z",
    )
    await store.remember(
        scope,
        "Temporal fact currently valid",
        kind=MemoryKind.FACT,
        valid_from="2026-07-10T12:00:00+00:00",
        valid_to="2026-07-10T13:00:00+00:00",
    )

    results = await store.recall(
        scope,
        "temporal fact",
        kinds=(MemoryKind.FACT,),
        as_of="2026-07-10T12:00:00Z",
    )

    assert [result.memory.content for result in results] == [
        "Temporal fact currently valid"
    ]
    assert results[0].memory.valid_from.tzinfo is timezone.utc
    assert results[0].memory.valid_to.tzinfo is timezone.utc
    await store.close()


@pytest.mark.asyncio
async def test_recall_orders_fractional_validity_around_whole_second_as_of(
    tmp_path, scope
):
    store = await SQLiteMemoryStore.open(tmp_path / "memory.db")
    await store.remember(
        scope,
        "Canonical boundary starts now",
        valid_from="2026-07-10T12:00:00.000000Z",
    )
    await store.remember(
        scope,
        "Canonical boundary starts later",
        valid_from="2026-07-10T12:00:00.000001Z",
    )
    await store.remember(
        scope,
        "Canonical boundary ended now",
        valid_to="2026-07-10T12:00:00.000000Z",
    )
    await store.remember(
        scope,
        "Canonical boundary ends later",
        valid_to="2026-07-10T12:00:00.000001Z",
    )

    results = await store.recall(
        scope,
        "canonical boundary",
        as_of="2026-07-10T12:00:00Z",
    )

    assert {result.memory.content for result in results} == {
        "Canonical boundary starts now",
        "Canonical boundary ends later",
    }
    await store.close()


@pytest.mark.asyncio
async def test_recall_reopens_legacy_whole_second_validity_at_canonical_as_of(
    tmp_path, scope
):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    starts_now = await store.remember(scope, "Legacy boundary starts now")
    ends_now = await store.remember(scope, "Legacy boundary ends now")
    await store.close()

    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "UPDATE memories SET valid_from = ? WHERE id = ?",
            ("2026-07-10T12:00:00Z", starts_now.id),
        )
        conn.execute(
            "UPDATE memories SET valid_to = ? WHERE id = ?",
            ("2026-07-10T12:00:00Z", ends_now.id),
        )
        conn.commit()

    reopened = await SQLiteMemoryStore.open(path)
    results = await reopened.recall(
        scope,
        "legacy boundary",
        as_of="2026-07-10T12:00:00.000000Z",
    )

    assert [result.memory.id for result in results] == [starts_now.id]
    await reopened.close()


@pytest.mark.asyncio
async def test_remember_rolls_back_every_projection_and_preserves_sqlite_cause(
    tmp_path, scope
):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            CREATE TRIGGER fail_memory_insert
            BEFORE INSERT ON memories
            BEGIN
                SELECT RAISE(ABORT, 'forced projection failure');
            END;
            """
        )

    with pytest.raises(StorageError, match="forced projection failure") as raised:
        await store.remember(scope, "Must roll back")
    assert isinstance(raised.value.__cause__, sqlite3.Error)
    await store.close()

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM memory_events").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM memory_history").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM memory_fts").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(
            {"nested": MappingProxyType({1: "not-a-string-key"})},
            id="non-string-key",
        ),
        pytest.param({"number": float("nan")}, id="nan"),
        pytest.param({"number": float("inf")}, id="infinity"),
    ],
)
async def test_remember_rejects_noncanonical_json_without_writes(
    tmp_path, scope, metadata
):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)

    with pytest.raises(ValidationError):
        await store.remember(scope, "Invalid metadata", metadata=metadata)
    await store.close()

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM memory_events").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM memory_history").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM memory_fts").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_idempotent_retry_precedes_new_payload_validation(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    original = await store.remember(
        scope,
        "Original valid payload",
        idempotency_key="retry-before-validation",
    )

    retried = await store.remember(
        scope,
        " ",
        metadata={1: float("nan")},
        observed_at="not-a-timestamp",
        idempotency_key="retry-before-validation",
    )

    assert retried == original
    await store.close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM memory_events").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM memory_history").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM memory_fts").fetchone()[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param("kind = 'not-a-kind'", id="enum"),
        pytest.param("metadata_json = '{'", id="json"),
        pytest.param("observed_at = 'not-a-timestamp'", id="timestamp"),
        pytest.param("content = ' '", id="model-validation"),
    ],
)
async def test_get_wraps_corrupt_stored_rows_with_path_and_cause(
    tmp_path, scope, corruption
):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(scope, "Valid before corruption")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"UPDATE memories SET {corruption} WHERE id = ?", (record.id,))
        conn.commit()

    with pytest.raises(StorageError) as raised:
        await store.get(scope, record.id)

    assert str(path.resolve()) in str(raised.value)
    assert raised.value.__cause__ is not None
    await store.close()


@pytest.mark.asyncio
async def test_history_wraps_corrupt_stored_rows_with_path_and_cause(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(scope, "Valid history before corruption")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "UPDATE memory_history SET details_json = '{' WHERE memory_id = ?",
            (record.id,),
        )
        conn.commit()

    with pytest.raises(StorageError) as raised:
        await store.history(scope, record.id)

    assert str(path.resolve()) in str(raised.value)
    assert raised.value.__cause__ is not None
    await store.close()


@pytest.mark.asyncio
async def test_rebuild_restores_projection_and_fts_from_immutable_events(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    active = await store.remember(
        scope,
        "Rebuild this durable observation",
        kind=MemoryKind.PREFERENCE,
        metadata={"nested": {"items": ["one"]}},
        confidence=0.8,
        observed_at="2026-07-10T10:00:00Z",
        valid_from="2026-07-10T09:00:00Z",
        valid_to="2026-07-11T09:00:00Z",
    )
    forgotten = await store.remember(scope, "Rebuild forgotten observation")
    forgotten = await store.forget(scope, forgotten.id, reason="expired")
    history_ids_before = [
        entry.id for entry in await store.history(scope, forgotten.id)
    ]
    foreign_scope = MemoryScope(user_id="other", agent_id="codex")
    foreign = await store.remember(foreign_scope, "Foreign projection remains")

    with closing(sqlite3.connect(path)) as conn:
        remembered_payload = conn.execute(
            "SELECT payload_json FROM memory_events WHERE memory_id = ? AND event_type = ?",
            (active.id, "remembered"),
        ).fetchone()[0]
        assert '"observation"' in remembered_payload
        conn.execute(
            "UPDATE memories SET content = 'corrupt', metadata_json = '{}' WHERE id = ?",
            (active.id,),
        )
        conn.execute(
            "UPDATE memory_fts SET content = 'corrupt' WHERE memory_id = ?",
            (active.id,),
        )
        conn.execute("DELETE FROM memories WHERE id = ?", (forgotten.id,))
        conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (forgotten.id,))
        conn.commit()

    assert await store.rebuild(scope) == 2
    assert await store.get(scope, active.id) == active
    restored_forgotten = await store.get(
        scope, forgotten.id, include_forgotten=True
    )
    assert restored_forgotten == forgotten
    assert [entry.action.value for entry in await store.history(scope, forgotten.id)] == [
        "remembered",
        "forgotten",
    ]
    assert [entry.id for entry in await store.history(scope, forgotten.id)] == history_ids_before
    assert [
        result.memory.id
        for result in await store.recall(
            scope, "durable observation", as_of="2026-07-10T12:00:00Z"
        )
    ] == [active.id]
    assert await store.recall(scope, "forgotten") == []
    assert await store.get(foreign_scope, foreign.id) == foreign
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_id = ?", (active.id,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_id = ?", (forgotten.id,)
        ).fetchone()[0] == 0
    await store.close()


@pytest.mark.asyncio
async def test_rebuild_empty_ledger_removes_all_exact_scope_projections(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    orphan_id = "orphan-projection"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            INSERT INTO memories (
                id, scope_key, tenant_id, user_id, agent_id, project_id,
                session_id, kind, content, metadata_json, status, confidence,
                observed_at, valid_from, valid_to, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                orphan_id,
                scope.scope_key,
                scope.tenant_id,
                scope.user_id,
                scope.agent_id,
                scope.project_id,
                scope.session_id,
                "observation",
                "orphan corrupt projection",
                "{}",
                "active",
                1.0,
                "2026-07-10T10:00:00.000000Z",
                None,
                None,
                "2026-07-10T10:00:00.000000Z",
                "2026-07-10T10:00:00.000000Z",
            ),
        )
        conn.execute(
            "INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)",
            (orphan_id, "orphan corrupt projection"),
        )
        conn.execute(
            """
            INSERT INTO memory_history (
                id, memory_id, event_id, action, status, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "orphan-history",
                orphan_id,
                "missing-event",
                "remembered",
                "active",
                "{}",
                "2026-07-10T10:00:00.000000Z",
            ),
        )
        conn.commit()

    assert await store.rebuild(scope) == 0
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT count(*) FROM memories WHERE scope_key = ?", (scope.scope_key,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_id = ?", (orphan_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT count(*) FROM memory_history WHERE memory_id = ?", (orphan_id,)
        ).fetchone()[0] == 0
    await store.close()


@pytest.mark.asyncio
async def test_rebuild_rejects_cross_scope_event_before_any_mutation(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    foreign_scope = MemoryScope(user_id="other", agent_id="codex")
    foreign = await store.remember(foreign_scope, "Foreign ownership survives")

    with closing(sqlite3.connect(path)) as conn:
        source = conn.execute(
            "SELECT * FROM memory_events WHERE memory_id = ?", (foreign.id,)
        ).fetchone()
        columns = [row[1] for row in conn.execute("PRAGMA table_info(memory_events)")]
        copied = dict(zip(columns, source))
        copied.update(
            {
                "id": "crafted-cross-scope-event",
                "scope_key": scope.scope_key,
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "agent_id": scope.agent_id,
                "project_id": scope.project_id,
                "session_id": scope.session_id,
                "idempotency_key": None,
            }
        )
        conn.execute(
            f"INSERT INTO memory_events ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            tuple(copied[column] for column in columns),
        )
        conn.execute(
            """
            INSERT INTO memories (
                id, scope_key, tenant_id, user_id, agent_id, project_id,
                session_id, kind, content, metadata_json, status, confidence,
                observed_at, valid_from, valid_to, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "scope-a-orphan",
                scope.scope_key,
                scope.tenant_id,
                scope.user_id,
                scope.agent_id,
                scope.project_id,
                scope.session_id,
                "observation",
                "Must survive rejected rebuild",
                "{}",
                "active",
                1.0,
                "2026-07-10T10:00:00.000000Z",
                None,
                None,
                "2026-07-10T10:00:00.000000Z",
                "2026-07-10T10:00:00.000000Z",
            ),
        )
        conn.execute(
            "INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)",
            ("scope-a-orphan", "Must survive rejected rebuild"),
        )
        conn.commit()

    with pytest.raises(StorageError, match="foreign event scope"):
        await store.rebuild(scope)

    assert await store.get(foreign_scope, foreign.id) == foreign
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT content FROM memories WHERE id = ?", ("scope-a-orphan",)
        ).fetchone()[0] == "Must survive rejected rebuild"
        assert conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_id IN (?, ?)",
            (foreign.id, "scope-a-orphan"),
        ).fetchone()[0] == 2
    await store.close()


def _scope_database_snapshot(path, target_scope):
    with closing(sqlite3.connect(path)) as conn:
        return (
            conn.execute(
                "SELECT * FROM memory_events WHERE scope_key = ? ORDER BY id",
                (target_scope.scope_key,),
            ).fetchall(),
            conn.execute(
                "SELECT * FROM memories WHERE scope_key = ? ORDER BY id",
                (target_scope.scope_key,),
            ).fetchall(),
            conn.execute(
                """
                SELECT history.* FROM memory_history AS history
                JOIN memories AS memory ON memory.id = history.memory_id
                WHERE memory.scope_key = ? ORDER BY history.id
                """,
                (target_scope.scope_key,),
            ).fetchall(),
            conn.execute(
                """
                SELECT fts.memory_id, fts.content FROM memory_fts AS fts
                JOIN memories AS memory ON memory.id = fts.memory_id
                WHERE memory.scope_key = ? ORDER BY fts.memory_id
                """,
                (target_scope.scope_key,),
            ).fetchall(),
        )


def _insert_crafted_remembered_event(
    path,
    *,
    event_id,
    memory_id,
    scope_key,
    denormalized_scope,
    payload_memory_id,
    template_payload,
):
    payload = json.loads(template_payload)
    payload["memory_id"] = payload_memory_id
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            INSERT INTO memory_events (
                id, memory_id, scope_key, tenant_id, user_id, agent_id,
                project_id, session_id, event_type, payload_json,
                idempotency_key, occurred_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                memory_id,
                scope_key,
                denormalized_scope.tenant_id,
                denormalized_scope.user_id,
                denormalized_scope.agent_id,
                denormalized_scope.project_id,
                denormalized_scope.session_id,
                "remembered",
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                None,
                "2026-07-10T10:00:00.000000Z",
                "2026-07-10T10:00:00.000000Z",
            ),
        )
        conn.commit()


@pytest.mark.asyncio
async def test_rebuild_rejects_denormalized_event_scope_before_mutation(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    local = await store.remember(scope, "Local projection remains unchanged")
    foreign_scope = MemoryScope(user_id="other", agent_id="codex")
    foreign = await store.remember(foreign_scope, "Foreign projection remains unchanged")
    with closing(sqlite3.connect(path)) as conn:
        payload = conn.execute(
            "SELECT payload_json FROM memory_events WHERE memory_id = ?", (local.id,)
        ).fetchone()[0]
    _insert_crafted_remembered_event(
        path,
        event_id="denormalized-scope-event",
        memory_id="denormalized-scope-memory",
        scope_key=scope.scope_key,
        denormalized_scope=foreign_scope,
        payload_memory_id="denormalized-scope-memory",
        template_payload=payload,
    )
    before = (
        _scope_database_snapshot(path, scope),
        _scope_database_snapshot(path, foreign_scope),
    )

    with pytest.raises(StorageError, match="scope fields do not match"):
        await store.rebuild(scope)

    assert _scope_database_snapshot(path, scope) == before[0]
    assert _scope_database_snapshot(path, foreign_scope) == before[1]
    assert await store.get(scope, local.id) == local
    assert await store.get(foreign_scope, foreign.id) == foreign
    await store.close()


@pytest.mark.asyncio
async def test_rebuild_rejects_payload_memory_id_mismatch_before_mutation(
    tmp_path, scope
):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    local = await store.remember(scope, "Local projection remains unchanged")
    foreign_scope = MemoryScope(user_id="other", agent_id="codex")
    foreign = await store.remember(foreign_scope, "Foreign projection remains unchanged")
    with closing(sqlite3.connect(path)) as conn:
        payload = conn.execute(
            "SELECT payload_json FROM memory_events WHERE memory_id = ?", (local.id,)
        ).fetchone()[0]
    _insert_crafted_remembered_event(
        path,
        event_id="payload-mismatch-event",
        memory_id="column-memory-id",
        scope_key=scope.scope_key,
        denormalized_scope=scope,
        payload_memory_id="different-payload-memory-id",
        template_payload=payload,
    )
    before = (
        _scope_database_snapshot(path, scope),
        _scope_database_snapshot(path, foreign_scope),
    )

    with pytest.raises(StorageError, match="payload memory_id does not match"):
        await store.rebuild(scope)

    assert _scope_database_snapshot(path, scope) == before[0]
    assert _scope_database_snapshot(path, foreign_scope) == before[1]
    assert await store.get(scope, local.id) == local
    assert await store.get(foreign_scope, foreign.id) == foreign
    await store.close()


@pytest.mark.asyncio
async def test_rebuild_rejects_foreign_ledger_only_memory_id_collision(
    tmp_path, scope
):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    local = await store.remember(scope, "Local projection remains unchanged")
    foreign_scope = MemoryScope(user_id="other", agent_id="codex")
    foreign = await store.remember(foreign_scope, "Foreign ledger remains unchanged")
    with closing(sqlite3.connect(path)) as conn:
        payload = conn.execute(
            "SELECT payload_json FROM memory_events WHERE memory_id = ?", (foreign.id,)
        ).fetchone()[0]
        conn.execute("DELETE FROM memory_history WHERE memory_id = ?", (foreign.id,))
        conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (foreign.id,))
        conn.execute("DELETE FROM memories WHERE id = ?", (foreign.id,))
        conn.commit()
    _insert_crafted_remembered_event(
        path,
        event_id="foreign-ledger-collision-event",
        memory_id=foreign.id,
        scope_key=scope.scope_key,
        denormalized_scope=scope,
        payload_memory_id=foreign.id,
        template_payload=payload,
    )
    before = (
        _scope_database_snapshot(path, scope),
        _scope_database_snapshot(path, foreign_scope),
    )

    with pytest.raises(StorageError, match="foreign event scope"):
        await store.rebuild(scope)

    assert _scope_database_snapshot(path, scope) == before[0]
    assert _scope_database_snapshot(path, foreign_scope) == before[1]
    assert await store.get(scope, local.id) == local
    await store.close()


@pytest.mark.asyncio
async def test_recall_returns_exact_scope_remember_event_as_evidence(tmp_path, scope):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(scope, "Evidence-bearing lexical memory")
    foreign_scope = MemoryScope(user_id="other", agent_id="codex")
    await store.remember(foreign_scope, "Evidence-bearing lexical memory")

    result = (await store.recall(scope, "evidence lexical", limit=1))[0]

    with closing(sqlite3.connect(path)) as conn:
        expected = conn.execute(
            """
            SELECT id FROM memory_events
            WHERE memory_id = ? AND scope_key = ? AND event_type = 'remembered'
            """,
            (record.id, scope.scope_key),
        ).fetchone()[0]
    assert result.evidence_ids == (expected,)
    await store.close()


@pytest.mark.asyncio
async def test_cancelled_worker_holds_lifecycle_lock_until_thread_finishes(tmp_path):
    import threading

    store = await SQLiteMemoryStore.open(tmp_path / "memory.db")
    started = threading.Event()
    release = threading.Event()

    def blocking_worker(path):
        started.set()
        assert release.wait(timeout=5)
        return path

    operation = asyncio.create_task(store._run(blocking_worker))
    assert await asyncio.to_thread(started.wait, 5)
    operation.cancel()
    close = asyncio.create_task(store.close())
    await asyncio.sleep(0.05)

    assert not operation.done()
    assert not close.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    await asyncio.wait_for(close, timeout=5)
