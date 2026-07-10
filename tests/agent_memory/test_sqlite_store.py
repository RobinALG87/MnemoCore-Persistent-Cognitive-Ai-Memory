import sqlite3
from contextlib import closing

import pytest

from mnemocore.agent_memory import MemoryScope, StorageError
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
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
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
