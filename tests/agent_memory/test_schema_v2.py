import json
import sqlite3
from contextlib import closing

import pytest

from mnemocore.agent_memory import MemoryScope, MemoryStatus, StorageError
from mnemocore.agent_memory.schema import (
    SCHEMA_VERSION,
    _AUXILIARY_SCHEMA_V1,
    _TABLE_SCHEMA_V1,
    _execute_statements,
)
from mnemocore.agent_memory.sqlite_store import SQLiteMemoryStore


def _observation(memory_id, content, timestamp):
    return {
        "memory_id": memory_id,
        "observation": {
            "kind": "observation",
            "content": content,
            "metadata": {},
            "status": "active",
            "confidence": 1.0,
            "observed_at": timestamp,
            "valid_from": None,
            "valid_to": None,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    }


def _create_v1_database(path):
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="timeline")
    active_at = "2026-07-10T10:00:00.000000Z"
    forgotten_at = "2026-07-10T11:00:00.000000Z"
    forget_at = "2026-07-10T12:00:00.000000Z"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _execute_statements(connection, _TABLE_SCHEMA_V1)
        _execute_statements(connection, _AUXILIARY_SCHEMA_V1)
        for memory_id, content, remembered_at, status, updated_at in (
            ("active-memory", "Still known", active_at, "active", active_at),
            (
                "forgotten-memory",
                "Was once known",
                forgotten_at,
                "forgotten",
                forget_at,
            ),
        ):
            remembered_id = f"{memory_id}:remembered"
            connection.execute(
                """
                INSERT INTO memory_events (
                    id, memory_id, scope_key, tenant_id, user_id, agent_id,
                    project_id, session_id, event_type, payload_json,
                    idempotency_key, occurred_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    remembered_id,
                    memory_id,
                    scope.scope_key,
                    scope.tenant_id,
                    scope.user_id,
                    scope.agent_id,
                    scope.project_id,
                    scope.session_id,
                    "remembered",
                    json.dumps(
                        _observation(memory_id, content, remembered_at),
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    None,
                    remembered_at,
                    remembered_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO memories (
                    id, scope_key, tenant_id, user_id, agent_id, project_id,
                    session_id, kind, content, metadata_json, status, confidence,
                    observed_at, valid_from, valid_to, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    scope.scope_key,
                    scope.tenant_id,
                    scope.user_id,
                    scope.agent_id,
                    scope.project_id,
                    scope.session_id,
                    "observation",
                    content,
                    "{}",
                    status,
                    1.0,
                    remembered_at,
                    None,
                    None,
                    remembered_at,
                    updated_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_history (
                    id, memory_id, event_id, action, status, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{remembered_id}:history",
                    memory_id,
                    remembered_id,
                    "remembered",
                    "active",
                    "{}",
                    remembered_at,
                ),
            )
        connection.execute(
            """
            INSERT INTO memory_events (
                id, memory_id, scope_key, tenant_id, user_id, agent_id,
                project_id, session_id, event_type, payload_json,
                idempotency_key, occurred_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "forgotten-memory:forgotten",
                "forgotten-memory",
                scope.scope_key,
                scope.tenant_id,
                scope.user_id,
                scope.agent_id,
                scope.project_id,
                scope.session_id,
                "forgotten",
                '{"memory_id":"forgotten-memory","reason":"obsolete"}',
                None,
                forget_at,
                forget_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_history (
                id, memory_id, event_id, action, status, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "forgotten-memory:forgotten:history",
                "forgotten-memory",
                "forgotten-memory:forgotten",
                "forgotten",
                "forgotten",
                '{"reason":"obsolete"}',
                forget_at,
            ),
        )
        connection.execute(
            "INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)",
            ("active-memory", "Still known"),
        )
        connection.execute(
            """
            INSERT INTO memory_evidence (
                memory_id, source_memory_id, event_id, relation, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "active-memory",
                "forgotten-memory",
                "active-memory:remembered",
                "supports",
                active_at,
            ),
        )
        connection.execute("PRAGMA user_version=1")
        connection.commit()
    return scope


def _database_snapshot(path):
    with closing(sqlite3.connect(path)) as connection:
        objects = connection.execute(
            """
            SELECT type, name, tbl_name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name
            """
        ).fetchall()
        tables = [
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]
        rows = {
            table: connection.execute(f'SELECT * FROM "{table}"').fetchall()
            for table in tables
            if not table.startswith("memory_fts_")
        }
        return connection.execute("PRAGMA user_version").fetchone()[0], objects, rows


def _restore_immutable_update_trigger(connection):
    _execute_statements(connection, _AUXILIARY_SCHEMA_V1)


def _replace_remembered_payload(path, payload):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TRIGGER trg_memory_events_immutable_update")
        connection.execute(
            "UPDATE memory_events SET payload_json = ? WHERE id = ?",
            (payload, "active-memory:remembered"),
        )
        _restore_immutable_update_trigger(connection)
        connection.commit()


async def _create_database_at_version(path, version):
    if version == 1:
        _create_v1_database(path)
        return
    store = await SQLiteMemoryStore.open(path)
    await store.close()


@pytest.mark.asyncio
async def test_open_migrates_v1_and_preserves_foundation_state(tmp_path):
    path = tmp_path / "memory.db"
    scope = _create_v1_database(path)

    store = await SQLiteMemoryStore.open(path)

    assert SCHEMA_VERSION == 2
    assert (await store.get(scope, "active-memory")).content == "Still known"
    forgotten = await store.get(scope, "forgotten-memory", include_forgotten=True)
    assert forgotten.status is MemoryStatus.FORGOTTEN
    assert [entry.action.value for entry in await store.history(scope, "forgotten-memory")] == [
        "remembered",
        "forgotten",
    ]
    await store.close()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        lifecycle = connection.execute(
            """
            SELECT memory_id, status, known_from, known_to, event_id
            FROM memory_lifecycle ORDER BY memory_id, known_from
            """
        ).fetchall()
        evidence = connection.execute(
            """
            SELECT memory_id, source_memory_id, event_id, scope_key, relation
            FROM memory_evidence
            """
        ).fetchone()
    assert lifecycle == [
        (
            "active-memory",
            "active",
            "2026-07-10T10:00:00.000000Z",
            None,
            "active-memory:remembered",
        ),
        (
            "forgotten-memory",
            "active",
            "2026-07-10T11:00:00.000000Z",
            "2026-07-10T12:00:00.000000Z",
            "forgotten-memory:remembered",
        ),
        (
            "forgotten-memory",
            "forgotten",
            "2026-07-10T12:00:00.000000Z",
            None,
            "forgotten-memory:forgotten",
        ),
    ]
    assert evidence == (
        "active-memory",
        "forgotten-memory",
        "active-memory:remembered",
        scope.scope_key,
        "supports",
    )


@pytest.mark.asyncio
async def test_failed_v1_migration_is_transactional(tmp_path):
    path = tmp_path / "memory.db"
    _create_v1_database(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP INDEX ux_memory_events_scope_idempotency")
        connection.commit()
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match="ux_memory_events_scope_idempotency") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    assert _database_snapshot(path) == before


@pytest.mark.asyncio
async def test_v1_migration_rejects_projection_without_remembered_stream(tmp_path):
    path = tmp_path / "memory.db"
    _create_v1_database(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TRIGGER trg_memory_events_immutable_delete")
        connection.execute(
            "DELETE FROM memory_history WHERE memory_id = 'active-memory'"
        )
        connection.execute(
            "DELETE FROM memory_events WHERE memory_id = 'active-memory'"
        )
        _execute_statements(connection, _AUXILIARY_SCHEMA_V1)
        connection.commit()
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match="has no immutable event stream"):
        await SQLiteMemoryStore.open(path)

    assert _database_snapshot(path) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    ("object_type", "create_sql"),
    [
        ("table", "CREATE TABLE unexpected_owned_table (id TEXT)"),
        (
            "index",
            "CREATE INDEX unexpected_owned_index ON memories(content)",
        ),
        (
            "trigger",
            """
            CREATE TRIGGER unexpected_owned_trigger
            BEFORE INSERT ON memories
            BEGIN
                SELECT 1;
            END
            """,
        ),
    ],
)
async def test_schema_rejects_unexpected_owned_objects_without_mutation(
    tmp_path, version, object_type, create_sql
):
    path = tmp_path / "memory.db"
    await _create_database_at_version(path, version)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(create_sql)
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match=f"unexpected {object_type}") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    assert _database_snapshot(path) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2])
async def test_schema_rejects_changed_index_collation_without_mutation(
    tmp_path, version
):
    path = tmp_path / "memory.db"
    await _create_database_at_version(path, version)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            DROP INDEX ix_memories_scope_validity;
            CREATE INDEX ix_memories_scope_validity
                ON memories(scope_key COLLATE NOCASE, valid_from, valid_to);
            """
        )
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match="ix_memories_scope_validity") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    assert _database_snapshot(path) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2])
async def test_schema_rejects_changed_primary_key_collation_without_mutation(
    tmp_path, version
):
    path = tmp_path / "memory.db"
    await _create_database_at_version(path, version)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            DROP TABLE memory_history;
            CREATE TABLE memory_history (
                id TEXT COLLATE NOCASE PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (event_id) REFERENCES memory_events(id)
            );
            """
        )
    before = _database_snapshot(path)

    with pytest.raises(
        StorageError, match="sqlite_autoindex_memory_history_1"
    ) as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    assert _database_snapshot(path) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2])
async def test_schema_allows_documented_sqlite_analyze_tables(tmp_path, version):
    path = tmp_path / "memory.db"
    await _create_database_at_version(path, version)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("ANALYZE")
        connection.commit()

    store = await SQLiteMemoryStore.open(path)
    await store.close()

    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'"
        ).fetchone() == (1,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("kind", None),
        ("kind", "not-a-kind"),
        ("content", None),
        ("content", "   "),
        ("confidence", None),
        ("confidence", float("nan")),
        ("confidence", 1.01),
        ("metadata", []),
        ("status", None),
        ("observed_at", None),
        ("created_at", None),
        ("updated_at", None),
    ],
)
async def test_v1_migration_fully_validates_remembered_observation_before_ddl(
    tmp_path, field, invalid_value
):
    path = tmp_path / "memory.db"
    _create_v1_database(path)
    payload = _observation(
        "active-memory",
        "Still known",
        "2026-07-10T10:00:00.000000Z",
    )
    payload["observation"][field] = invalid_value
    _replace_remembered_payload(
        path,
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match="remembered event") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    assert raised.value.__cause__ is not None
    assert _database_snapshot(path) == before


@pytest.mark.asyncio
async def test_v1_migration_rejects_invalid_remembered_validity_interval(tmp_path):
    path = tmp_path / "memory.db"
    _create_v1_database(path)
    payload = _observation(
        "active-memory",
        "Still known",
        "2026-07-10T10:00:00.000000Z",
    )
    payload["observation"].update(
        {
            "valid_from": "2026-07-12T00:00:00.000000Z",
            "valid_to": "2026-07-11T00:00:00.000000Z",
        }
    )
    _replace_remembered_payload(
        path,
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match="remembered event"):
        await SQLiteMemoryStore.open(path)

    assert _database_snapshot(path) == before


@pytest.mark.asyncio
async def test_v1_migration_wraps_invalid_utf8_blob_payload_without_mutation(tmp_path):
    path = tmp_path / "memory.db"
    _create_v1_database(path)
    _replace_remembered_payload(path, sqlite3.Binary(b"\xff\xfe\x80"))
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match="invalid payload") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    assert isinstance(raised.value.__cause__, sqlite3.Error)
    assert isinstance(raised.value.__cause__.__cause__, UnicodeDecodeError)
    assert _database_snapshot(path) == before


@pytest.mark.asyncio
async def test_v1_migration_rejects_json_blob_even_when_utf8_is_valid(tmp_path):
    path = tmp_path / "memory.db"
    _create_v1_database(path)
    payload = json.dumps(
        _observation(
            "active-memory",
            "Still known",
            "2026-07-10T10:00:00.000000Z",
        ),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _replace_remembered_payload(path, sqlite3.Binary(payload))
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match="payload_json must be TEXT") as raised:
        await SQLiteMemoryStore.open(path)

    assert str(path.resolve()) in str(raised.value)
    assert _database_snapshot(path) == before


@pytest.mark.asyncio
async def test_v1_migration_rejects_blank_event_id_before_ddl(tmp_path):
    path = tmp_path / "memory.db"
    _create_v1_database(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TRIGGER trg_memory_events_immutable_update")
        connection.execute(
            "UPDATE memory_events SET id = '' WHERE id = 'active-memory:remembered'"
        )
        connection.execute(
            "UPDATE memory_history SET event_id = '' WHERE memory_id = 'active-memory'"
        )
        connection.execute(
            "UPDATE memory_evidence SET event_id = '' WHERE memory_id = 'active-memory'"
        )
        _restore_immutable_update_trigger(connection)
        connection.commit()
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match="invalid event id"):
        await SQLiteMemoryStore.open(path)

    assert _database_snapshot(path) == before


@pytest.mark.asyncio
async def test_v1_migration_hydrates_and_validates_event_scope_before_ddl(tmp_path):
    path = tmp_path / "memory.db"
    scope = _create_v1_database(path)
    invalid_scope_key = json.dumps(
        [scope.tenant_id, "   ", scope.agent_id, scope.project_id, scope.session_id],
        separators=(",", ":"),
    )
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TRIGGER trg_memory_events_immutable_update")
        connection.execute(
            """
            UPDATE memory_events SET user_id = '   ', scope_key = ?
            WHERE id = 'active-memory:remembered'
            """,
            (invalid_scope_key,),
        )
        connection.execute(
            """
            UPDATE memories SET user_id = '   ', scope_key = ?
            WHERE id = 'active-memory'
            """,
            (invalid_scope_key,),
        )
        connection.execute("DELETE FROM memory_evidence")
        _restore_immutable_update_trigger(connection)
        connection.commit()
    before = _database_snapshot(path)

    with pytest.raises(StorageError, match=r"invalid .* scope"):
        await SQLiteMemoryStore.open(path)

    assert _database_snapshot(path) == before


async def _guard_fixture(path):
    local = MemoryScope(user_id="local", agent_id="codex")
    foreign = MemoryScope(user_id="foreign", agent_id="codex")
    store = await SQLiteMemoryStore.open(path)
    local_target = await store.remember(local, "local target")
    local_source = await store.remember(local, "local source")
    foreign_memory = await store.remember(foreign, "foreign memory")
    await store.close()
    with closing(sqlite3.connect(path)) as connection:
        local_event = connection.execute(
            "SELECT id FROM memory_events WHERE memory_id = ?", (local_target.id,)
        ).fetchone()[0]
        foreign_event = connection.execute(
            "SELECT id FROM memory_events WHERE memory_id = ?", (foreign_memory.id,)
        ).fetchone()[0]
    return local, local_target.id, local_source.id, local_event, foreign_memory.id, foreign_event


@pytest.mark.asyncio
@pytest.mark.parametrize("guarded_table", ["memory_evidence", "memory_relations"])
async def test_scope_guards_accept_same_scope_and_reject_cross_scope_inserts(
    tmp_path, guarded_table
):
    path = tmp_path / "memory.db"
    local, target, source, event, foreign, foreign_event = await _guard_fixture(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        if guarded_table == "memory_evidence":
            sql = """
                INSERT INTO memory_evidence (
                    memory_id, source_memory_id, event_id, scope_key, relation, created_at
                ) VALUES (?, ?, ?, ?, 'supports', '2026-07-10T00:00:00.000000Z')
            """
            connection.execute(sql, (target, source, event, local.scope_key))
            bad_rows = [
                (foreign, source, event, local.scope_key),
                (target, foreign, event, local.scope_key),
                (target, source, foreign_event, local.scope_key),
            ]
        else:
            sql = """
                INSERT INTO memory_relations (
                    id, scope_key, source_id, target_id, relation_type,
                    valid_from, valid_to, confidence, event_id, created_at
                ) VALUES (?, ?, ?, ?, 'supports', NULL, NULL, 1.0, ?,
                          '2026-07-10T00:00:00.000000Z')
            """
            connection.execute(sql, ("same-scope", local.scope_key, source, target, event))
            bad_rows = [
                ("bad-source", local.scope_key, foreign, target, event),
                ("bad-target", local.scope_key, source, foreign, event),
                ("bad-event", local.scope_key, source, target, foreign_event),
            ]
        for row in bad_rows:
            with pytest.raises(sqlite3.IntegrityError, match="scope mismatch"):
                connection.execute(sql, row)


@pytest.mark.asyncio
@pytest.mark.parametrize("guarded_table", ["memory_evidence", "memory_relations"])
async def test_scope_guards_reject_cross_scope_updates(tmp_path, guarded_table):
    path = tmp_path / "memory.db"
    local, target, source, event, foreign, foreign_event = await _guard_fixture(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        if guarded_table == "memory_evidence":
            connection.execute(
                """
                INSERT INTO memory_evidence (
                    memory_id, source_memory_id, event_id, scope_key, relation, created_at
                ) VALUES (?, ?, ?, ?, 'supports', '2026-07-10T00:00:00.000000Z')
                """,
                (target, source, event, local.scope_key),
            )
            mutations = [
                ("memory_id", foreign),
                ("source_memory_id", foreign),
                ("event_id", foreign_event),
                ("scope_key", "not-the-owner"),
            ]
            where = "memory_id = ? AND source_memory_id = ?"
            where_values = (target, source)
        else:
            connection.execute(
                """
                INSERT INTO memory_relations (
                    id, scope_key, source_id, target_id, relation_type,
                    valid_from, valid_to, confidence, event_id, created_at
                ) VALUES ('same-scope', ?, ?, ?, 'supports', NULL, NULL, 1.0, ?,
                          '2026-07-10T00:00:00.000000Z')
                """,
                (local.scope_key, source, target, event),
            )
            mutations = [
                ("source_id", foreign),
                ("target_id", foreign),
                ("event_id", foreign_event),
                ("scope_key", "not-the-owner"),
            ]
            where = "id = ?"
            where_values = ("same-scope",)
        for column, value in mutations:
            with pytest.raises(sqlite3.IntegrityError, match="scope mismatch"):
                connection.execute(
                    f"UPDATE {guarded_table} SET {column} = ? WHERE {where}",
                    (value, *where_values),
                )
