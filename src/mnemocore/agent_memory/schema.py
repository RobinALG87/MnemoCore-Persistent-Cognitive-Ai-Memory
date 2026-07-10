"""Strict SQLite schema ownership and transactional v1-to-v2 migration."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

_TABLE_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    memory_id TEXT,
    scope_key TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    idempotency_key TEXT,
    occurred_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    scope_key TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    observed_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_evidence (
    memory_id TEXT NOT NULL,
    source_memory_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES memory_events(id)
);
CREATE TABLE IF NOT EXISTS memory_relations (
    id TEXT PRIMARY KEY,
    scope_key TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    confidence REAL NOT NULL,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES memory_events(id)
);
CREATE TABLE IF NOT EXISTS memory_history (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES memory_events(id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize='unicode61'
);
"""

_AUXILIARY_SCHEMA_V1 = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_memory_events_scope_idempotency
    ON memory_events(scope_key, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_memories_scope_status_kind_created
    ON memories(scope_key, status, kind, created_at);
CREATE INDEX IF NOT EXISTS ix_memories_scope_validity
    ON memories(scope_key, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS ix_memory_relations_scope_validity
    ON memory_relations(scope_key, valid_from, valid_to);
CREATE TRIGGER IF NOT EXISTS trg_memory_events_immutable_update
BEFORE UPDATE ON memory_events
BEGIN
    SELECT RAISE(ABORT, 'memory_events is immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_memory_events_immutable_delete
BEFORE DELETE ON memory_events
BEGIN
    SELECT RAISE(ABORT, 'memory_events is immutable');
END;
"""

_TABLE_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    memory_id TEXT,
    scope_key TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    idempotency_key TEXT,
    occurred_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    scope_key TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    observed_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_evidence (
    memory_id TEXT NOT NULL,
    source_memory_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES memory_events(id)
);
CREATE TABLE IF NOT EXISTS memory_relations (
    id TEXT PRIMARY KEY,
    scope_key TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    confidence REAL NOT NULL,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES memory_events(id)
);
CREATE TABLE IF NOT EXISTS memory_history (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES memory_events(id)
);
CREATE TABLE IF NOT EXISTS memory_lifecycle (
    memory_id TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    status TEXT NOT NULL,
    known_from TEXT NOT NULL,
    known_to TEXT,
    valid_from TEXT,
    valid_to TEXT,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(memory_id, known_from, status),
    FOREIGN KEY(memory_id) REFERENCES memories(id),
    FOREIGN KEY(event_id) REFERENCES memory_events(id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize='unicode61'
);
"""

_GUARD_TRIGGER_SQL = {
    "trg_memory_evidence_scope_insert": """
        CREATE TRIGGER IF NOT EXISTS trg_memory_evidence_scope_insert
        BEFORE INSERT ON memory_evidence
        WHEN (SELECT scope_key FROM memories WHERE id = NEW.memory_id) IS NOT NEW.scope_key
          OR (SELECT scope_key FROM memories WHERE id = NEW.source_memory_id) IS NOT NEW.scope_key
          OR (SELECT scope_key FROM memory_events WHERE id = NEW.event_id) IS NOT NEW.scope_key
        BEGIN
            SELECT RAISE(ABORT, 'memory_evidence scope mismatch');
        END
    """,
    "trg_memory_evidence_scope_update": """
        CREATE TRIGGER IF NOT EXISTS trg_memory_evidence_scope_update
        BEFORE UPDATE ON memory_evidence
        WHEN (SELECT scope_key FROM memories WHERE id = NEW.memory_id) IS NOT NEW.scope_key
          OR (SELECT scope_key FROM memories WHERE id = NEW.source_memory_id) IS NOT NEW.scope_key
          OR (SELECT scope_key FROM memory_events WHERE id = NEW.event_id) IS NOT NEW.scope_key
        BEGIN
            SELECT RAISE(ABORT, 'memory_evidence scope mismatch');
        END
    """,
    "trg_memory_relations_scope_insert": """
        CREATE TRIGGER IF NOT EXISTS trg_memory_relations_scope_insert
        BEFORE INSERT ON memory_relations
        WHEN (SELECT scope_key FROM memories WHERE id = NEW.source_id) IS NOT NEW.scope_key
          OR (SELECT scope_key FROM memories WHERE id = NEW.target_id) IS NOT NEW.scope_key
          OR (SELECT scope_key FROM memory_events WHERE id = NEW.event_id) IS NOT NEW.scope_key
        BEGIN
            SELECT RAISE(ABORT, 'memory_relations scope mismatch');
        END
    """,
    "trg_memory_relations_scope_update": """
        CREATE TRIGGER IF NOT EXISTS trg_memory_relations_scope_update
        BEFORE UPDATE ON memory_relations
        WHEN (SELECT scope_key FROM memories WHERE id = NEW.source_id) IS NOT NEW.scope_key
          OR (SELECT scope_key FROM memories WHERE id = NEW.target_id) IS NOT NEW.scope_key
          OR (SELECT scope_key FROM memory_events WHERE id = NEW.event_id) IS NOT NEW.scope_key
        BEGIN
            SELECT RAISE(ABORT, 'memory_relations scope mismatch');
        END
    """,
}

_IMMUTABLE_TRIGGER_SQL = {
    "trg_memory_events_immutable_update": """
        CREATE TRIGGER IF NOT EXISTS trg_memory_events_immutable_update
        BEFORE UPDATE ON memory_events
        BEGIN
            SELECT RAISE(ABORT, 'memory_events is immutable');
        END
    """,
    "trg_memory_events_immutable_delete": """
        CREATE TRIGGER IF NOT EXISTS trg_memory_events_immutable_delete
        BEFORE DELETE ON memory_events
        BEGIN
            SELECT RAISE(ABORT, 'memory_events is immutable');
        END
    """,
}

_AUXILIARY_SCHEMA_V2 = _AUXILIARY_SCHEMA_V1 + "\n" + "\n".join(
    [
        """
        CREATE INDEX IF NOT EXISTS ix_memory_lifecycle_scope_status_known
            ON memory_lifecycle(scope_key, status, known_from, known_to);
        CREATE INDEX IF NOT EXISTS ix_memory_lifecycle_memory_known
            ON memory_lifecycle(memory_id, known_from, known_to);
        """,
        *[definition + ";" for definition in _GUARD_TRIGGER_SQL.values()],
    ]
)

_COMMON_TABLE_INFO = {
    "memory_events": (
        ("id", "TEXT", 0, None, 1),
        ("memory_id", "TEXT", 0, None, 0),
        ("scope_key", "TEXT", 1, None, 0),
        ("tenant_id", "TEXT", 1, None, 0),
        ("user_id", "TEXT", 1, None, 0),
        ("agent_id", "TEXT", 1, None, 0),
        ("project_id", "TEXT", 0, None, 0),
        ("session_id", "TEXT", 0, None, 0),
        ("event_type", "TEXT", 1, None, 0),
        ("payload_json", "TEXT", 1, None, 0),
        ("idempotency_key", "TEXT", 0, None, 0),
        ("occurred_at", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memories": (
        ("id", "TEXT", 0, None, 1),
        ("scope_key", "TEXT", 1, None, 0),
        ("tenant_id", "TEXT", 1, None, 0),
        ("user_id", "TEXT", 1, None, 0),
        ("agent_id", "TEXT", 1, None, 0),
        ("project_id", "TEXT", 0, None, 0),
        ("session_id", "TEXT", 0, None, 0),
        ("kind", "TEXT", 1, None, 0),
        ("content", "TEXT", 1, None, 0),
        ("metadata_json", "TEXT", 1, None, 0),
        ("status", "TEXT", 1, None, 0),
        ("confidence", "REAL", 1, None, 0),
        ("observed_at", "TEXT", 1, None, 0),
        ("valid_from", "TEXT", 0, None, 0),
        ("valid_to", "TEXT", 0, None, 0),
        ("created_at", "TEXT", 1, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
    ),
    "memory_relations": (
        ("id", "TEXT", 0, None, 1),
        ("scope_key", "TEXT", 1, None, 0),
        ("source_id", "TEXT", 1, None, 0),
        ("target_id", "TEXT", 1, None, 0),
        ("relation_type", "TEXT", 1, None, 0),
        ("valid_from", "TEXT", 0, None, 0),
        ("valid_to", "TEXT", 0, None, 0),
        ("confidence", "REAL", 1, None, 0),
        ("event_id", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_history": (
        ("id", "TEXT", 0, None, 1),
        ("memory_id", "TEXT", 1, None, 0),
        ("event_id", "TEXT", 1, None, 0),
        ("action", "TEXT", 1, None, 0),
        ("status", "TEXT", 1, None, 0),
        ("details_json", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
}
_TABLE_INFO_V1 = {
    **_COMMON_TABLE_INFO,
    "memory_evidence": (
        ("memory_id", "TEXT", 1, None, 0),
        ("source_memory_id", "TEXT", 1, None, 0),
        ("event_id", "TEXT", 1, None, 0),
        ("relation", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
}
_TABLE_INFO_V2 = {
    **_COMMON_TABLE_INFO,
    "memory_evidence": (
        ("memory_id", "TEXT", 1, None, 0),
        ("source_memory_id", "TEXT", 1, None, 0),
        ("event_id", "TEXT", 1, None, 0),
        ("scope_key", "TEXT", 1, None, 0),
        ("relation", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
    "memory_lifecycle": (
        ("memory_id", "TEXT", 1, None, 1),
        ("scope_key", "TEXT", 1, None, 0),
        ("status", "TEXT", 1, None, 3),
        ("known_from", "TEXT", 1, None, 2),
        ("known_to", "TEXT", 0, None, 0),
        ("valid_from", "TEXT", 0, None, 0),
        ("valid_to", "TEXT", 0, None, 0),
        ("event_id", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ),
}

_COMMON_INDEXES = {
    "ux_memory_events_scope_idempotency": (
        "memory_events", ("scope_key", "idempotency_key"), True, True
    ),
    "ix_memories_scope_status_kind_created": (
        "memories", ("scope_key", "status", "kind", "created_at"), False, False
    ),
    "ix_memories_scope_validity": (
        "memories", ("scope_key", "valid_from", "valid_to"), False, False
    ),
    "ix_memory_relations_scope_validity": (
        "memory_relations", ("scope_key", "valid_from", "valid_to"), False, False
    ),
}
_INDEXES_V2 = {
    **_COMMON_INDEXES,
    "ix_memory_lifecycle_scope_status_known": (
        "memory_lifecycle",
        ("scope_key", "status", "known_from", "known_to"),
        False,
        False,
    ),
    "ix_memory_lifecycle_memory_known": (
        "memory_lifecycle", ("memory_id", "known_from", "known_to"), False, False
    ),
}
_FOREIGN_KEYS_V1 = {
    "memory_evidence": {
        ("memory_id", "memories", "id", "CASCADE"),
        ("source_memory_id", "memories", "id", "CASCADE"),
        ("event_id", "memory_events", "id", "NO ACTION"),
    },
    "memory_relations": {
        ("source_id", "memories", "id", "CASCADE"),
        ("target_id", "memories", "id", "CASCADE"),
        ("event_id", "memory_events", "id", "NO ACTION"),
    },
    "memory_history": {
        ("memory_id", "memories", "id", "CASCADE"),
        ("event_id", "memory_events", "id", "NO ACTION"),
    },
}
_FOREIGN_KEYS_V2 = {
    **_FOREIGN_KEYS_V1,
    "memory_lifecycle": {
        ("memory_id", "memories", "id", "NO ACTION"),
        ("event_id", "memory_events", "id", "NO ACTION"),
    },
}

_FTS5_PATTERN = re.compile(r"\busing\s+fts5\s*\(", re.IGNORECASE)
_UNICODE61_PATTERN = re.compile(r"\btokenize\s*=\s*(['\"])unicode61\1", re.IGNORECASE)
_MEMORY_ID_UNINDEXED_PATTERN = re.compile(r"\bmemory_id\s+unindexed\s*,", re.IGNORECASE)


def _execute_statements(connection: sqlite3.Connection, script: str) -> None:
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            connection.execute(pending)
            pending = ""
    if pending.strip():
        raise sqlite3.DatabaseError("incomplete schema statement")


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.casefold().split()).rstrip(";")


def _validate_tables(connection: sqlite3.Connection, expected: dict[str, tuple]) -> None:
    for table, fingerprint in expected.items():
        actual = tuple(row[1:6] for row in connection.execute(f'PRAGMA table_info("{table}")'))
        if actual != fingerprint:
            raise sqlite3.DatabaseError(
                f"{table} columns do not match schema; {table} table_info fingerprint differs"
            )


def _validate_indexes(connection: sqlite3.Connection, expected: dict[str, tuple]) -> None:
    for name, (table, columns, unique, partial) in expected.items():
        indexes = {row[1]: row for row in connection.execute(f'PRAGMA index_list("{table}")')}
        index = indexes.get(name)
        actual_columns = tuple(row[2] for row in connection.execute(f'PRAGMA index_info("{name}")'))
        if (
            index is None
            or actual_columns != columns
            or bool(index[2]) is not unique
            or bool(index[4]) is not partial
        ):
            raise sqlite3.DatabaseError(f"index {name} does not match schema")
        if partial:
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
            ).fetchone()
            sql = "" if row is None or row[0] is None else _normalize_sql(row[0])
            if not re.search(r"\bwhere idempotency_key is not null$", sql):
                raise sqlite3.DatabaseError(f"index {name} does not match schema")


def _validate_foreign_keys(connection: sqlite3.Connection, expected: dict[str, set]) -> None:
    for table, fingerprint in expected.items():
        actual = {
            (row[3], row[2], row[4], row[6])
            for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
        }
        if actual != fingerprint:
            raise sqlite3.DatabaseError(f"{table} foreign keys do not match schema")


def _validate_triggers(connection: sqlite3.Connection, expected: dict[str, str]) -> None:
    actual = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    for name, definition in expected.items():
        normalized = _normalize_sql(definition)
        accepted = {normalized, normalized.replace(" if not exists", "", 1)}
        if actual.get(name) is None or _normalize_sql(actual[name]) not in accepted:
            if name.startswith("trg_memory_events"):
                raise sqlite3.DatabaseError(
                    f"trigger {name} does not enforce immutable memory_events"
                )
            raise sqlite3.DatabaseError(f"trigger {name} does not enforce scope ownership")


def _validate_fts(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
    ).fetchone()
    sql = "" if row is None or row[0] is None else row[0]
    if not _FTS5_PATTERN.search(sql):
        raise sqlite3.DatabaseError("memory_fts must be an FTS5 virtual table")
    columns = tuple(row[1] for row in connection.execute('PRAGMA table_info("memory_fts")'))
    if columns != ("memory_id", "content"):
        raise sqlite3.DatabaseError("memory_fts columns do not match schema")
    if not _UNICODE61_PATTERN.search(sql):
        raise sqlite3.DatabaseError("memory_fts must use unicode61")
    if not _MEMORY_ID_UNINDEXED_PATTERN.search(sql):
        raise sqlite3.DatabaseError("memory_fts memory_id must be UNINDEXED")


def _validate_schema(connection: sqlite3.Connection, version: int) -> None:
    if version == 1:
        _validate_tables(connection, _TABLE_INFO_V1)
        _validate_indexes(connection, _COMMON_INDEXES)
        _validate_foreign_keys(connection, _FOREIGN_KEYS_V1)
        _validate_triggers(connection, _IMMUTABLE_TRIGGER_SQL)
    elif version == 2:
        _validate_tables(connection, _TABLE_INFO_V2)
        _validate_indexes(connection, _INDEXES_V2)
        _validate_foreign_keys(connection, _FOREIGN_KEYS_V2)
        _validate_triggers(connection, {**_IMMUTABLE_TRIGGER_SQL, **_GUARD_TRIGGER_SQL})
    else:
        raise sqlite3.DatabaseError(f"unsupported schema version {version}")
    _validate_fts(connection)


def _scope_key(row: sqlite3.Row) -> str:
    return json.dumps(
        [row["tenant_id"], row["user_id"], row["agent_id"], row["project_id"], row["session_id"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_timestamp(value: Any, field: str, event_id: str) -> datetime:
    if not isinstance(value, str):
        raise sqlite3.DatabaseError(f"event {event_id!r} has incomplete {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("naive timestamp")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise sqlite3.DatabaseError(
            f"event {event_id!r} has invalid {field}"
        ) from error


def _timestamp_to_storage(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _optional_event_timestamp(value: Any, field: str, event_id: str) -> str | None:
    if value is None:
        return None
    return _timestamp_to_storage(_parse_timestamp(value, field, event_id))


def _migration_plan(connection: sqlite3.Connection) -> tuple[list[tuple], list[tuple]]:
    connection.row_factory = sqlite3.Row
    memory_rows = {
        row["id"]: row for row in connection.execute("SELECT * FROM memories")
    }
    events = connection.execute(
        "SELECT * FROM memory_events ORDER BY scope_key, occurred_at, created_at, id"
    ).fetchall()
    owners: dict[str, str] = {}
    active: dict[str, tuple] = {}
    completed: set[str] = set()
    lifecycle: list[list[Any]] = []
    active_positions: dict[str, int] = {}
    for event in events:
        event_id = event["id"]
        if event["scope_key"] != _scope_key(event):
            raise sqlite3.DatabaseError(f"event {event_id!r} has cross-scope fields")
        memory_id = event["memory_id"]
        if not isinstance(memory_id, str) or not memory_id:
            raise sqlite3.DatabaseError(f"event {event_id!r} has incomplete memory_id")
        prior_owner = owners.setdefault(memory_id, event["scope_key"])
        if prior_owner != event["scope_key"]:
            raise sqlite3.DatabaseError(f"memory {memory_id!r} has cross-scope event ownership")
        projection = memory_rows.get(memory_id)
        if projection is None:
            raise sqlite3.DatabaseError(f"memory {memory_id!r} has no projection owner")
        if projection["scope_key"] != event["scope_key"] or projection["scope_key"] != _scope_key(projection):
            raise sqlite3.DatabaseError(f"memory {memory_id!r} is foreign-owned")
        try:
            payload = json.loads(event["payload_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise sqlite3.DatabaseError(f"event {event_id!r} has invalid payload") from error
        if not isinstance(payload, dict) or payload.get("memory_id") != memory_id:
            raise sqlite3.DatabaseError(f"event {event_id!r} has incomplete payload memory_id")
        occurred = _parse_timestamp(event["occurred_at"], "occurred_at", event_id)
        occurred_at = _timestamp_to_storage(occurred)
        created_at = _timestamp_to_storage(
            _parse_timestamp(event["created_at"], "created_at", event_id)
        )
        if event["event_type"] == "remembered":
            if memory_id in active or memory_id in completed:
                raise sqlite3.DatabaseError(f"duplicate remembered stream for memory {memory_id!r}")
            observation = payload.get("observation")
            required = {
                "kind", "content", "metadata", "status", "confidence", "observed_at",
                "valid_from", "valid_to", "created_at", "updated_at",
            }
            if not isinstance(observation, dict) or not required <= observation.keys():
                raise sqlite3.DatabaseError(f"remembered event {event_id!r} is incomplete")
            if observation["status"] != "active" or not isinstance(observation["metadata"], dict):
                raise sqlite3.DatabaseError(f"remembered event {event_id!r} is invalid")
            _parse_timestamp(observation["observed_at"], "observed_at", event_id)
            _parse_timestamp(observation["created_at"], "observation.created_at", event_id)
            _parse_timestamp(observation["updated_at"], "observation.updated_at", event_id)
            valid_from = _optional_event_timestamp(
                observation["valid_from"], "valid_from", event_id
            )
            valid_to = _optional_event_timestamp(
                observation["valid_to"], "valid_to", event_id
            )
            position = len(lifecycle)
            lifecycle.append(
                [
                    memory_id,
                    event["scope_key"],
                    "active",
                    occurred_at,
                    None,
                    valid_from,
                    valid_to,
                    event_id,
                    created_at,
                ]
            )
            active[memory_id] = (occurred, valid_from, valid_to)
            active_positions[memory_id] = position
        elif event["event_type"] == "forgotten":
            current = active.get(memory_id)
            if current is None or memory_id in completed or occurred <= current[0]:
                raise sqlite3.DatabaseError(f"forgotten event {event_id!r} is out-of-order")
            lifecycle[active_positions[memory_id]][4] = occurred_at
            lifecycle.append(
                [
                    memory_id,
                    event["scope_key"],
                    "forgotten",
                    occurred_at,
                    None,
                    current[1],
                    current[2],
                    event_id,
                    created_at,
                ]
            )
            del active[memory_id]
            completed.add(memory_id)
        else:
            raise sqlite3.DatabaseError(
                f"unsupported v1 event type {event['event_type']!r} in event {event_id!r}"
            )

    projection_without_stream = next(
        (memory_id for memory_id in memory_rows if memory_id not in owners),
        None,
    )
    if projection_without_stream is not None:
        raise sqlite3.DatabaseError(
            f"memory {projection_without_stream!r} has no immutable event stream"
        )

    evidence: list[tuple] = []
    for row in connection.execute(
        """
        SELECT evidence.*, target.scope_key AS target_scope,
               source.scope_key AS source_scope, event.scope_key AS event_scope
        FROM memory_evidence AS evidence
        LEFT JOIN memories AS target ON target.id = evidence.memory_id
        LEFT JOIN memories AS source ON source.id = evidence.source_memory_id
        LEFT JOIN memory_events AS event ON event.id = evidence.event_id
        """
    ):
        scope = row["target_scope"]
        if scope is None or row["source_scope"] != scope or row["event_scope"] != scope:
            raise sqlite3.DatabaseError("memory_evidence contains a cross-scope endpoint")
        evidence.append(
            (
                row["memory_id"], row["source_memory_id"], row["event_id"], scope,
                row["relation"], row["created_at"],
            )
        )
    for row in connection.execute(
        """
        SELECT relation.id FROM memory_relations AS relation
        LEFT JOIN memories AS source ON source.id = relation.source_id
        LEFT JOIN memories AS target ON target.id = relation.target_id
        LEFT JOIN memory_events AS event ON event.id = relation.event_id
        WHERE source.scope_key IS NOT relation.scope_key
           OR target.scope_key IS NOT relation.scope_key
           OR event.scope_key IS NOT relation.scope_key
        LIMIT 1
        """
    ):
        raise sqlite3.DatabaseError(
            f"memory_relations row {row['id']!r} contains a cross-scope endpoint"
        )
    return [tuple(row) for row in lifecycle], evidence


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    lifecycle, evidence = _migration_plan(connection)
    connection.execute("ALTER TABLE memory_evidence RENAME TO memory_evidence_v1")
    connection.execute(
        """
        CREATE TABLE memory_evidence (
            memory_id TEXT NOT NULL,
            source_memory_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            relation TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (event_id) REFERENCES memory_events(id)
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO memory_evidence (
            memory_id, source_memory_id, event_id, scope_key, relation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        evidence,
    )
    connection.execute("DROP TABLE memory_evidence_v1")
    connection.execute(
        """
        CREATE TABLE memory_lifecycle (
            memory_id TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            status TEXT NOT NULL,
            known_from TEXT NOT NULL,
            known_to TEXT,
            valid_from TEXT,
            valid_to TEXT,
            event_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(memory_id, known_from, status),
            FOREIGN KEY(memory_id) REFERENCES memories(id),
            FOREIGN KEY(event_id) REFERENCES memory_events(id)
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO memory_lifecycle (
            memory_id, scope_key, status, known_from, known_to, valid_from,
            valid_to, event_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        lifecycle,
    )
    _execute_statements(connection, _AUXILIARY_SCHEMA_V2)


def initialize_or_migrate(connection: sqlite3.Connection, path: Path) -> None:
    """Initialize or migrate *connection* atomically, validating exact fingerprints."""
    try:
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            _execute_statements(connection, _TABLE_SCHEMA_V2)
            _validate_tables(connection, _TABLE_INFO_V2)
            _execute_statements(connection, _AUXILIARY_SCHEMA_V2)
        elif version == 1:
            _validate_schema(connection, 1)
            _migrate_v1_to_v2(connection)
        elif version != SCHEMA_VERSION:
            raise sqlite3.DatabaseError(
                f"unsupported schema version {version}; expected {SCHEMA_VERSION}"
            )
        _validate_schema(connection, SCHEMA_VERSION)
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise sqlite3.DatabaseError(f"schema operation failed for {path}: {error}") from error
