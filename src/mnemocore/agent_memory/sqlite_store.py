"""Versioned SQLite storage foundation for agent memory."""

from __future__ import annotations

import asyncio
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from .errors import StorageError

SCHEMA_VERSION = 1

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

_REQUIRED_COLUMNS = {
    "memory_events": (
        "id",
        "memory_id",
        "scope_key",
        "tenant_id",
        "user_id",
        "agent_id",
        "project_id",
        "session_id",
        "event_type",
        "payload_json",
        "idempotency_key",
        "occurred_at",
        "created_at",
    ),
    "memories": (
        "id",
        "scope_key",
        "tenant_id",
        "user_id",
        "agent_id",
        "project_id",
        "session_id",
        "kind",
        "content",
        "metadata_json",
        "status",
        "confidence",
        "observed_at",
        "valid_from",
        "valid_to",
        "created_at",
        "updated_at",
    ),
    "memory_evidence": (
        "memory_id",
        "source_memory_id",
        "event_id",
        "relation",
        "created_at",
    ),
    "memory_relations": (
        "id",
        "scope_key",
        "source_id",
        "target_id",
        "relation_type",
        "valid_from",
        "valid_to",
        "confidence",
        "event_id",
        "created_at",
    ),
    "memory_history": (
        "id",
        "memory_id",
        "event_id",
        "action",
        "status",
        "details_json",
        "created_at",
    ),
    "memory_fts": ("memory_id", "content"),
}

_REQUIRED_INDEXES = {
    "ux_memory_events_scope_idempotency": (
        "memory_events",
        ("scope_key", "idempotency_key"),
        True,
        True,
    ),
    "ix_memories_scope_status_kind_created": (
        "memories",
        ("scope_key", "status", "kind", "created_at"),
        False,
        False,
    ),
    "ix_memories_scope_validity": (
        "memories",
        ("scope_key", "valid_from", "valid_to"),
        False,
        False,
    ),
    "ix_memory_relations_scope_validity": (
        "memory_relations",
        ("scope_key", "valid_from", "valid_to"),
        False,
        False,
    ),
}

_REQUIRED_FOREIGN_KEYS = {
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

_FTS5_PATTERN = re.compile(r"\busing\s+fts5\s*\(", re.IGNORECASE)
_UNICODE61_PATTERN = re.compile(
    r"\btokenize\s*=\s*(['\"])unicode61\1",
    re.IGNORECASE,
)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def _execute_statements(connection: sqlite3.Connection, script: str) -> None:
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            connection.execute(pending)
            pending = ""
    if pending.strip():
        raise sqlite3.DatabaseError("incomplete schema statement")


def _validate_table_columns(connection: sqlite3.Connection) -> None:
    for table, required_columns in _REQUIRED_COLUMNS.items():
        actual_columns = tuple(
            row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
        )
        if actual_columns != required_columns:
            raise sqlite3.DatabaseError(
                f"{table} columns do not match schema version {SCHEMA_VERSION}"
            )


def _validate_indexes(connection: sqlite3.Connection) -> None:
    for name, (table, required_columns, unique, partial) in _REQUIRED_INDEXES.items():
        indexes = {
            row[1]: row
            for row in connection.execute(f'PRAGMA index_list("{table}")')
        }
        index = indexes.get(name)
        actual_columns = tuple(
            row[2] for row in connection.execute(f'PRAGMA index_info("{name}")')
        )
        if (
            index is None
            or actual_columns != required_columns
            or bool(index[2]) is not unique
            or bool(index[4]) is not partial
        ):
            raise sqlite3.DatabaseError(
                f"index {name} does not match schema version {SCHEMA_VERSION}"
            )

        if partial:
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                (name,),
            ).fetchone()
            sql = "" if row is None or row[0] is None else _normalize_sql(row[0])
            if not re.search(r"\bwhere idempotency_key is not null$", sql):
                raise sqlite3.DatabaseError(
                    f"index {name} does not match schema version {SCHEMA_VERSION}"
                )


def _validate_foreign_keys(connection: sqlite3.Connection) -> None:
    for table, required_keys in _REQUIRED_FOREIGN_KEYS.items():
        actual_keys = {
            (row[3], row[2], row[4], row[6])
            for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
        }
        if actual_keys != required_keys:
            raise sqlite3.DatabaseError(
                f"{table} foreign keys do not match schema version {SCHEMA_VERSION}"
            )


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.casefold().split()).rstrip(";")


def _validate_immutable_triggers(connection: sqlite3.Connection) -> None:
    actual_triggers = {
        row[0]: row[1]
        for row in connection.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'trigger' AND tbl_name = 'memory_events'
            """
        )
    }
    for name, definition in _IMMUTABLE_TRIGGER_SQL.items():
        actual = actual_triggers.get(name)
        expected = _normalize_sql(definition)
        expected_without_guard = expected.replace(" if not exists", "", 1)
        if actual is None or _normalize_sql(actual) not in {
            expected,
            expected_without_guard,
        }:
            raise sqlite3.DatabaseError(
                f"trigger {name} does not enforce immutable memory_events"
            )


def _validate_fts(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
    ).fetchone()
    sql = "" if row is None or row[0] is None else row[0]
    if not _FTS5_PATTERN.search(sql):
        raise sqlite3.DatabaseError("memory_fts must be an FTS5 virtual table")
    if not _UNICODE61_PATTERN.search(sql):
        raise sqlite3.DatabaseError("memory_fts must use unicode61")


def _validate_schema(connection: sqlite3.Connection) -> None:
    _validate_table_columns(connection)
    _validate_indexes(connection)
    _validate_foreign_keys(connection)
    _validate_immutable_triggers(connection)
    _validate_fts(connection)


def _initialize_schema(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    _execute_statements(connection, _TABLE_SCHEMA_V1)
                    _validate_table_columns(connection)
                    _execute_statements(connection, _AUXILIARY_SCHEMA_V1)
                    _validate_schema(connection)
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif version == SCHEMA_VERSION:
                    _validate_schema(connection)
                else:
                    raise sqlite3.DatabaseError(
                        f"unsupported schema version {version}; expected {SCHEMA_VERSION}"
                    )
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise
    except (OSError, sqlite3.Error) as error:
        raise StorageError(
            f"Failed to initialize SQLite database at {path}: {error}"
        ) from error


class SQLiteMemoryStore:
    """SQLite store lifecycle and version-1 schema initialization."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._closed = False
        self._lifecycle_lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: str | Path) -> SQLiteMemoryStore:
        resolved_path = Path(path).expanduser().resolve()
        await asyncio.to_thread(_initialize_schema, resolved_path)
        return cls(resolved_path)

    async def close(self) -> None:
        async with self._lifecycle_lock:
            self._closed = True
