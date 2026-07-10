"""Versioned SQLite storage foundation for agent memory."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from .errors import StorageError

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
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

CREATE UNIQUE INDEX IF NOT EXISTS ux_memory_events_scope_idempotency
    ON memory_events(scope_key, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_memories_scope_status_kind_created
    ON memories(scope_key, status, kind, created_at);

CREATE INDEX IF NOT EXISTS ix_memories_scope_validity
    ON memories(scope_key, valid_from, valid_to);

CREATE INDEX IF NOT EXISTS ix_memory_relations_scope_validity
    ON memory_relations(scope_key, valid_from, valid_to);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize='unicode61'
);
"""


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=10000")
    return connection


def _initialize_schema(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise sqlite3.DatabaseError(
                    f"unsupported schema version {version}; expected {SCHEMA_VERSION}"
                )
            connection.executescript(
                "BEGIN IMMEDIATE;\n"
                + _SCHEMA_V1
                + f"\nPRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
            )
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
