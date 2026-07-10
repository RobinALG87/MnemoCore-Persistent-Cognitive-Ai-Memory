"""Versioned SQLite storage foundation for agent memory."""

from __future__ import annotations

import asyncio
import builtins
import json
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, TypeVar
from uuid import uuid4

from .errors import ClosedStoreError, MemoryNotFoundError, StorageError, ValidationError
from .models import (
    MemoryEvent,
    MemoryEventType,
    MemoryHistoryEntry,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    RecallResult,
    utc_now,
)

SCHEMA_VERSION = 1
_T = TypeVar("_T")

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

_REQUIRED_TABLE_INFO = {
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
    "memory_evidence": (
        ("memory_id", "TEXT", 1, None, 0),
        ("source_memory_id", "TEXT", 1, None, 0),
        ("event_id", "TEXT", 1, None, 0),
        ("relation", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
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

_REQUIRED_FTS_COLUMNS = ("memory_id", "content")

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
_MEMORY_ID_UNINDEXED_PATTERN = re.compile(
    r"\bmemory_id\s+unindexed\s*,",
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


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        thawed: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError("JSON mapping keys must be strings")
            thawed[key] = _thaw_json(item)
        return thawed
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _thaw_json(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError("value must contain only JSON-compatible values") from error


def _timestamp_to_storage(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _timestamp_from_storage(value: str, name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"stored {name} must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"stored {name} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _timestamp_from_input(value: Optional[str], name: str) -> Optional[datetime]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp is not timezone-aware")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be a valid timezone-aware ISO-8601 string") from error


def _record_observation_payload(record: MemoryRecord) -> dict[str, Any]:
    """Return the normalized event payload needed to rebuild a projection."""
    return {
        "memory_id": record.id,
        "observation": {
            "kind": record.kind.value,
            "content": record.content,
            "metadata": record.metadata,
            "status": record.status.value,
            "confidence": record.confidence,
            "observed_at": _timestamp_to_storage(record.observed_at),
            "valid_from": (
                _timestamp_to_storage(record.valid_from)
                if record.valid_from is not None
                else None
            ),
            "valid_to": (
                _timestamp_to_storage(record.valid_to)
                if record.valid_to is not None
                else None
            ),
            "created_at": _timestamp_to_storage(record.created_at),
            "updated_at": _timestamp_to_storage(record.updated_at),
        },
    }


def _scope_from_row(row: sqlite3.Row) -> MemoryScope:
    return MemoryScope(
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        project_id=row["project_id"],
        session_id=row["session_id"],
    )


def _row_to_record(path: Path, row: sqlite3.Row) -> MemoryRecord:
    try:
        metadata = json.loads(row["metadata_json"])
        if not isinstance(metadata, Mapping):
            raise ValidationError("stored memory metadata must be a mapping")
        return MemoryRecord(
            id=row["id"],
            scope=_scope_from_row(row),
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            metadata=dict(metadata),
            status=MemoryStatus(row["status"]),
            confidence=row["confidence"],
            observed_at=_timestamp_from_storage(row["observed_at"], "observed_at"),
            valid_from=(
                _timestamp_from_storage(row["valid_from"], "valid_from")
                if row["valid_from"] is not None
                else None
            ),
            valid_to=(
                _timestamp_from_storage(row["valid_to"], "valid_to")
                if row["valid_to"] is not None
                else None
            ),
            created_at=_timestamp_from_storage(row["created_at"], "created_at"),
            updated_at=_timestamp_from_storage(row["updated_at"], "updated_at"),
        )
    except Exception as error:
        raise StorageError(f"Failed to hydrate memory row from {path}: {error}") from error


def _row_to_event(path: Path, row: sqlite3.Row) -> MemoryEvent:
    try:
        payload = json.loads(row["payload_json"])
        return MemoryEvent(
            id=row["id"],
            scope=_scope_from_row(row),
            event_type=MemoryEventType(row["event_type"]),
            payload=payload,
            occurred_at=_timestamp_from_storage(row["occurred_at"], "occurred_at"),
            created_at=_timestamp_from_storage(row["created_at"], "created_at"),
            memory_id=row["memory_id"],
            idempotency_key=row["idempotency_key"],
        )
    except Exception as error:
        raise StorageError(f"Failed to hydrate memory event from {path}: {error}") from error


def _row_to_history(path: Path, row: sqlite3.Row) -> MemoryHistoryEntry:
    try:
        details = json.loads(row["details_json"])
        if not isinstance(details, Mapping):
            raise ValidationError("stored memory history details must be a mapping")
        return MemoryHistoryEntry(
            id=row["id"],
            memory_id=row["memory_id"],
            event_id=row["event_id"],
            action=MemoryEventType(row["action"]),
            status=MemoryStatus(row["status"]),
            details=dict(details),
            created_at=_timestamp_from_storage(row["created_at"], "created_at"),
        )
    except Exception as error:
        raise StorageError(f"Failed to hydrate memory history from {path}: {error}") from error


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
    for table, required_info in _REQUIRED_TABLE_INFO.items():
        actual_info = tuple(
            row[1:6] for row in connection.execute(f'PRAGMA table_info("{table}")')
        )
        if actual_info != required_info:
            raise sqlite3.DatabaseError(
                f"{table} columns do not match schema version {SCHEMA_VERSION}; "
                f"{table} table_info fingerprint differs"
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
    actual_columns = tuple(
        item[1] for item in connection.execute('PRAGMA table_info("memory_fts")')
    )
    if actual_columns != _REQUIRED_FTS_COLUMNS:
        raise sqlite3.DatabaseError(
            f"memory_fts columns do not match schema version {SCHEMA_VERSION}"
        )
    if not _UNICODE61_PATTERN.search(sql):
        raise sqlite3.DatabaseError("memory_fts must use unicode61")
    if not _MEMORY_ID_UNINDEXED_PATTERN.search(sql):
        raise sqlite3.DatabaseError("memory_fts memory_id must be UNINDEXED")


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


def _remember(
    path: Path,
    scope: MemoryScope,
    content: str,
    *,
    kind: MemoryKind,
    metadata: Optional[Mapping[str, Any]],
    idempotency_key: Optional[str],
    confidence: float,
    observed_at: Optional[str],
    valid_from: Optional[str],
    valid_to: Optional[str],
) -> MemoryRecord:
    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("BEGIN IMMEDIATE")
                if idempotency_key is not None:
                    existing = connection.execute(
                        """
                        SELECT memory_id
                        FROM memory_events
                        WHERE scope_key = ? AND idempotency_key = ?
                        """,
                        (scope.scope_key, idempotency_key),
                    ).fetchone()
                    if existing is not None:
                        existing_record = connection.execute(
                            "SELECT * FROM memories WHERE scope_key = ? AND id = ?",
                            (scope.scope_key, existing["memory_id"]),
                        ).fetchone()
                        if existing_record is None:
                            raise StorageError(
                                "Idempotency event refers to a missing memory projection"
                            )
                        record = _row_to_record(path, existing_record)
                        connection.commit()
                        return record

                try:
                    normalized_kind = MemoryKind(kind)
                except (TypeError, ValueError) as error:
                    raise ValidationError("kind must be a valid MemoryKind") from error

                now = utc_now()
                memory_id = uuid4().hex
                metadata_json = _canonical_json({} if metadata is None else metadata)
                record = MemoryRecord(
                    id=memory_id,
                    scope=scope,
                    kind=normalized_kind,
                    content=content,
                    metadata=json.loads(metadata_json),
                    confidence=confidence,
                    observed_at=_timestamp_from_input(observed_at, "observed_at")
                    or now,
                    valid_from=_timestamp_from_input(valid_from, "valid_from"),
                    valid_to=_timestamp_from_input(valid_to, "valid_to"),
                    created_at=now,
                    updated_at=now,
                )
                event = MemoryEvent(
                    id=uuid4().hex,
                    memory_id=memory_id,
                    scope=scope,
                    event_type=MemoryEventType.REMEMBERED,
                    payload=_record_observation_payload(record),
                    idempotency_key=idempotency_key,
                    occurred_at=now,
                    created_at=now,
                )
                history_id = f"{event.id}:history"
                payload_json = _canonical_json(event.payload)
                details_json = _canonical_json({})
                timestamp = _timestamp_to_storage(now)
                observed_timestamp = _timestamp_to_storage(record.observed_at)
                valid_from_timestamp = (
                    _timestamp_to_storage(record.valid_from)
                    if record.valid_from is not None
                    else None
                )
                valid_to_timestamp = (
                    _timestamp_to_storage(record.valid_to)
                    if record.valid_to is not None
                    else None
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
                        event.id,
                        memory_id,
                        scope.scope_key,
                        scope.tenant_id,
                        scope.user_id,
                        scope.agent_id,
                        scope.project_id,
                        scope.session_id,
                        event.event_type.value,
                        payload_json,
                        idempotency_key,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO memories (
                        id, scope_key, tenant_id, user_id, agent_id, project_id,
                        session_id, kind, content, metadata_json, status,
                        confidence, observed_at, valid_from, valid_to, created_at,
                        updated_at
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
                        record.kind.value,
                        record.content,
                        metadata_json,
                        record.status.value,
                        record.confidence,
                        observed_timestamp,
                        valid_from_timestamp,
                        valid_to_timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO memory_history (
                        id, memory_id, event_id, action, status, details_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        history_id,
                        memory_id,
                        event.id,
                        event.event_type.value,
                        record.status.value,
                        details_json,
                        timestamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)",
                    (memory_id, record.content),
                )
                connection.commit()
                return record
            except Exception:
                connection.rollback()
                raise
    except sqlite3.Error as error:
        raise StorageError(f"Failed to remember memory in {path}: {error}") from error


def _get(
    path: Path,
    scope: MemoryScope,
    memory_id: str,
    *,
    include_forgotten: bool,
) -> MemoryRecord:
    sql = "SELECT * FROM memories WHERE scope_key = ? AND id = ?"
    parameters: list[Any] = [scope.scope_key, memory_id]
    if not include_forgotten:
        sql += " AND status != ?"
        parameters.append(MemoryStatus.FORGOTTEN.value)
    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(sql, parameters).fetchone()
    except sqlite3.Error as error:
        raise StorageError(f"Failed to get memory from {path}: {error}") from error
    if row is None:
        raise MemoryNotFoundError(f"Memory {memory_id!r} was not found in this scope")
    return _row_to_record(path, row)


def _forget(
    path: Path,
    scope: MemoryScope,
    memory_id: str,
    *,
    reason: Optional[str],
) -> MemoryRecord:
    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM memories WHERE scope_key = ? AND id = ?",
                    (scope.scope_key, memory_id),
                ).fetchone()
                if existing is None:
                    raise MemoryNotFoundError(
                        f"Memory {memory_id!r} was not found in this scope"
                    )

                record = _row_to_record(path, existing)
                if record.status is MemoryStatus.FORGOTTEN:
                    connection.commit()
                    return record
                if record.status is not MemoryStatus.ACTIVE:
                    raise MemoryNotFoundError(
                        f"Active memory {memory_id!r} was not found in this scope"
                    )

                now = utc_now()
                if now <= record.updated_at:
                    now = record.updated_at + timedelta(microseconds=1)
                timestamp = _timestamp_to_storage(now)
                event = MemoryEvent(
                    id=uuid4().hex,
                    memory_id=memory_id,
                    scope=scope,
                    event_type=MemoryEventType.FORGOTTEN,
                    payload={"memory_id": memory_id, "reason": reason},
                    occurred_at=now,
                    created_at=now,
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
                        event.id,
                        memory_id,
                        scope.scope_key,
                        scope.tenant_id,
                        scope.user_id,
                        scope.agent_id,
                        scope.project_id,
                        scope.session_id,
                        event.event_type.value,
                        _canonical_json(event.payload),
                        None,
                        timestamp,
                        timestamp,
                    ),
                )
                updated = connection.execute(
                    """
                    UPDATE memories
                    SET status = ?, updated_at = ?
                    WHERE scope_key = ? AND id = ? AND status = ?
                    """,
                    (
                        MemoryStatus.FORGOTTEN.value,
                        timestamp,
                        scope.scope_key,
                        memory_id,
                        MemoryStatus.ACTIVE.value,
                    ),
                )
                if updated.rowcount != 1:
                    raise sqlite3.DatabaseError(
                        "active memory projection changed during forget"
                    )
                connection.execute(
                    """
                    INSERT INTO memory_history (
                        id, memory_id, event_id, action, status, details_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{event.id}:history",
                        memory_id,
                        event.id,
                        event.event_type.value,
                        MemoryStatus.FORGOTTEN.value,
                        _canonical_json({"reason": reason}),
                        timestamp,
                    ),
                )
                connection.execute(
                    "DELETE FROM memory_fts WHERE memory_id = ?",
                    (memory_id,),
                )
                forgotten = connection.execute(
                    "SELECT * FROM memories WHERE scope_key = ? AND id = ?",
                    (scope.scope_key, memory_id),
                ).fetchone()
                if forgotten is None:
                    raise sqlite3.DatabaseError(
                        "forgotten memory projection disappeared during forget"
                    )
                result = _row_to_record(path, forgotten)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise
    except sqlite3.Error as error:
        raise StorageError(f"Failed to forget memory in {path}: {error}") from error


def _record_from_remembered_event(path: Path, event: MemoryEvent) -> MemoryRecord:
    observation = event.payload.get("observation")
    if event.memory_id is None or not isinstance(observation, Mapping):
        raise StorageError(
            f"Cannot rebuild memory from incomplete remembered event {event.id!r} in {path}"
        )
    try:
        if observation.get("status") != MemoryStatus.ACTIVE.value:
            raise ValidationError("remembered observation status must be active")
        metadata = observation.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValidationError("remembered observation metadata must be a mapping")
        return MemoryRecord(
            id=event.memory_id,
            scope=event.scope,
            kind=MemoryKind(observation["kind"]),
            content=observation["content"],
            metadata=metadata,
            status=MemoryStatus(observation["status"]),
            confidence=observation["confidence"],
            observed_at=_timestamp_from_storage(
                observation["observed_at"], "observed_at"
            ),
            valid_from=(
                _timestamp_from_storage(observation["valid_from"], "valid_from")
                if observation.get("valid_from") is not None
                else None
            ),
            valid_to=(
                _timestamp_from_storage(observation["valid_to"], "valid_to")
                if observation.get("valid_to") is not None
                else None
            ),
            created_at=_timestamp_from_storage(
                observation["created_at"], "created_at"
            ),
            updated_at=_timestamp_from_storage(
                observation["updated_at"], "updated_at"
            ),
        )
    except Exception as error:
        raise StorageError(
            f"Cannot rebuild memory from invalid remembered event {event.id!r} in {path}: {error}"
        ) from error


def _rebuild(path: Path, scope: MemoryScope) -> int:
    """Atomically rebuild one exact scope's projections from immutable events."""
    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    """
                    SELECT * FROM memory_events
                    WHERE scope_key = ?
                    ORDER BY occurred_at ASC, created_at ASC, id ASC
                    """,
                    (scope.scope_key,),
                ).fetchall()
                events = [_row_to_event(path, row) for row in rows]
                for event in events:
                    if event.scope != scope:
                        raise StorageError(
                            f"Event {event.id!r} scope fields do not match requested scope"
                        )
                    if event.event_type in {
                        MemoryEventType.REMEMBERED,
                        MemoryEventType.FORGOTTEN,
                    } and event.payload.get("memory_id") != event.memory_id:
                        raise StorageError(
                            f"Event {event.id!r} payload memory_id does not match column"
                        )
                event_memory_ids = tuple(
                    dict.fromkeys(
                        event.memory_id
                        for event in events
                        if event.memory_id is not None
                    )
                )
                if event_memory_ids:
                    event_placeholders = ", ".join("?" for _ in event_memory_ids)
                    foreign_event = connection.execute(
                        f"""
                        SELECT id, memory_id FROM memory_events
                        WHERE memory_id IN ({event_placeholders}) AND scope_key != ?
                        LIMIT 1
                        """,
                        (*event_memory_ids, scope.scope_key),
                    ).fetchone()
                    if foreign_event is not None:
                        raise StorageError(
                            f"Memory {foreign_event['memory_id']!r} also belongs to a foreign event scope"
                        )
                    foreign_owner = connection.execute(
                        f"""
                        SELECT id FROM memories
                        WHERE id IN ({event_placeholders}) AND scope_key != ?
                        LIMIT 1
                        """,
                        (*event_memory_ids, scope.scope_key),
                    ).fetchone()
                    if foreign_owner is not None:
                        raise StorageError(
                            f"Memory {foreign_owner['id']!r} is owned by another scope"
                        )
                records: dict[str, MemoryRecord] = {}
                histories: list[
                    tuple[MemoryEvent, MemoryStatus, Mapping[str, Any]]
                ] = []
                for event in events:
                    if event.event_type is MemoryEventType.REMEMBERED:
                        record = _record_from_remembered_event(path, event)
                        if record.id in records:
                            raise StorageError(
                                f"Duplicate remembered event for memory {record.id!r} in {path}"
                            )
                        records[record.id] = record
                        histories.append((event, MemoryStatus.ACTIVE, {}))
                    elif event.event_type is MemoryEventType.FORGOTTEN:
                        if event.memory_id is None or event.memory_id not in records:
                            raise StorageError(
                                f"Forgotten event {event.id!r} has no remembered source in {path}"
                            )
                        records[event.memory_id] = replace(
                            records[event.memory_id],
                            status=MemoryStatus.FORGOTTEN,
                            updated_at=event.occurred_at,
                        )
                        histories.append(
                            (
                                event,
                                MemoryStatus.FORGOTTEN,
                                {"reason": event.payload.get("reason")},
                            )
                        )
                    else:
                        raise StorageError(
                            f"Unsupported event {event.event_type.value!r} while rebuilding {path}"
                        )

                scope_projection_ids = tuple(
                    row["id"]
                    for row in connection.execute(
                        "SELECT id FROM memories WHERE scope_key = ?",
                        (scope.scope_key,),
                    ).fetchall()
                )
                cleanup_ids = tuple(
                    dict.fromkeys((*scope_projection_ids, *event_memory_ids))
                )
                if cleanup_ids:
                    cleanup_placeholders = ", ".join("?" for _ in cleanup_ids)
                    connection.execute(
                        f"DELETE FROM memory_fts WHERE memory_id IN ({cleanup_placeholders})",
                        cleanup_ids,
                    )
                    connection.execute(
                        f"DELETE FROM memory_history WHERE memory_id IN ({cleanup_placeholders})",
                        cleanup_ids,
                    )
                connection.execute(
                    "DELETE FROM memories WHERE scope_key = ?",
                    (scope.scope_key,),
                )

                for record in records.values():
                    connection.execute(
                        """
                        INSERT INTO memories (
                            id, scope_key, tenant_id, user_id, agent_id, project_id,
                            session_id, kind, content, metadata_json, status,
                            confidence, observed_at, valid_from, valid_to, created_at,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.id,
                            record.scope.scope_key,
                            record.scope.tenant_id,
                            record.scope.user_id,
                            record.scope.agent_id,
                            record.scope.project_id,
                            record.scope.session_id,
                            record.kind.value,
                            record.content,
                            _canonical_json(record.metadata),
                            record.status.value,
                            record.confidence,
                            _timestamp_to_storage(record.observed_at),
                            (
                                _timestamp_to_storage(record.valid_from)
                                if record.valid_from is not None
                                else None
                            ),
                            (
                                _timestamp_to_storage(record.valid_to)
                                if record.valid_to is not None
                                else None
                            ),
                            _timestamp_to_storage(record.created_at),
                            _timestamp_to_storage(record.updated_at),
                        ),
                    )
                    if record.status is MemoryStatus.ACTIVE:
                        connection.execute(
                            "INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)",
                            (record.id, record.content),
                        )

                for event, status, details in histories:
                    connection.execute(
                        """
                        INSERT INTO memory_history (
                            id, memory_id, event_id, action, status, details_json,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"{event.id}:history",
                            event.memory_id,
                            event.id,
                            event.event_type.value,
                            status.value,
                            _canonical_json(details),
                            _timestamp_to_storage(event.created_at),
                        ),
                    )
                connection.commit()
                return len(records)
            except Exception:
                connection.rollback()
                raise
    except sqlite3.Error as error:
        raise StorageError(f"Failed to rebuild memory projections in {path}: {error}") from error


def _list(
    path: Path,
    scope: MemoryScope,
    *,
    kind: Optional[MemoryKind],
    status: MemoryStatus,
    limit: int,
) -> list[MemoryRecord]:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise ValidationError("limit must be between 1 and 1000")
    sql = "SELECT * FROM memories WHERE scope_key = ? AND status = ?"
    parameters: list[Any] = [scope.scope_key, status.value]
    if kind is not None:
        sql += " AND kind = ?"
        parameters.append(kind.value)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    parameters.append(limit)
    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, parameters).fetchall()
    except sqlite3.Error as error:
        raise StorageError(f"Failed to list memories from {path}: {error}") from error
    return [_row_to_record(path, row) for row in rows]


def _fts_match_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValidationError("query must be a string")
    tokens = re.findall(r"\w+", query.casefold())
    if not tokens:
        raise ValidationError("query must contain at least one searchable word")
    return " OR ".join(f'"{token}"' for token in tokens)


def _normalize_memory_kinds(kinds: Sequence[MemoryKind]) -> list[MemoryKind]:
    normalized_kinds: list[MemoryKind] = []
    try:
        for kind in kinds:
            normalized_kind = MemoryKind(kind)
            if normalized_kind not in normalized_kinds:
                normalized_kinds.append(normalized_kind)
    except (TypeError, ValueError) as error:
        raise ValidationError("kinds must contain only valid MemoryKind values") from error
    return normalized_kinds


def _legacy_compatible_timestamp_sql(column: str) -> str:
    return (
        f"CASE WHEN length({column}) = 20 AND substr({column}, 20, 1) = 'Z' "
        f"THEN substr({column}, 1, 19) || '.000000Z' ELSE {column} END"
    )


def _rows_to_recall_results(
    path: Path,
    rows: list[sqlite3.Row],
    limit: int,
) -> list[RecallResult]:
    results: list[RecallResult] = []
    seen_memory_ids: set[str] = set()
    for row in rows:
        memory_id = row["id"]
        if memory_id in seen_memory_ids:
            continue
        seen_memory_ids.add(memory_id)
        rank = len(results) + 1
        reciprocal_rank = 1.0 / rank
        results.append(
            RecallResult(
                memory=_row_to_record(path, row),
                score=reciprocal_rank,
                score_components={
                    "bm25_rank": reciprocal_rank,
                    "bm25_raw": float(row["bm25_raw"]),
                },
                reason="Matched lexical terms in the authorized scope",
                evidence_ids=(row["source_event_id"],),
            )
        )
        if len(results) == limit:
            break
    return results


def _recall(
    path: Path,
    scope: MemoryScope,
    query: str,
    *,
    kinds: Sequence[MemoryKind],
    limit: int,
    as_of: Optional[str],
) -> list[RecallResult]:
    match_query = _fts_match_query(query)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise ValidationError("limit must be between 1 and 1000")
    normalized_kinds = _normalize_memory_kinds(kinds)

    as_of_timestamp = _timestamp_to_storage(
        _timestamp_from_input(as_of, "as_of") or utc_now()
    )
    valid_from_sql = _legacy_compatible_timestamp_sql("memories.valid_from")
    valid_to_sql = _legacy_compatible_timestamp_sql("memories.valid_to")
    sql = f"""
        SELECT memories.*, bm25(memory_fts) AS bm25_raw,
               (
                   SELECT source_event.id
                   FROM memory_events AS source_event
                   WHERE source_event.memory_id = memories.id
                     AND source_event.scope_key = memories.scope_key
                     AND source_event.event_type = 'remembered'
                   ORDER BY source_event.occurred_at ASC, source_event.id ASC
                   LIMIT 1
               ) AS source_event_id
        FROM memory_fts
        JOIN memories ON memories.id = memory_fts.memory_id
        WHERE memory_fts MATCH ?
          AND memories.scope_key = ?
          AND memories.status = ?
          AND (memories.valid_from IS NULL OR {valid_from_sql} <= ?)
          AND (memories.valid_to IS NULL OR {valid_to_sql} > ?)
    """
    parameters: list[Any] = [
        match_query,
        scope.scope_key,
        MemoryStatus.ACTIVE.value,
        as_of_timestamp,
        as_of_timestamp,
    ]
    if normalized_kinds:
        placeholders = ", ".join("?" for _ in normalized_kinds)
        sql += f" AND memories.kind IN ({placeholders})"
        parameters.extend(kind.value for kind in normalized_kinds)
    sql += " ORDER BY bm25(memory_fts) ASC, memories.id ASC LIMIT ?"
    parameters.append(limit * 3)

    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, parameters).fetchall()
    except sqlite3.Error as error:
        raise StorageError(f"Failed to recall memories from {path}: {error}") from error
    if any(row["source_event_id"] is None for row in rows):
        raise StorageError(f"Recall projection in {path} has no remembered source event")
    return _rows_to_recall_results(path, rows, limit)


def _history(
    path: Path,
    scope: MemoryScope,
    memory_id: str,
) -> list[MemoryHistoryEntry]:
    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            exists = connection.execute(
                "SELECT 1 FROM memories WHERE scope_key = ? AND id = ?",
                (scope.scope_key, memory_id),
            ).fetchone()
            if exists is None:
                raise MemoryNotFoundError(
                    f"Memory {memory_id!r} was not found in this scope"
                )
            rows = connection.execute(
                """
                SELECT history.*
                FROM memory_history AS history
                JOIN memories AS memory ON memory.id = history.memory_id
                WHERE memory.scope_key = ? AND history.memory_id = ?
                ORDER BY history.created_at ASC, history.id ASC
                """,
                (scope.scope_key, memory_id),
            ).fetchall()
    except sqlite3.Error as error:
        raise StorageError(f"Failed to read memory history from {path}: {error}") from error
    return [_row_to_history(path, row) for row in rows]


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

    async def _run(
        self,
        worker: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        async with self._lifecycle_lock:
            if self._closed:
                raise ClosedStoreError("SQLite memory store is closed")
            worker_task = asyncio.create_task(
                asyncio.to_thread(worker, self._path, *args, **kwargs)
            )
            try:
                return await asyncio.shield(worker_task)
            except asyncio.CancelledError as cancellation:
                while not worker_task.done():
                    try:
                        await asyncio.shield(worker_task)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                try:
                    worker_task.result()
                except BaseException:
                    # Cancellation remains caller-visible, but the worker is
                    # observed before the lifecycle lock is released.
                    pass
                raise cancellation

    async def remember(
        self,
        scope: MemoryScope,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.OBSERVATION,
        metadata: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        confidence: float = 1.0,
        observed_at: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> MemoryRecord:
        return await self._run(
            _remember,
            scope,
            content,
            kind=kind,
            metadata=metadata,
            idempotency_key=idempotency_key,
            confidence=confidence,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
        )

    async def get(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        include_forgotten: bool = False,
    ) -> MemoryRecord:
        return await self._run(
            _get,
            scope,
            memory_id,
            include_forgotten=include_forgotten,
        )

    async def list(
        self,
        scope: MemoryScope,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
    ) -> builtins.list[MemoryRecord]:
        return await self._run(
            _list,
            scope,
            kind=kind,
            status=status,
            limit=limit,
        )

    async def history(
        self,
        scope: MemoryScope,
        memory_id: str,
    ) -> builtins.list[MemoryHistoryEntry]:
        return await self._run(_history, scope, memory_id)

    async def forget(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        reason: Optional[str] = None,
    ) -> MemoryRecord:
        return await self._run(
            _forget,
            scope,
            memory_id,
            reason=reason,
        )

    async def rebuild(self, scope: MemoryScope) -> int:
        return await self._run(_rebuild, scope)

    async def recall(
        self,
        scope: MemoryScope,
        query: str,
        *,
        kinds: Sequence[MemoryKind] = (),
        limit: int = 10,
        as_of: Optional[str] = None,
    ) -> builtins.list[RecallResult]:
        return await self._run(
            _recall,
            scope,
            query,
            kinds=kinds,
            limit=limit,
            as_of=as_of,
        )
