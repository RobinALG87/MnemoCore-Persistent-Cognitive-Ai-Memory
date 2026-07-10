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

from .errors import (
    ClosedStoreError,
    MemoryConflictError,
    MemoryNotFoundError,
    StorageError,
    ValidationError,
)
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
from .schema import initialize_or_migrate
from .timeline import build_superseded_payload, parse_superseded_payload

_T = TypeVar("_T")


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


def _initialize_schema(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path)) as connection:
            initialize_or_migrate(connection, path)
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
                    INSERT INTO memory_lifecycle (
                        memory_id, scope_key, status, known_from, known_to,
                        valid_from, valid_to, event_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        scope.scope_key,
                        MemoryStatus.ACTIVE.value,
                        timestamp,
                        None,
                        valid_from_timestamp,
                        valid_to_timestamp,
                        event.id,
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
                lifecycle_updated = connection.execute(
                    """
                    UPDATE memory_lifecycle
                    SET known_to = ?
                    WHERE memory_id = ? AND scope_key = ? AND status = ?
                      AND known_to IS NULL
                    """,
                    (
                        timestamp,
                        memory_id,
                        scope.scope_key,
                        MemoryStatus.ACTIVE.value,
                    ),
                )
                if lifecycle_updated.rowcount != 1:
                    raise sqlite3.DatabaseError(
                        "active memory lifecycle changed during forget"
                    )
                connection.execute(
                    """
                    INSERT INTO memory_lifecycle (
                        memory_id, scope_key, status, known_from, known_to,
                        valid_from, valid_to, event_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        scope.scope_key,
                        MemoryStatus.FORGOTTEN.value,
                        timestamp,
                        None,
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
                        event.id,
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


def _validate_supersede_idempotency_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError("idempotency_key must be a string or None")
    normalized = value.strip()
    if not normalized:
        raise ValidationError("idempotency_key must not be blank")
    if normalized != value:
        raise ValidationError("idempotency_key must be normalized")
    return normalized


def _supersede(
    path: Path,
    scope: MemoryScope,
    memory_id: str,
    content: str,
    *,
    effective_at: str,
    reason: Optional[str],
    metadata: Optional[Mapping[str, Any]],
    confidence: float,
    idempotency_key: Optional[str],
) -> MemoryRecord:
    """Atomically supersede one active fact in one exact scope."""
    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("BEGIN IMMEDIATE")

                # A well-formed retry key is deliberately consulted before any
                # validation of the new payload or source id.
                if isinstance(idempotency_key, str):
                    existing_event_row = connection.execute(
                        """
                        SELECT * FROM memory_events
                        WHERE scope_key = ? AND idempotency_key = ?
                        """,
                        (scope.scope_key, idempotency_key),
                    ).fetchone()
                    if existing_event_row is not None:
                        existing_event = _row_to_event(path, existing_event_row)
                        if existing_event.event_type is not MemoryEventType.SUPERSEDED:
                            raise MemoryConflictError(
                                "idempotency_key is already used by another memory event"
                            )
                        replay = parse_superseded_payload(existing_event, path=path)
                        replacement_row = connection.execute(
                            "SELECT * FROM memories WHERE scope_key = ? AND id = ?",
                            (scope.scope_key, replay.replacement.id),
                        ).fetchone()
                        if replacement_row is None:
                            raise StorageError(
                                "Idempotent supersession refers to a missing replacement projection"
                            )
                        replacement = _row_to_record(path, replacement_row)
                        connection.commit()
                        return replacement

                normalized_idempotency_key = _validate_supersede_idempotency_key(idempotency_key)
                if not isinstance(memory_id, str) or not memory_id.strip():
                    raise ValidationError("memory_id must be a nonblank string")
                if memory_id != memory_id.strip():
                    raise ValidationError("memory_id must be normalized")

                source_row = connection.execute(
                    "SELECT * FROM memories WHERE scope_key = ? AND id = ?",
                    (scope.scope_key, memory_id),
                ).fetchone()
                if source_row is None:
                    raise MemoryNotFoundError(f"Memory {memory_id!r} was not found in this scope")
                source_before = _row_to_record(path, source_row)
                if source_before.kind is not MemoryKind.FACT:
                    raise MemoryConflictError("Only fact memories can be superseded")
                if source_before.status is not MemoryStatus.ACTIVE:
                    raise MemoryConflictError(f"Memory {memory_id!r} is no longer active")

                boundary = _timestamp_from_input(effective_at, "effective_at")
                if boundary is None:
                    raise ValidationError("effective_at is required")
                if source_before.valid_from is not None and boundary <= source_before.valid_from:
                    raise MemoryConflictError(
                        "effective_at must be after the source valid_from boundary"
                    )
                if source_before.valid_to is not None and boundary >= source_before.valid_to:
                    raise MemoryConflictError(
                        "effective_at must be before the source valid_to boundary"
                    )

                now = utc_now()
                if now <= source_before.updated_at:
                    now = source_before.updated_at + timedelta(microseconds=1)
                timestamp = _timestamp_to_storage(now)
                effective_timestamp = _timestamp_to_storage(boundary)
                previous_valid_to = source_before.valid_to
                event_id = uuid4().hex
                replacement_id = uuid4().hex
                relation_id = f"{event_id}:relation:supersedes"

                source = replace(
                    source_before,
                    status=MemoryStatus.SUPERSEDED,
                    valid_to=boundary,
                    updated_at=now,
                )
                metadata_json = _canonical_json({} if metadata is None else metadata)
                replacement = MemoryRecord(
                    id=replacement_id,
                    scope=scope,
                    kind=MemoryKind.FACT,
                    content=content,
                    metadata=json.loads(metadata_json),
                    status=MemoryStatus.ACTIVE,
                    confidence=confidence,
                    observed_at=now,
                    valid_from=boundary,
                    valid_to=previous_valid_to,
                    created_at=now,
                    updated_at=now,
                )
                payload = build_superseded_payload(
                    source,
                    replacement,
                    reason=reason,
                    relation_id=relation_id,
                )
                event = MemoryEvent(
                    id=event_id,
                    memory_id=source.id,
                    scope=scope,
                    event_type=MemoryEventType.SUPERSEDED,
                    payload=payload,
                    idempotency_key=normalized_idempotency_key,
                    occurred_at=now,
                    created_at=now,
                )
                payload_json = _canonical_json(event.payload)
                reason_json = _canonical_json(
                    {
                        "reason": event.payload["reason"],
                        "replacement_memory_id": replacement.id,
                        "role": "source",
                    }
                )
                replacement_details_json = _canonical_json(
                    {
                        "reason": event.payload["reason"],
                        "source_memory_id": source.id,
                        "role": "replacement",
                    }
                )
                source_valid_from = (
                    _timestamp_to_storage(source.valid_from)
                    if source.valid_from is not None
                    else None
                )
                replacement_valid_to = (
                    _timestamp_to_storage(replacement.valid_to)
                    if replacement.valid_to is not None
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
                        source.id,
                        scope.scope_key,
                        scope.tenant_id,
                        scope.user_id,
                        scope.agent_id,
                        scope.project_id,
                        scope.session_id,
                        event.event_type.value,
                        payload_json,
                        normalized_idempotency_key,
                        timestamp,
                        timestamp,
                    ),
                )
                updated = connection.execute(
                    """
                    UPDATE memories
                    SET status = ?, valid_to = ?, updated_at = ?
                    WHERE scope_key = ? AND id = ? AND kind = ? AND status = ?
                    """,
                    (
                        MemoryStatus.SUPERSEDED.value,
                        effective_timestamp,
                        timestamp,
                        scope.scope_key,
                        source.id,
                        MemoryKind.FACT.value,
                        MemoryStatus.ACTIVE.value,
                    ),
                )
                if updated.rowcount != 1:
                    raise MemoryConflictError(f"Memory {source.id!r} changed during supersession")
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
                        replacement.id,
                        scope.scope_key,
                        scope.tenant_id,
                        scope.user_id,
                        scope.agent_id,
                        scope.project_id,
                        scope.session_id,
                        replacement.kind.value,
                        replacement.content,
                        metadata_json,
                        replacement.status.value,
                        replacement.confidence,
                        timestamp,
                        effective_timestamp,
                        replacement_valid_to,
                        timestamp,
                        timestamp,
                    ),
                )
                lifecycle_updated = connection.execute(
                    """
                    UPDATE memory_lifecycle
                    SET known_to = ?
                    WHERE memory_id = ? AND scope_key = ? AND status = ?
                      AND known_to IS NULL
                    """,
                    (
                        timestamp,
                        source.id,
                        scope.scope_key,
                        MemoryStatus.ACTIVE.value,
                    ),
                )
                if lifecycle_updated.rowcount != 1:
                    raise StorageError("active memory lifecycle changed during supersession")
                connection.execute(
                    """
                    INSERT INTO memory_lifecycle (
                        memory_id, scope_key, status, known_from, known_to,
                        valid_from, valid_to, event_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source.id,
                        scope.scope_key,
                        MemoryStatus.SUPERSEDED.value,
                        timestamp,
                        None,
                        source_valid_from,
                        effective_timestamp,
                        event.id,
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO memory_lifecycle (
                        memory_id, scope_key, status, known_from, known_to,
                        valid_from, valid_to, event_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        replacement.id,
                        scope.scope_key,
                        MemoryStatus.ACTIVE.value,
                        timestamp,
                        None,
                        effective_timestamp,
                        replacement_valid_to,
                        event.id,
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
                        f"{event.id}:history:source",
                        source.id,
                        event.id,
                        event.event_type.value,
                        MemoryStatus.SUPERSEDED.value,
                        reason_json,
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
                        f"{event.id}:history:replacement",
                        replacement.id,
                        event.id,
                        event.event_type.value,
                        MemoryStatus.ACTIVE.value,
                        replacement_details_json,
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO memory_evidence (
                        memory_id, source_memory_id, event_id, scope_key,
                        relation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        replacement.id,
                        source.id,
                        event.id,
                        scope.scope_key,
                        "supersedes",
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO memory_relations (
                        id, scope_key, source_id, target_id, relation_type,
                        valid_from, valid_to, confidence, event_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relation_id,
                        scope.scope_key,
                        replacement.id,
                        source.id,
                        "supersedes",
                        effective_timestamp,
                        None,
                        replacement.confidence,
                        event.id,
                        timestamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)",
                    (replacement.id, replacement.content),
                )
                connection.commit()
                return replacement
            except Exception:
                connection.rollback()
                raise
    except sqlite3.Error as error:
        raise StorageError(f"Failed to supersede memory in {path}: {error}") from error


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
                invalid_lifecycle_owner = connection.execute(
                    """
                    SELECT lifecycle.memory_id
                    FROM memory_lifecycle AS lifecycle
                    LEFT JOIN memories AS memory ON memory.id = lifecycle.memory_id
                    LEFT JOIN memory_events AS event ON event.id = lifecycle.event_id
                    WHERE (lifecycle.scope_key = ? OR memory.scope_key = ?)
                      AND (
                          (
                              memory.id IS NOT NULL
                              AND memory.scope_key IS NOT lifecycle.scope_key
                          )
                          OR event.scope_key IS NOT lifecycle.scope_key
                      )
                    LIMIT 1
                    """,
                    (scope.scope_key, scope.scope_key),
                ).fetchone()
                if invalid_lifecycle_owner is not None:
                    raise StorageError(
                        f"Memory {invalid_lifecycle_owner['memory_id']!r} lifecycle is owned by another scope"
                    )
                records: dict[str, MemoryRecord] = {}
                histories: list[
                    tuple[MemoryEvent, MemoryStatus, Mapping[str, Any]]
                ] = []
                lifecycles: list[list[Any]] = []
                active_lifecycle_positions: dict[str, int] = {}
                for event in events:
                    if event.event_type is MemoryEventType.REMEMBERED:
                        record = _record_from_remembered_event(path, event)
                        if record.id in records:
                            raise StorageError(
                                f"Duplicate remembered event for memory {record.id!r} in {path}"
                            )
                        records[record.id] = record
                        histories.append((event, MemoryStatus.ACTIVE, {}))
                        active_lifecycle_positions[record.id] = len(lifecycles)
                        lifecycles.append(
                            [
                                record.id,
                                scope.scope_key,
                                MemoryStatus.ACTIVE.value,
                                _timestamp_to_storage(event.occurred_at),
                                None,
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
                                event.id,
                                _timestamp_to_storage(event.created_at),
                            ]
                        )
                    elif event.event_type is MemoryEventType.FORGOTTEN:
                        if (
                            event.memory_id is None
                            or event.memory_id not in records
                            or records[event.memory_id].status is not MemoryStatus.ACTIVE
                        ):
                            raise StorageError(
                                f"Forgotten event {event.id!r} has no remembered source in {path}"
                            )
                        previous = records[event.memory_id]
                        known_from = lifecycles[
                            active_lifecycle_positions[event.memory_id]
                        ][3]
                        known_to = _timestamp_to_storage(event.occurred_at)
                        if known_to <= known_from:
                            raise StorageError(
                                f"Forgotten event {event.id!r} is out of order in {path}"
                            )
                        lifecycles[
                            active_lifecycle_positions[event.memory_id]
                        ][4] = known_to
                        lifecycles.append(
                            [
                                event.memory_id,
                                scope.scope_key,
                                MemoryStatus.FORGOTTEN.value,
                                known_to,
                                None,
                                (
                                    _timestamp_to_storage(previous.valid_from)
                                    if previous.valid_from is not None
                                    else None
                                ),
                                (
                                    _timestamp_to_storage(previous.valid_to)
                                    if previous.valid_to is not None
                                    else None
                                ),
                                event.id,
                                _timestamp_to_storage(event.created_at),
                            ]
                        )
                        records[event.memory_id] = replace(
                            previous,
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
                        f"DELETE FROM memory_lifecycle WHERE memory_id IN ({cleanup_placeholders})",
                        cleanup_ids,
                    )
                connection.execute(
                    "DELETE FROM memory_lifecycle WHERE scope_key = ?",
                    (scope.scope_key,),
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

                connection.executemany(
                    """
                    INSERT INTO memory_lifecycle (
                        memory_id, scope_key, status, known_from, known_to,
                        valid_from, valid_to, event_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [tuple(row) for row in lifecycles],
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
    """SQLite store lifecycle and versioned schema initialization."""

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

    async def supersede(
        self,
        scope: MemoryScope,
        memory_id: str,
        content: str,
        *,
        effective_at: str,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        confidence: float = 1.0,
        idempotency_key: Optional[str] = None,
    ) -> MemoryRecord:
        return await self._run(
            _supersede,
            scope,
            memory_id,
            content,
            effective_at=effective_at,
            reason=reason,
            metadata=metadata,
            confidence=confidence,
            idempotency_key=idempotency_key,
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
