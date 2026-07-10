"""Versioned SQLite storage foundation for agent memory."""

from __future__ import annotations

import asyncio
import builtins
import json
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
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
    MemoryReceipt,
    MemoryRelation,
    MemoryScope,
    MemoryStatus,
    RecallResult,
    utc_now,
)
from .schema import initialize_or_migrate
from .timeline import (
    build_superseded_payload,
    normalize_timeline_query,
    parse_superseded_payload,
)

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


def _row_to_relation(
    path: Path,
    row: sqlite3.Row,
    scope: MemoryScope,
) -> MemoryRelation:
    try:
        return MemoryRelation(
            id=row["id"],
            scope=scope,
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            valid_from=_timestamp_from_storage(row["valid_from"], "valid_from"),
            valid_to=(
                _timestamp_from_storage(row["valid_to"], "valid_to")
                if row["valid_to"] is not None
                else None
            ),
            confidence=row["confidence"],
            event_id=row["event_id"],
            created_at=_timestamp_from_storage(row["created_at"], "created_at"),
        )
    except Exception as error:
        raise StorageError(f"Failed to hydrate memory relation from {path}: {error}") from error


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


def _timeline_validity_value_sql(column: str) -> str:
    """Use lifecycle snapshots after transitions and legacy projection values otherwise."""
    return (
        f"CASE WHEN lifecycle.known_to IS NOT NULL THEN lifecycle.{column} "
        f"ELSE memories.{column} END"
    )


def _row_to_lifecycle_record(path: Path, row: sqlite3.Row) -> MemoryRecord:
    try:
        return replace(
            _row_to_record(path, row),
            status=MemoryStatus(row["lifecycle_status"]),
            valid_from=(
                _timestamp_from_storage(row["lifecycle_valid_from"], "valid_from")
                if row["lifecycle_valid_from"] is not None
                else None
            ),
            valid_to=(
                _timestamp_from_storage(row["lifecycle_valid_to"], "valid_to")
                if row["lifecycle_valid_to"] is not None
                else None
            ),
        )
    except Exception as error:
        if isinstance(error, StorageError):
            raise
        raise StorageError(f"Failed to hydrate memory lifecycle from {path}: {error}") from error


def _unique_recall_rows(rows: list[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
    selected: list[sqlite3.Row] = []
    seen_memory_ids: set[str] = set()
    for row in rows:
        if row["id"] in seen_memory_ids:
            continue
        seen_memory_ids.add(row["id"])
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


@dataclass(frozen=True, slots=True)
class _EvidenceLineage:
    memory_ids: tuple[str, ...]
    event_ids: tuple[str, ...]


def _recall_evidence_ids(
    connection: sqlite3.Connection,
    path: Path,
    scope: MemoryScope,
    rows: list[sqlite3.Row],
    known_at: str,
) -> dict[str, tuple[str, ...]]:
    lineages = _load_evidence_lineages(
        connection,
        path,
        scope,
        {
            row["id"]: row["lifecycle_event_id"]
            for row in rows
        },
        known_at,
    )
    return {
        memory_id: lineage.event_ids
        for memory_id, lineage in lineages.items()
    }


def _rows_to_recall_results(
    path: Path,
    rows: list[sqlite3.Row],
    evidence_ids: Mapping[str, tuple[str, ...]],
) -> list[RecallResult]:
    results: list[RecallResult] = []
    for row in rows:
        rank = len(results) + 1
        reciprocal_rank = 1.0 / rank
        results.append(
            RecallResult(
                memory=_row_to_lifecycle_record(path, row),
                score=reciprocal_rank,
                score_components={
                    "bm25_rank": reciprocal_rank,
                    "bm25_raw": float(row["bm25_raw"]),
                },
                reason="Matched lexical terms in the authorized scope",
                evidence_ids=evidence_ids[row["id"]],
            )
        )
    return results


def _recall(
    path: Path,
    scope: MemoryScope,
    query: str,
    *,
    kinds: Sequence[MemoryKind],
    limit: int,
    as_of: Optional[str],
    valid_at: Optional[str],
    known_at: Optional[str],
) -> list[RecallResult]:
    match_query = _fts_match_query(query)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise ValidationError("limit must be between 1 and 1000")
    normalized_kinds = _normalize_memory_kinds(kinds)

    valid_timestamp, known_timestamp = normalize_timeline_query(
        as_of=as_of,
        valid_at=valid_at,
        known_at=known_at,
        now=utc_now(),
    )
    known_from_sql = _legacy_compatible_timestamp_sql("lifecycle.known_from")
    known_to_sql = _legacy_compatible_timestamp_sql("lifecycle.known_to")
    validity_from_value_sql = _timeline_validity_value_sql("valid_from")
    validity_to_value_sql = _timeline_validity_value_sql("valid_to")
    valid_from_sql = _legacy_compatible_timestamp_sql(validity_from_value_sql)
    valid_to_sql = _legacy_compatible_timestamp_sql(validity_to_value_sql)
    sql = f"""
        SELECT memories.*, bm25(memory_fts) AS bm25_raw,
               lifecycle.status AS lifecycle_status,
               {validity_from_value_sql} AS lifecycle_valid_from,
               {validity_to_value_sql} AS lifecycle_valid_to,
               lifecycle.event_id AS lifecycle_event_id
        FROM memory_fts
        JOIN memories ON memories.id = memory_fts.memory_id
        JOIN memory_lifecycle AS lifecycle
          ON lifecycle.memory_id = memories.id
         AND lifecycle.scope_key = memories.scope_key
        WHERE memory_fts MATCH ?
          AND memories.scope_key = ?
          AND lifecycle.status IN (?, ?)
          AND {known_from_sql} <= ?
          AND (lifecycle.known_to IS NULL OR {known_to_sql} > ?)
          AND ({validity_from_value_sql} IS NULL OR {valid_from_sql} <= ?)
          AND ({validity_to_value_sql} IS NULL OR {valid_to_sql} > ?)
    """
    parameters: list[Any] = [
        match_query,
        scope.scope_key,
        MemoryStatus.ACTIVE.value,
        MemoryStatus.SUPERSEDED.value,
        known_timestamp,
        known_timestamp,
        valid_timestamp,
        valid_timestamp,
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
            selected_rows = _unique_recall_rows(rows, limit)
            evidence_ids = _recall_evidence_ids(
                connection,
                path,
                scope,
                selected_rows,
                known_timestamp,
            )
    except sqlite3.Error as error:
        raise StorageError(f"Failed to recall memories from {path}: {error}") from error
    return _rows_to_recall_results(path, selected_rows, evidence_ids)


def _require_receipt_memory_scope(
    connection: sqlite3.Connection,
    path: Path,
    scope: MemoryScope,
    memory_id: str,
) -> MemoryRecord:
    row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        raise StorageError(f"Receipt provenance in {path} refers to a missing memory")
    record = _row_to_record(path, row)
    if record.scope != scope:
        raise StorageError(f"Receipt provenance in {path} crosses memory scope")
    return record


def _require_receipt_event_scope(
    connection: sqlite3.Connection,
    path: Path,
    scope: MemoryScope,
    event_id: str,
) -> MemoryEvent:
    row = connection.execute("SELECT * FROM memory_events WHERE id = ?", (event_id,)).fetchone()
    if row is None:
        raise StorageError(f"Receipt provenance in {path} refers to a missing event")
    event = _row_to_event(path, row)
    if event.scope != scope:
        raise StorageError(f"Receipt provenance in {path} crosses event scope")
    return event


def _load_evidence_lineages(
    connection: sqlite3.Connection,
    path: Path,
    scope: MemoryScope,
    root_event_ids: Mapping[str, str],
    known_at: str,
) -> dict[str, _EvidenceLineage]:
    """Hydrate complete upstream evidence DAGs for one bitemporal knowledge view."""
    if not root_event_ids:
        return {}
    known_instant = _timestamp_from_storage(known_at, "known_at")
    memory_cache: dict[str, MemoryRecord] = {}
    event_cache: dict[str, MemoryEvent] = {}
    edge_cache: dict[str, tuple[tuple[str, str], ...]] = {}

    def require_memory(memory_id: str) -> MemoryRecord:
        if memory_id not in memory_cache:
            memory_cache[memory_id] = _require_receipt_memory_scope(
                connection,
                path,
                scope,
                memory_id,
            )
        return memory_cache[memory_id]

    def require_event(event_id: str) -> MemoryEvent:
        if event_id not in event_cache:
            event_cache[event_id] = _require_receipt_event_scope(
                connection,
                path,
                scope,
                event_id,
            )
        return event_cache[event_id]

    def load_edges(memory_id: str) -> tuple[tuple[str, str], ...]:
        if memory_id in edge_cache:
            return edge_cache[memory_id]
        require_memory(memory_id)
        evidence_rows = connection.execute(
            """
            SELECT *
            FROM memory_evidence
            WHERE memory_id = ?
            ORDER BY created_at ASC, event_id ASC, source_memory_id ASC
            """,
            (memory_id,),
        ).fetchall()
        edges: list[tuple[str, str, datetime, datetime]] = []
        seen_edges: set[tuple[str, str]] = set()
        for evidence in evidence_rows:
            if evidence["scope_key"] != scope.scope_key:
                raise StorageError(f"Memory evidence provenance in {path} crosses scope")
            event = require_event(evidence["event_id"])
            if event.occurred_at > known_instant:
                continue
            source_memory_id = evidence["source_memory_id"]
            require_memory(source_memory_id)
            if event.event_type is not MemoryEventType.SUPERSEDED:
                raise StorageError(
                    f"Memory evidence provenance in {path} has unsupported immutable evidence"
                )
            replay = parse_superseded_payload(event, path=path)
            if (
                replay.evidence_memory_id != memory_id
                or replay.evidence_source_memory_id != source_memory_id
                or replay.relation_type != evidence["relation"]
            ):
                raise StorageError(
                    f"Memory evidence provenance in {path} has a malformed evidence edge"
                )
            edge = (source_memory_id, event.id)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            edges.append((source_memory_id, event.id, event.occurred_at, event.created_at))
        edges.sort(key=lambda edge: (edge[2], edge[3], edge[1], edge[0]))
        edge_cache[memory_id] = tuple((edge[0], edge[1]) for edge in edges)
        return edge_cache[memory_id]

    lineages: dict[str, _EvidenceLineage] = {}
    for root_memory_id, root_event_id in root_event_ids.items():
        require_memory(root_memory_id)
        root_event = require_event(root_event_id)
        if root_event.occurred_at > known_instant:
            raise StorageError(
                f"Memory evidence provenance in {path} contains future root evidence"
            )
        visited: set[str] = set()
        active: set[str] = set()
        upstream_memory_ids: set[str] = set()
        edge_event_ids: set[str] = set()

        def visit(memory_id: str) -> None:
            if memory_id in active:
                raise StorageError(f"Memory evidence provenance in {path} contains a cycle")
            if memory_id in visited:
                return
            if len(active) >= 1000:
                raise StorageError(f"Memory evidence provenance in {path} is too deep")
            active.add(memory_id)
            for source_memory_id, event_id in load_edges(memory_id):
                if source_memory_id in active:
                    raise StorageError(
                        f"Memory evidence provenance in {path} contains a cycle"
                    )
                visit(source_memory_id)
                upstream_memory_ids.add(source_memory_id)
                edge_event_ids.add(event_id)
            active.remove(memory_id)
            visited.add(memory_id)

        try:
            visit(root_memory_id)
        except RecursionError as error:
            raise StorageError(
                f"Memory evidence provenance in {path} is too deep"
            ) from error

        all_memory_ids = {root_memory_id, *upstream_memory_ids}
        placeholders = ", ".join("?" for _ in all_memory_ids)
        remembered_rows = connection.execute(
            f"""
            SELECT *
            FROM memory_events AS event
            WHERE event.scope_key = ?
              AND event.event_type = ?
              AND event.memory_id IN ({placeholders})
              AND {_legacy_compatible_timestamp_sql('event.occurred_at')} <= ?
            ORDER BY {_legacy_compatible_timestamp_sql('event.occurred_at')} ASC,
                     {_legacy_compatible_timestamp_sql('event.created_at')} ASC,
                     event.id ASC
            """,
            [
                scope.scope_key,
                MemoryEventType.REMEMBERED.value,
                *sorted(all_memory_ids),
                known_at,
            ],
        ).fetchall()
        remembered_events = [_row_to_event(path, event_row) for event_row in remembered_rows]
        for event in remembered_events:
            if event.scope != scope:
                raise StorageError(f"Memory evidence provenance in {path} crosses event scope")
            event_cache[event.id] = event
        remembered_memory_ids = {
            event.memory_id for event in remembered_events if event.memory_id is not None
        }
        leaf_memory_ids = {
            memory_id for memory_id in all_memory_ids if not load_edges(memory_id)
        }
        if not leaf_memory_ids.issubset(remembered_memory_ids):
            raise StorageError(
                f"Memory evidence provenance in {path} has missing remembered evidence"
            )

        wanted_event_ids = {
            root_event_id,
            *edge_event_ids,
            *(event.id for event in remembered_events),
        }
        wanted_events = [require_event(event_id) for event_id in wanted_event_ids]
        wanted_events.sort(
            key=lambda event: (event.occurred_at, event.created_at, event.id)
        )
        upstream_records = [require_memory(memory_id) for memory_id in upstream_memory_ids]
        upstream_records.sort(key=lambda record: (record.created_at, record.id))
        lineages[root_memory_id] = _EvidenceLineage(
            memory_ids=tuple(record.id for record in upstream_records),
            event_ids=tuple(event.id for event in wanted_events),
        )
    return lineages


def _receipt_memory_from_lifecycle_event(
    connection: sqlite3.Connection,
    path: Path,
    scope: MemoryScope,
    row: sqlite3.Row,
) -> MemoryRecord:
    timeline_record = _row_to_lifecycle_record(path, row)
    event = _require_receipt_event_scope(
        connection,
        path,
        scope,
        row["lifecycle_event_id"],
    )
    if event.event_type is MemoryEventType.REMEMBERED:
        snapshot = _record_from_remembered_event(path, event)
    elif event.event_type is MemoryEventType.SUPERSEDED:
        replay = parse_superseded_payload(event, path=path)
        if replay.source.id == timeline_record.id:
            snapshot = replay.source
        elif replay.replacement.id == timeline_record.id:
            snapshot = replay.replacement
        else:
            raise StorageError(
                f"Receipt lifecycle in {path} does not match its immutable event"
            )
    else:
        raise StorageError(
            f"Receipt lifecycle in {path} has unsupported immutable evidence"
        )
    if snapshot.scope != scope or snapshot.id != timeline_record.id:
        raise StorageError(f"Receipt lifecycle in {path} crosses memory scope")
    return replace(
        snapshot,
        status=timeline_record.status,
        valid_from=timeline_record.valid_from,
        valid_to=timeline_record.valid_to,
    )


def _receipt_explanation(
    memory: MemoryRecord,
    relations: Sequence[MemoryRelation],
    evidence_event_ids: Sequence[str],
) -> str:
    supersedes = next(
        (
            relation
            for relation in relations
            if relation.source_id == memory.id and relation.relation_type == "supersedes"
        ),
        None,
    )
    if supersedes is not None:
        return (
            f"Memory {memory.id} supersedes memory {supersedes.target_id} from "
            f"{_timestamp_to_storage(supersedes.valid_from)} via event "
            f"{supersedes.event_id}; confidence {memory.confidence:.6f}."
        )
    source_event = evidence_event_ids[0]
    return (
        f"Memory {memory.id} is {memory.status.value} in the requested timeline via "
        f"event {source_event}; confidence {memory.confidence:.6f}."
    )


def _explain(
    path: Path,
    scope: MemoryScope,
    memory_id: str,
    *,
    valid_at: Optional[str],
    known_at: Optional[str],
) -> MemoryReceipt:
    valid_timestamp, known_timestamp = normalize_timeline_query(
        as_of=None,
        valid_at=valid_at,
        known_at=known_at,
        now=utc_now(),
    )
    known_from_sql = _legacy_compatible_timestamp_sql("lifecycle.known_from")
    known_to_sql = _legacy_compatible_timestamp_sql("lifecycle.known_to")
    validity_from_value_sql = _timeline_validity_value_sql("valid_from")
    validity_to_value_sql = _timeline_validity_value_sql("valid_to")
    valid_from_sql = _legacy_compatible_timestamp_sql(validity_from_value_sql)
    valid_to_sql = _legacy_compatible_timestamp_sql(validity_to_value_sql)

    try:
        with closing(_connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("BEGIN")
                row = connection.execute(
                    f"""
                    SELECT memories.*,
                           lifecycle.status AS lifecycle_status,
                           {validity_from_value_sql} AS lifecycle_valid_from,
                           {validity_to_value_sql} AS lifecycle_valid_to,
                           lifecycle.event_id AS lifecycle_event_id
                    FROM memories
                    JOIN memory_lifecycle AS lifecycle
                      ON lifecycle.memory_id = memories.id
                     AND lifecycle.scope_key = memories.scope_key
                    WHERE memories.scope_key = ?
                      AND memories.id = ?
                      AND lifecycle.status IN (?, ?)
                      AND {known_from_sql} <= ?
                      AND (lifecycle.known_to IS NULL OR {known_to_sql} > ?)
                      AND ({validity_from_value_sql} IS NULL OR {valid_from_sql} <= ?)
                      AND ({validity_to_value_sql} IS NULL OR {valid_to_sql} > ?)
                    ORDER BY lifecycle.known_from DESC, lifecycle.status ASC
                    LIMIT 1
                    """,
                    (
                        scope.scope_key,
                        memory_id,
                        MemoryStatus.ACTIVE.value,
                        MemoryStatus.SUPERSEDED.value,
                        known_timestamp,
                        known_timestamp,
                        valid_timestamp,
                        valid_timestamp,
                    ),
                ).fetchone()
                if row is None:
                    raise MemoryNotFoundError(
                        f"Memory {memory_id!r} was not found in this scope"
                    )
                memory = _receipt_memory_from_lifecycle_event(
                    connection,
                    path,
                    scope,
                    row,
                )

                lineage = _load_evidence_lineages(
                    connection,
                    path,
                    scope,
                    {memory.id: row["lifecycle_event_id"]},
                    known_timestamp,
                )[memory.id]
                evidence_memory_ids = list(lineage.memory_ids)
                evidence_event_ids = set(lineage.event_ids)
                lineage_memory_ids = [*evidence_memory_ids, memory.id]
                lineage_placeholders = ", ".join("?" for _ in lineage_memory_ids)
                event_placeholders = ", ".join("?" for _ in evidence_event_ids)
                relation_valid_from_sql = _legacy_compatible_timestamp_sql(
                    "relation.valid_from"
                )
                relation_valid_to_sql = _legacy_compatible_timestamp_sql(
                    "relation.valid_to"
                )
                relation_rows = connection.execute(
                    f"""
                    SELECT relation.*
                    FROM memory_relations AS relation
                    JOIN memory_events AS relation_event
                      ON relation_event.id = relation.event_id
                     AND relation_event.scope_key = relation.scope_key
                    WHERE relation.scope_key = ?
                      AND relation.event_id IN ({event_placeholders})
                      AND relation_event.scope_key = ?
                      AND {_legacy_compatible_timestamp_sql('relation_event.occurred_at')} <= ?
                      AND (
                          relation.valid_from IS NULL
                          OR {relation_valid_from_sql} <= ?
                      )
                      AND (
                          relation.valid_to IS NULL
                          OR {relation_valid_to_sql} > ?
                      )
                    ORDER BY {_legacy_compatible_timestamp_sql('relation_event.occurred_at')} ASC,
                             {_legacy_compatible_timestamp_sql('relation_event.created_at')} ASC,
                             relation.id ASC
                    """,
                    [
                        scope.scope_key,
                        *sorted(evidence_event_ids),
                        scope.scope_key,
                        known_timestamp,
                        valid_timestamp,
                        valid_timestamp,
                    ],
                ).fetchall()
                relations: list[MemoryRelation] = []
                lineage_memory_id_set = set(lineage_memory_ids)
                for relation_row in relation_rows:
                    if (
                        relation_row["source_id"] not in lineage_memory_id_set
                        or relation_row["target_id"] not in lineage_memory_id_set
                    ):
                        raise StorageError(
                            f"Receipt provenance in {path} has malformed relation endpoints"
                        )
                    _require_receipt_memory_scope(
                        connection,
                        path,
                        scope,
                        relation_row["source_id"],
                    )
                    _require_receipt_memory_scope(
                        connection,
                        path,
                        scope,
                        relation_row["target_id"],
                    )
                    _require_receipt_event_scope(
                        connection,
                        path,
                        scope,
                        relation_row["event_id"],
                    )
                    relation = _row_to_relation(path, relation_row, scope)
                    relations.append(relation)
                    evidence_event_ids.add(relation.event_id)

                history_rows = connection.execute(
                    f"""
                    SELECT history.*
                    FROM memory_history AS history
                    WHERE history.memory_id IN ({lineage_placeholders})
                      AND history.event_id IN ({event_placeholders})
                      AND {_legacy_compatible_timestamp_sql('history.created_at')} <= ?
                    ORDER BY {_legacy_compatible_timestamp_sql('history.created_at')} ASC,
                             history.id ASC
                    """,
                    [
                        *lineage_memory_ids,
                        *sorted(evidence_event_ids),
                        known_timestamp,
                    ],
                ).fetchall()
                history: list[MemoryHistoryEntry] = []
                for history_row in history_rows:
                    _require_receipt_memory_scope(
                        connection,
                        path,
                        scope,
                        history_row["memory_id"],
                    )
                    _require_receipt_event_scope(
                        connection,
                        path,
                        scope,
                        history_row["event_id"],
                    )
                    history_entry = _row_to_history(path, history_row)
                    history.append(history_entry)
                    evidence_event_ids.add(history_entry.event_id)

                event_placeholders = ", ".join("?" for _ in evidence_event_ids)
                event_rows = connection.execute(
                    f"""
                    SELECT *
                    FROM memory_events AS event
                    WHERE event.scope_key = ?
                      AND event.id IN ({event_placeholders})
                      AND {_legacy_compatible_timestamp_sql('event.occurred_at')} <= ?
                    ORDER BY {_legacy_compatible_timestamp_sql('event.occurred_at')} ASC,
                             {_legacy_compatible_timestamp_sql('event.created_at')} ASC,
                             event.id ASC
                    """,
                    [scope.scope_key, *sorted(evidence_event_ids), known_timestamp],
                ).fetchall()
                events = [_row_to_event(path, event_row) for event_row in event_rows]
                if any(event.scope != scope for event in events):
                    raise StorageError(f"Receipt provenance in {path} crosses event scope")
                ordered_event_ids = tuple(event.id for event in events)
                if set(ordered_event_ids) != evidence_event_ids:
                    raise StorageError(
                        f"Receipt provenance in {path} has missing immutable evidence"
                    )

                receipt = MemoryReceipt(
                    memory=memory,
                    evidence_event_ids=ordered_event_ids,
                    evidence_memory_ids=evidence_memory_ids,
                    relations=relations,
                    history=history,
                    explanation=_receipt_explanation(
                        memory,
                        relations,
                        ordered_event_ids,
                    ),
                )
                connection.commit()
                return receipt
            except Exception:
                connection.rollback()
                raise
    except sqlite3.Error as error:
        raise StorageError(f"Failed to explain memory from {path}: {error}") from error


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
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
    ) -> builtins.list[RecallResult]:
        return await self._run(
            _recall,
            scope,
            query,
            kinds=kinds,
            limit=limit,
            as_of=as_of,
            valid_at=valid_at,
            known_at=known_at,
        )

    async def explain(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
    ) -> MemoryReceipt:
        return await self._run(
            _explain,
            scope,
            memory_id,
            valid_at=valid_at,
            known_at=known_at,
        )
