"""Private SQLite value and row codecs for agent memory persistence."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .errors import StorageError, ValidationError
from .models import (
    MemoryEvent,
    MemoryEventType,
    MemoryHistoryEntry,
    MemoryKind,
    MemoryRecord,
    MemoryRelation,
    MemoryScope,
    MemoryStatus,
)


def _timestamp_to_storage(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


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
