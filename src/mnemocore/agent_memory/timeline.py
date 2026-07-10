"""Pure policy helpers for bitemporal memory timelines and supersession replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .errors import StorageError, ValidationError
from .models import (
    MAX_IDENTIFIER_LENGTH,
    MemoryEvent,
    MemoryEventType,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
)

_RELATION_TYPE = "supersedes"
_SNAPSHOT_KEYS = {
    "id",
    "scope",
    "kind",
    "content",
    "metadata",
    "status",
    "confidence",
    "observed_at",
    "valid_from",
    "valid_to",
    "created_at",
    "updated_at",
}
_SCOPE_KEYS = {
    "scope_key",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "session_id",
}
_PAYLOAD_KEYS = {
    "source_memory_id",
    "replacement_memory_id",
    "scope_key",
    "effective_at",
    "reason",
    "source",
    "replacement",
    "relation",
    "evidence",
}


@dataclass(frozen=True, slots=True)
class SupersessionReplay:
    """Detached, validated facts required to replay one supersession event."""

    source: MemoryRecord
    replacement: MemoryRecord
    effective_at: datetime
    reason: Optional[str]
    relation_id: str
    relation_type: str
    evidence_memory_id: str
    evidence_source_memory_id: str


def _canonical_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_query_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp is not timezone-aware")
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be a valid timezone-aware ISO-8601 string") from error


def _normalize_now(now: Any) -> datetime:
    if not isinstance(now, datetime):
        raise ValidationError("now must be a timezone-aware datetime")
    try:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("timestamp is not timezone-aware")
        return now.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("now must be a timezone-aware datetime") from error


def normalize_timeline_query(
    *,
    as_of: Optional[str],
    valid_at: Optional[str],
    known_at: Optional[str],
    now: datetime,
) -> tuple[str, str]:
    """Return canonical UTC valid-time and knowledge-time query boundaries."""
    if as_of is not None and valid_at is not None:
        raise ValidationError("as_of and valid_at cannot both be supplied")
    normalized_now = _normalize_now(now)
    valid_value = valid_at if valid_at is not None else as_of
    normalized_valid = (
        _parse_query_timestamp(valid_value, "valid_at" if valid_at is not None else "as_of")
        if valid_value is not None
        else normalized_now
    )
    normalized_known = (
        _parse_query_timestamp(known_at, "known_at") if known_at is not None else normalized_now
    )
    return _canonical_timestamp(normalized_valid), _canonical_timestamp(normalized_known)


def _contains_half_open(
    instant: datetime,
    start: Optional[datetime],
    end: Optional[datetime],
) -> bool:
    """Return whether an instant lies in the half-open interval [start, end)."""
    return (start is None or start <= instant) and (end is None or instant < end)


def _normalize_reason(reason: Any) -> Optional[str]:
    if reason is None:
        return None
    if not isinstance(reason, str):
        raise ValidationError("reason must be a string or None")
    normalized = reason.strip()
    return normalized or None


def _normalize_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValidationError(f"{name} must not be blank")
    if normalized != value:
        raise ValidationError(f"{name} must be normalized")
    if len(normalized) > MAX_IDENTIFIER_LENGTH:
        raise ValidationError(f"{name} must be at most {MAX_IDENTIFIER_LENGTH} characters")
    return normalized


def _detach_json(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        detached: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{name} keys must be strings")
            detached[key] = _detach_json(item, f"{name}.{key}")
        return detached
    if isinstance(value, (tuple, list)):
        return [_detach_json(item, name) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{name} must not contain non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValidationError(f"{name} must contain only JSON-compatible values")


def _scope_snapshot(scope: MemoryScope) -> dict[str, Any]:
    return {
        "scope_key": scope.scope_key,
        "tenant_id": scope.tenant_id,
        "user_id": scope.user_id,
        "agent_id": scope.agent_id,
        "project_id": scope.project_id,
        "session_id": scope.session_id,
    }


def _record_snapshot(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "scope": _scope_snapshot(record.scope),
        "kind": record.kind.value,
        "content": record.content,
        "metadata": _detach_json(record.metadata, "metadata"),
        "status": record.status.value,
        "confidence": record.confidence,
        "observed_at": _canonical_timestamp(record.observed_at),
        "valid_from": (
            _canonical_timestamp(record.valid_from) if record.valid_from is not None else None
        ),
        "valid_to": (
            _canonical_timestamp(record.valid_to) if record.valid_to is not None else None
        ),
        "created_at": _canonical_timestamp(record.created_at),
        "updated_at": _canonical_timestamp(record.updated_at),
    }


def _validate_supersession_records(
    source: MemoryRecord,
    replacement: MemoryRecord,
) -> datetime:
    if not isinstance(source, MemoryRecord) or not isinstance(replacement, MemoryRecord):
        raise ValidationError("source and replacement must be MemoryRecord values")
    _normalize_identifier(source.id, "source memory id")
    _normalize_identifier(replacement.id, "replacement memory id")
    if source.id == replacement.id:
        raise ValidationError("source and replacement memory ids must differ")
    if source.scope != replacement.scope:
        raise ValidationError("source and replacement scope must match")
    if source.kind is not MemoryKind.FACT or replacement.kind is not MemoryKind.FACT:
        raise ValidationError("supersession snapshots must both be facts")
    if source.status is not MemoryStatus.SUPERSEDED:
        raise ValidationError("source snapshot status must be superseded")
    if replacement.status is not MemoryStatus.ACTIVE:
        raise ValidationError("replacement snapshot status must be active")
    if source.valid_to is None or replacement.valid_from is None:
        raise ValidationError("supersession snapshots require a complete effective boundary")
    if source.valid_to != replacement.valid_from:
        raise ValidationError("source and replacement boundary must match")
    boundary = source.valid_to
    if source.valid_from is not None and boundary <= source.valid_from:
        raise ValidationError("source boundary must be after valid_from")
    if replacement.valid_to is not None and boundary >= replacement.valid_to:
        raise ValidationError("replacement boundary must be before valid_to")
    return boundary


def build_superseded_payload(
    source: MemoryRecord,
    replacement: MemoryRecord,
    *,
    reason: Optional[str],
    relation_id: str,
) -> Mapping[str, Any]:
    """Build the complete immutable payload for a supersession event."""
    boundary = _validate_supersession_records(source, replacement)
    normalized_reason = _normalize_reason(reason)
    normalized_relation_id = _normalize_identifier(relation_id, "relation_id")
    return {
        "source_memory_id": source.id,
        "replacement_memory_id": replacement.id,
        "scope_key": source.scope.scope_key,
        "effective_at": _canonical_timestamp(boundary),
        "reason": normalized_reason,
        "source": _record_snapshot(source),
        "replacement": _record_snapshot(replacement),
        "relation": {
            "id": normalized_relation_id,
            "relation_type": _RELATION_TYPE,
            "source_id": replacement.id,
            "target_id": source.id,
        },
        "evidence": {
            "memory_id": replacement.id,
            "source_memory_id": source.id,
            "relation": _RELATION_TYPE,
        },
    }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected.difference(value)
    extra = set(value).difference(expected)
    if missing or extra:
        detail = ""
        if missing:
            detail += f" missing {sorted(missing)!r}"
        if extra:
            detail += f" unexpected {sorted(extra)!r}"
        raise ValidationError(f"{name} has invalid fields:{detail}")


def _parse_stored_timestamp(value: Any, name: str, *, optional: bool = False) -> Optional[datetime]:
    if value is None and optional:
        return None
    parsed = _parse_query_timestamp(value, name)
    if value != _canonical_timestamp(parsed):
        raise ValidationError(f"{name} must use canonical UTC microsecond storage")
    return parsed


def _hydrate_scope(snapshot: Any, name: str) -> MemoryScope:
    value = _require_mapping(snapshot, name)
    _require_exact_keys(value, _SCOPE_KEYS, name)
    scope = MemoryScope(
        tenant_id=value["tenant_id"],
        user_id=value["user_id"],
        agent_id=value["agent_id"],
        project_id=value["project_id"],
        session_id=value["session_id"],
    )
    for field in ("tenant_id", "user_id", "agent_id", "project_id", "session_id"):
        if value[field] != getattr(scope, field):
            raise ValidationError(f"{name}.{field} must be normalized")
    if value["scope_key"] != scope.scope_key:
        raise ValidationError(f"{name} scope_key does not match normalized scope")
    return scope


def _hydrate_record(snapshot: Any, name: str) -> MemoryRecord:
    value = _require_mapping(snapshot, name)
    _require_exact_keys(value, _SNAPSHOT_KEYS, name)
    memory_id = _normalize_identifier(value["id"], f"{name}.id")
    metadata = _detach_json(
        _require_mapping(value["metadata"], f"{name}.metadata"), f"{name}.metadata"
    )
    return MemoryRecord(
        id=memory_id,
        scope=_hydrate_scope(value["scope"], f"{name}.scope"),
        kind=MemoryKind(value["kind"]),
        content=value["content"],
        metadata=metadata,
        status=MemoryStatus(value["status"]),
        confidence=value["confidence"],
        observed_at=_parse_stored_timestamp(value["observed_at"], f"{name}.observed_at"),
        valid_from=_parse_stored_timestamp(
            value["valid_from"], f"{name}.valid_from", optional=True
        ),
        valid_to=_parse_stored_timestamp(value["valid_to"], f"{name}.valid_to", optional=True),
        created_at=_parse_stored_timestamp(value["created_at"], f"{name}.created_at"),
        updated_at=_parse_stored_timestamp(value["updated_at"], f"{name}.updated_at"),
    )


def parse_superseded_payload(
    event: MemoryEvent,
    *,
    path: str | Path,
) -> SupersessionReplay:
    """Detach and validate every persisted supersession payload value."""
    try:
        if not isinstance(event, MemoryEvent):
            raise ValidationError("event must be a MemoryEvent")
        if event.event_type is not MemoryEventType.SUPERSEDED:
            raise ValidationError("event type must be superseded")
        payload = _require_mapping(event.payload, "payload")
        _require_exact_keys(payload, _PAYLOAD_KEYS, "payload")
        source = _hydrate_record(payload["source"], "source")
        replacement = _hydrate_record(payload["replacement"], "replacement")
        boundary = _validate_supersession_records(source, replacement)
        if source.updated_at != event.occurred_at or replacement.updated_at != event.occurred_at:
            raise ValidationError("supersession snapshot updated_at does not match event time")
        effective_at = _parse_stored_timestamp(payload["effective_at"], "effective_at")
        if effective_at != boundary:
            raise ValidationError("effective boundary does not match snapshots")
        if source.scope != event.scope or replacement.scope != event.scope:
            raise ValidationError("snapshot scope does not match event scope")
        if payload["scope_key"] != event.scope.scope_key:
            raise ValidationError("payload scope_key does not match event scope")
        if payload["source_memory_id"] != source.id or event.memory_id != source.id:
            raise ValidationError("source_memory_id does not match source snapshot")
        if payload["replacement_memory_id"] != replacement.id:
            raise ValidationError("replacement_memory_id does not match replacement snapshot")

        reason = _normalize_reason(payload["reason"])
        if reason != payload["reason"]:
            raise ValidationError("reason must be normalized")

        relation = _require_mapping(payload["relation"], "relation")
        _require_exact_keys(
            relation,
            {"id", "relation_type", "source_id", "target_id"},
            "relation",
        )
        relation_id = _normalize_identifier(relation["id"], "relation.id")
        if relation["relation_type"] != _RELATION_TYPE:
            raise ValidationError("relation type must be supersedes")
        if relation["source_id"] != replacement.id or relation["target_id"] != source.id:
            raise ValidationError("relation endpoints do not match snapshots")

        evidence = _require_mapping(payload["evidence"], "evidence")
        _require_exact_keys(
            evidence,
            {"memory_id", "source_memory_id", "relation"},
            "evidence",
        )
        if (
            evidence["memory_id"] != replacement.id
            or evidence["source_memory_id"] != source.id
            or evidence["relation"] != _RELATION_TYPE
        ):
            raise ValidationError("evidence endpoints do not match snapshots")

        return SupersessionReplay(
            source=source,
            replacement=replacement,
            effective_at=effective_at,
            reason=reason,
            relation_id=relation_id,
            relation_type=_RELATION_TYPE,
            evidence_memory_id=replacement.id,
            evidence_source_memory_id=source.id,
        )
    except Exception as error:
        if isinstance(error, StorageError):
            raise
        raise StorageError(
            f"Cannot replay invalid superseded event {getattr(event, 'id', None)!r} from {path}: {error}"
        ) from error
