"""Typed public models for MnemoCore's agent-memory API."""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional

from .errors import ScopeError, ValidationError

MAX_IDENTIFIER_LENGTH = 256
MAX_CONTENT_LENGTH = 100_000


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _normalize_identifier(value: Optional[str], name: str, *, required: bool) -> Optional[str]:
    if value is None:
        if required:
            raise ScopeError(f"{name} is required")
        return None
    if not isinstance(value, str):
        raise ScopeError(f"{name} must be a string")

    normalized = value.strip()
    if not normalized:
        if required:
            raise ScopeError(f"{name} must not be blank")
        return None
    if len(normalized) > MAX_IDENTIFIER_LENGTH:
        raise ScopeError(f"{name} must be at most {MAX_IDENTIFIER_LENGTH} characters")
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ScopeError(f"{name} must not contain control characters")
    return normalized


def _normalize_datetime(value: Any, name: str, *, required: bool = True) -> Optional[datetime]:
    if value is None:
        if required:
            raise ValidationError(f"{name} is required")
        return None
    if not isinstance(value, datetime):
        raise ValidationError(f"{name} must be a datetime")
    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be timezone-aware") from error
    if value.tzinfo is None or offset is None:
        raise ValidationError(f"{name} must be timezone-aware")
    try:
        return value.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be a valid timezone-aware datetime") from error


def _freeze_json_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{name} keys must be strings")
            frozen[key] = _freeze_json_value(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, name) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{name} must not contain non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValidationError(f"{name} must contain only JSON-compatible values")


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


class MemoryKind(str, Enum):
    OBSERVATION = "observation"
    FACT = "fact"
    EPISODE = "episode"
    PROCEDURE = "procedure"
    PREFERENCE = "preference"
    SUMMARY = "summary"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    FORGOTTEN = "forgotten"


class MemoryEventType(str, Enum):
    REMEMBERED = "remembered"
    REINFORCED = "reinforced"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    FORGOTTEN = "forgotten"
    RESTORED = "restored"


@dataclass(frozen=True, slots=True)
class MemoryScope:
    user_id: str
    agent_id: str
    tenant_id: str = "local"
    project_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name, required in (
            ("tenant_id", True),
            ("user_id", True),
            ("agent_id", True),
            ("project_id", False),
            ("session_id", False),
        ):
            value = _normalize_identifier(getattr(self, name), name, required=required)
            object.__setattr__(self, name, value)

    @property
    def scope_key(self) -> str:
        """Return an unambiguous stable key for this exact scope."""
        return json.dumps(
            [
                self.tenant_id,
                self.user_id,
                self.agent_id,
                self.project_id,
                self.session_id,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    scope: MemoryScope
    kind: MemoryKind
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.ACTIVE
    confidence: float = 1.0
    observed_at: datetime = field(default_factory=utc_now)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValidationError("content must not be blank")
        if len(self.content) > MAX_CONTENT_LENGTH:
            raise ValidationError(f"content must be at most {MAX_CONTENT_LENGTH} characters")
        confidence_is_valid = (
            _is_finite_number(self.confidence) and 0 <= self.confidence <= 1
        )
        if not confidence_is_valid:
            raise ValidationError("confidence must be between 0 and 1")
        if not isinstance(self.metadata, Mapping):
            raise ValidationError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_json_value(self.metadata, "metadata"))
        for name, required in (
            ("observed_at", True),
            ("valid_from", False),
            ("valid_to", False),
            ("created_at", True),
            ("updated_at", True),
        ):
            value = _normalize_datetime(getattr(self, name), name, required=required)
            object.__setattr__(self, name, value)
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to <= self.valid_from
        ):
            raise ValidationError("valid_to must be after valid_from")


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    id: str
    scope: MemoryScope
    event_type: MemoryEventType
    payload: Mapping[str, Any]
    occurred_at: datetime = field(default_factory=utc_now)
    created_at: datetime = field(default_factory=utc_now)
    memory_id: Optional[str] = None
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise ValidationError("payload must be a mapping")
        object.__setattr__(self, "payload", _freeze_json_value(self.payload, "payload"))
        for name in ("occurred_at", "created_at"):
            value = _normalize_datetime(getattr(self, name), name)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class MemoryHistoryEntry:
    id: str
    memory_id: str
    event_id: str
    action: MemoryEventType
    status: MemoryStatus
    created_at: datetime = field(default_factory=utc_now)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.details, Mapping):
            raise ValidationError("details must be a mapping")
        object.__setattr__(self, "details", _freeze_json_value(self.details, "details"))
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, "created_at"))


@dataclass(frozen=True, slots=True)
class MemoryRelation:
    id: str
    scope: MemoryScope
    source_id: str
    target_id: str
    relation_type: str
    valid_from: datetime
    valid_to: Optional[datetime]
    confidence: float
    event_id: str
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("id", "source_id", "target_id", "relation_type", "event_id"):
            try:
                value = _normalize_identifier(getattr(self, name), name, required=True)
            except ScopeError as error:
                raise ValidationError(str(error)) from error
            object.__setattr__(self, name, value)
        if not isinstance(self.scope, MemoryScope):
            raise ValidationError("scope must be a MemoryScope")
        confidence_is_valid = (
            _is_finite_number(self.confidence) and 0 <= self.confidence <= 1
        )
        if not confidence_is_valid:
            raise ValidationError("confidence must be between 0 and 1")
        for name, required in (
            ("valid_from", True),
            ("valid_to", False),
            ("created_at", True),
        ):
            value = _normalize_datetime(getattr(self, name), name, required=required)
            object.__setattr__(self, name, value)
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValidationError("valid_to must be after valid_from")


@dataclass(frozen=True, slots=True)
class MemoryReceipt:
    memory: MemoryRecord
    evidence_event_ids: tuple[str, ...] = ()
    evidence_memory_ids: tuple[str, ...] = ()
    relations: tuple[MemoryRelation, ...] = ()
    history: tuple[MemoryHistoryEntry, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.memory, MemoryRecord):
            raise ValidationError("memory must be a MemoryRecord")
        for name in ("evidence_event_ids", "evidence_memory_ids"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise ValidationError(f"{name} must be a sequence of strings")
            try:
                normalized = tuple(
                    _normalize_identifier(item, name, required=True) for item in value
                )
            except (TypeError, ScopeError) as error:
                raise ValidationError(f"{name} must contain nonblank strings") from error
            object.__setattr__(self, name, normalized)
        for name, expected_type in (
            ("relations", MemoryRelation),
            ("history", MemoryHistoryEntry),
        ):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise ValidationError(f"{name} must be a sequence")
            try:
                normalized = tuple(value)
            except TypeError as error:
                raise ValidationError(f"{name} must be a sequence") from error
            if any(not isinstance(item, expected_type) for item in normalized):
                raise ValidationError(f"{name} contains an invalid item")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValidationError("explanation must not be blank")


@dataclass(frozen=True, slots=True)
class RecallResult:
    memory: MemoryRecord
    score: float
    score_components: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _is_finite_number(self.score):
            raise ValidationError("score must be finite")
        if not isinstance(self.score_components, Mapping):
            raise ValidationError("score_components must be a mapping")
        object.__setattr__(
            self,
            "score_components",
            _freeze_json_value(self.score_components, "score_components"),
        )
        if isinstance(self.evidence_ids, (str, bytes)):
            raise ValidationError("evidence_ids must be a sequence of strings")
        try:
            evidence_ids = tuple(self.evidence_ids)
        except TypeError as error:
            raise ValidationError("evidence_ids must be a sequence of strings") from error
        if any(not isinstance(event_id, str) or not event_id for event_id in evidence_ids):
            raise ValidationError("evidence_ids must contain non-empty strings")
        object.__setattr__(self, "evidence_ids", evidence_ids)
