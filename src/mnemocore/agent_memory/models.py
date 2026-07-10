"""Typed public models for MnemoCore's agent-memory API."""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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
    metadata: dict[str, Any] = field(default_factory=dict)
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
        try:
            confidence_is_valid = math.isfinite(self.confidence) and 0 <= self.confidence <= 1
        except TypeError:
            confidence_is_valid = False
        if not confidence_is_valid:
            raise ValidationError("confidence must be between 0 and 1")
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
    payload: dict[str, Any]
    occurred_at: datetime = field(default_factory=utc_now)
    created_at: datetime = field(default_factory=utc_now)
    memory_id: Optional[str] = None
    idempotency_key: Optional[str] = None


@dataclass(frozen=True, slots=True)
class MemoryHistoryEntry:
    id: str
    memory_id: str
    event_id: str
    action: MemoryEventType
    status: MemoryStatus
    created_at: datetime = field(default_factory=utc_now)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecallResult:
    memory: MemoryRecord
    score: float
    score_components: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    evidence_ids: tuple[str, ...] = ()
