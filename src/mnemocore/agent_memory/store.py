"""Storage protocol for MnemoCore's agent-memory API."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from .models import (
    MemoryHistoryEntry,
    MemoryKind,
    MemoryRecord,
    MemoryReceipt,
    MemoryScope,
    MemoryStatus,
    RecallResult,
)


@dataclass(frozen=True, slots=True)
class MemoryWrite:
    """One remember-only write accepted by an atomic store batch."""

    content: str
    kind: MemoryKind
    confidence: float
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must not be blank")
        if not isinstance(self.kind, MemoryKind):
            raise TypeError("kind must be a MemoryKind")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be between 0 and 1")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@runtime_checkable
class MemoryStore(Protocol):
    async def remember_many(
        self, scope: MemoryScope, writes: Sequence[MemoryWrite]
    ) -> builtins.list[MemoryRecord]: ...

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
    ) -> MemoryRecord: ...

    async def get(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        include_forgotten: bool = False,
    ) -> MemoryRecord: ...

    async def list(
        self,
        scope: MemoryScope,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[MemoryRecord]: ...

    async def recall(
        self,
        scope: MemoryScope,
        query: str,
        *,
        kinds: Sequence[MemoryKind] = (),
        limit: int = 10,
        as_of: Optional[str] = None,
        use_hdv_rerank: bool = False,
        embedder: Optional[Callable[[str], Any]] = None,
        include_ancestors: bool = False,
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
    ) -> builtins.list[RecallResult]: ...

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
    ) -> MemoryRecord: ...

    async def explain(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
    ) -> MemoryReceipt: ...

    async def history(
        self,
        scope: MemoryScope,
        memory_id: str,
    ) -> builtins.list[MemoryHistoryEntry]: ...

    async def forget(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        reason: Optional[str] = None,
    ) -> MemoryRecord: ...

    async def rebuild(self, scope: MemoryScope) -> int: ...

    async def close(self) -> None: ...
