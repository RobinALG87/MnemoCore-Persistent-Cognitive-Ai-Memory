"""Storage protocol for MnemoCore's agent-memory API."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Protocol, runtime_checkable

from .models import (
    MemoryHistoryEntry,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    RecallResult,
)


@runtime_checkable
class MemoryStore(Protocol):
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
        ...

    async def get(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        include_forgotten: bool = False,
    ) -> MemoryRecord:
        ...

    async def list(
        self,
        scope: MemoryScope,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
    ) -> builtins.list[MemoryRecord]:
        ...

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
    ) -> builtins.list[RecallResult]:
        ...

    async def history(
        self,
        scope: MemoryScope,
        memory_id: str,
    ) -> builtins.list[MemoryHistoryEntry]:
        ...

    async def forget(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        reason: Optional[str] = None,
    ) -> MemoryRecord:
        ...

    async def rebuild(self, scope: MemoryScope) -> int:
        ...

    async def close(self) -> None:
        ...
