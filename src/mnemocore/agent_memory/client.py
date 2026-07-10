"""Public asynchronous and explicit synchronous agent-memory clients."""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar
from uuid import uuid4

from .errors import AgentMemoryError, ClosedStoreError
from .models import (
    MemoryHistoryEntry,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    RecallResult,
)
from .sqlite_store import SQLiteMemoryStore
from .store import MemoryStore

_T = TypeVar("_T")


class AgentMemory:
    """Scope-bound asynchronous facade over a memory store."""

    def __init__(self, store: MemoryStore, scope: MemoryScope) -> None:
        self._store = store
        self._scope = scope
        self._closed = False

    @classmethod
    async def open(cls, path: str | Path, *, scope: MemoryScope) -> AgentMemory:
        """Open a persistent SQLite-backed memory client for one exact scope."""
        store = await SQLiteMemoryStore.open(path)
        return cls(store, scope)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedStoreError("Agent memory store is closed")

    async def remember(
        self,
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
        self._ensure_open()
        return await self._store.remember(
            self._scope,
            content,
            kind=kind,
            metadata=metadata,
            idempotency_key=idempotency_key,
            confidence=confidence,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
        )

    async def recall(
        self,
        query: str,
        *,
        kinds: Sequence[MemoryKind] = (),
        limit: int = 10,
        as_of: Optional[str] = None,
        use_hdv_rerank: bool = False,
        embedder: Optional[Callable[[str], Any]] = None,
        include_ancestors: bool = False,
    ) -> builtins.list[RecallResult]:
        self._ensure_open()
        return await self._store.recall(
            self._scope,
            query,
            kinds=kinds,
            limit=limit,
            as_of=as_of,
            use_hdv_rerank=use_hdv_rerank,
            embedder=embedder,
            include_ancestors=include_ancestors,
        )

    async def get(
        self,
        memory_id: str,
        *,
        include_forgotten: bool = False,
    ) -> MemoryRecord:
        self._ensure_open()
        return await self._store.get(
            self._scope,
            memory_id,
            include_forgotten=include_forgotten,
        )

    async def list(
        self,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
    ) -> builtins.list[MemoryRecord]:
        self._ensure_open()
        return await self._store.list(
            self._scope,
            kind=kind,
            status=status,
            limit=limit,
        )

    async def history(self, memory_id: str) -> builtins.list[MemoryHistoryEntry]:
        self._ensure_open()
        return await self._store.history(self._scope, memory_id)

    async def forget(
        self,
        memory_id: str,
        *,
        reason: Optional[str] = None,
    ) -> MemoryRecord:
        self._ensure_open()
        return await self._store.forget(self._scope, memory_id, reason=reason)

    async def rebuild(self) -> int:
        """Repair this exact scope's projections from its immutable ledger."""
        self._ensure_open()
        return await self._store.rebuild(self._scope)

    async def start_session(
        self,
        goal: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "MemorySession":
        """Start a child session scope for grouped episodic work.

        Creates (or accepts) a session_id and returns a lightweight MemorySession
        whose operations are automatically scoped to base_scope + session_id.
        """
        self._ensure_open()
        if session_id is None:
            session_id = uuid4().hex
        child_scope = MemoryScope(
            tenant_id=self._scope.tenant_id,
            user_id=self._scope.user_id,
            agent_id=self._scope.agent_id,
            project_id=self._scope.project_id,
            session_id=session_id,
        )
        return MemorySession(self._store, child_scope, goal=goal)

    async def close(self) -> None:
        if self._closed:
            return
        await self._store.close()
        self._closed = True

    async def __aenter__(self) -> AgentMemory:
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()


class SyncAgentMemory:
    """Explicit synchronous facade backed by one private event loop."""

    def __init__(self, client: AgentMemory, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop
        self._closed = False

    @staticmethod
    def _reject_running_loop() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise AgentMemoryError("Use AgentMemory inside async code")

    @classmethod
    def open(cls, path: str | Path, *, scope: MemoryScope) -> SyncAgentMemory:
        """Open a scope-bound sync client outside asynchronous code."""
        cls._reject_running_loop()
        loop = asyncio.new_event_loop()
        try:
            client = loop.run_until_complete(AgentMemory.open(path, scope=scope))
        except BaseException:
            loop.close()
            raise
        return cls(client, loop)

    def _run(self, operation: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
        self._reject_running_loop()
        if self._closed:
            raise ClosedStoreError("Agent memory store is closed")
        return self._loop.run_until_complete(operation())

    def remember(
        self,
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
        return self._run(
            lambda: self._client.remember(
                content,
                kind=kind,
                metadata=metadata,
                idempotency_key=idempotency_key,
                confidence=confidence,
                observed_at=observed_at,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        )

    def recall(
        self,
        query: str,
        *,
        kinds: Sequence[MemoryKind] = (),
        limit: int = 10,
        as_of: Optional[str] = None,
        use_hdv_rerank: bool = False,
        embedder: Optional[Callable[[str], Any]] = None,
        include_ancestors: bool = False,
    ) -> builtins.list[RecallResult]:
        return self._run(
            lambda: self._client.recall(
                query,
                kinds=kinds,
                limit=limit,
                as_of=as_of,
                use_hdv_rerank=use_hdv_rerank,
                embedder=embedder,
                include_ancestors=include_ancestors,
            )
        )

    def get(
        self,
        memory_id: str,
        *,
        include_forgotten: bool = False,
    ) -> MemoryRecord:
        return self._run(
            lambda: self._client.get(
                memory_id,
                include_forgotten=include_forgotten,
            )
        )

    def list(
        self,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
    ) -> builtins.list[MemoryRecord]:
        return self._run(
            lambda: self._client.list(kind=kind, status=status, limit=limit)
        )

    def history(self, memory_id: str) -> builtins.list[MemoryHistoryEntry]:
        return self._run(lambda: self._client.history(memory_id))

    def forget(
        self,
        memory_id: str,
        *,
        reason: Optional[str] = None,
    ) -> MemoryRecord:
        return self._run(lambda: self._client.forget(memory_id, reason=reason))

    def rebuild(self) -> int:
        return self._run(self._client.rebuild)

    def start_session(
        self,
        goal: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SyncMemorySession:
        """Start session (sync version): returns SyncMemorySession that delegates via the loop."""
        async def _start() -> MemorySession:
            return await self._client.start_session(goal=goal, session_id=session_id)
        async_sess = self._run(_start)
        return SyncMemorySession(async_sess, self._loop)

    def close(self) -> None:
        self._reject_running_loop()
        if self._closed:
            return
        try:
            self._loop.run_until_complete(self._client.close())
        finally:
            self._closed = True
            self._loop.close()

    def __enter__(self) -> SyncAgentMemory:
        self._reject_running_loop()
        if self._closed:
            raise ClosedStoreError("Agent memory store is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class MemorySession:
    """Lightweight session-scoped memory handle.

    All remember/recall/etc. are automatically bound to a child scope
    (base + session_id). finish() records an EPISODE summary in the session scope.
    """

    def __init__(
        self,
        store: MemoryStore,
        scope: MemoryScope,
        goal: Optional[str] = None,
    ) -> None:
        self._store = store
        self._scope = scope
        self.goal = goal
        self.session_id = scope.session_id
        self._finished = False

    async def remember(
        self,
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
        return await self._store.remember(
            self._scope,
            content,
            kind=kind,
            metadata=metadata,
            idempotency_key=idempotency_key,
            confidence=confidence,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
        )

    async def recall(
        self,
        query: str,
        *,
        kinds: Sequence[MemoryKind] = (),
        limit: int = 10,
        as_of: Optional[str] = None,
        use_hdv_rerank: bool = False,
        embedder: Optional[Callable[[str], Any]] = None,
        include_ancestors: bool = True,  # sessions benefit from project/agent level knowledge by default
    ) -> builtins.list[RecallResult]:
        return await self._store.recall(
            self._scope,
            query,
            kinds=kinds,
            limit=limit,
            as_of=as_of,
            use_hdv_rerank=use_hdv_rerank,
            embedder=embedder,
            include_ancestors=include_ancestors,
        )

    async def get(
        self,
        memory_id: str,
        *,
        include_forgotten: bool = False,
    ) -> MemoryRecord:
        return await self._store.get(
            self._scope,
            memory_id,
            include_forgotten=include_forgotten,
        )

    async def list(
        self,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
    ) -> builtins.list[MemoryRecord]:
        return await self._store.list(
            self._scope,
            kind=kind,
            status=status,
            limit=limit,
        )

    async def history(self, memory_id: str) -> builtins.list[MemoryHistoryEntry]:
        return await self._store.history(self._scope, memory_id)

    async def forget(
        self,
        memory_id: str,
        *,
        reason: Optional[str] = None,
    ) -> MemoryRecord:
        return await self._store.forget(self._scope, memory_id, reason=reason)

    async def observe(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.OBSERVATION,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> MemoryRecord:
        """Convenience alias for remember(..., kind=OBSERVATION) or with explicit kind."""
        return await self.remember(
            content, kind=kind, metadata=metadata, **kwargs
        )

    async def finish(
        self,
        outcome: str = "success",
        reward: float = 0.0,
        notes: Optional[str] = None,
    ) -> MemoryRecord:
        """Close the session by recording an EPISODE memory with outcome/reward.

        The EPISODE is stored in the *session* scope.
        """
        if self._finished:
            # Idempotent-ish: still allow re-finish but don't duplicate?
            # For minimal, just record again or return last? Here we proceed.
            pass
        meta: dict[str, Any] = {
            "goal": self.goal,
            "outcome": outcome,
            "reward": reward,
        }
        if notes is not None:
            meta["notes"] = notes
        content = f"Session goal={self.goal!r} outcome={outcome!r} reward={reward}"
        rec = await self.remember(
            content,
            kind=MemoryKind.EPISODE,
            metadata=meta,
        )
        self._finished = True
        return rec


class SyncMemorySession:
    """Explicit synchronous wrapper around a MemorySession (reuses parent's loop)."""

    def __init__(
        self, session: MemorySession, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._session = session
        self._loop = loop
        self.goal = session.goal
        self.session_id = session.session_id

    def _run(self, operation: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
        SyncAgentMemory._reject_running_loop()
        return self._loop.run_until_complete(operation())

    def remember(
        self,
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
        return self._run(
            lambda: self._session.remember(
                content,
                kind=kind,
                metadata=metadata,
                idempotency_key=idempotency_key,
                confidence=confidence,
                observed_at=observed_at,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        )

    def recall(
        self,
        query: str,
        *,
        kinds: Sequence[MemoryKind] = (),
        limit: int = 10,
        as_of: Optional[str] = None,
        use_hdv_rerank: bool = False,
        embedder: Optional[Callable[[str], Any]] = None,
        include_ancestors: bool = True,
    ) -> builtins.list[RecallResult]:
        return self._run(
            lambda: self._session.recall(
                query,
                kinds=kinds,
                limit=limit,
                as_of=as_of,
                use_hdv_rerank=use_hdv_rerank,
                embedder=embedder,
                include_ancestors=include_ancestors,
            )
        )

    def get(
        self,
        memory_id: str,
        *,
        include_forgotten: bool = False,
    ) -> MemoryRecord:
        return self._run(
            lambda: self._session.get(
                memory_id,
                include_forgotten=include_forgotten,
            )
        )

    def list(
        self,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
    ) -> builtins.list[MemoryRecord]:
        return self._run(
            lambda: self._session.list(kind=kind, status=status, limit=limit)
        )

    def history(self, memory_id: str) -> builtins.list[MemoryHistoryEntry]:
        return self._run(lambda: self._session.history(memory_id))

    def forget(
        self,
        memory_id: str,
        *,
        reason: Optional[str] = None,
    ) -> MemoryRecord:
        return self._run(lambda: self._session.forget(memory_id, reason=reason))

    def observe(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.OBSERVATION,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> MemoryRecord:
        return self._run(
            lambda: self._session.observe(
                content, kind=kind, metadata=metadata, **kwargs
            )
        )

    def finish(
        self,
        outcome: str = "success",
        reward: float = 0.0,
        notes: Optional[str] = None,
    ) -> MemoryRecord:
        return self._run(
            lambda: self._session.finish(outcome=outcome, reward=reward, notes=notes)
        )
