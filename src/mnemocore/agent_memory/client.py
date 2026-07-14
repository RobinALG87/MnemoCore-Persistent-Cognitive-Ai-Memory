"""Public asynchronous and explicit synchronous agent-memory clients."""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar
from uuid import uuid4

from .context import CONTEXT_LEVELS, compile_context_pack
from .errors import AgentMemoryError, ClosedStoreError, StorageError
from .models import (
    ContextPack,
    MemoryHistoryEntry,
    MemoryKind,
    MemoryRecord,
    MemoryReceipt,
    MemoryScope,
    MemoryStatus,
    RecallResult,
)
from .sqlite_store import SQLiteMemoryStore
from .store import MemoryStore, MemoryWrite

_T = TypeVar("_T")


async def _compile_context(
    store: MemoryStore,
    scope: MemoryScope,
    query: str,
    *,
    token_budget: int,
    include_ancestors: bool,
) -> ContextPack:
    results_by_level = {}
    for level, kinds in CONTEXT_LEVELS:
        results_by_level[level] = await store.recall(
            scope,
            query,
            kinds=kinds,
            limit=10,
            include_ancestors=include_ancestors,
        )
    return compile_context_pack(
        query,
        token_budget=token_budget,
        results_by_level=results_by_level,
    )


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

    async def remember_many(
        self, writes: Sequence[MemoryWrite]
    ) -> builtins.list[MemoryRecord]:
        """Atomically persist a non-empty sequence of remember-only writes."""
        self._ensure_open()
        return await self._store.remember_many(self._scope, writes)

    async def remember_many_with_active_sources(
        self,
        writes: Sequence[MemoryWrite],
        *,
        source_memory_ids: Sequence[str],
    ) -> builtins.list[MemoryRecord]:
        """Atomically require active exact-scope sources before a batch write.

        Stores without this explicit capability are rejected rather than using
        a check-then-write fallback that could violate cognitive provenance.
        """
        self._ensure_open()
        operation = getattr(self._store, "remember_many_with_active_sources", None)
        if not callable(operation):
            raise StorageError(
                "memory store does not support atomic active-source batch writes"
            )
        return await operation(self._scope, writes, source_memory_ids=source_memory_ids)

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
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
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
            valid_at=valid_at,
            known_at=known_at,
        )

    async def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        effective_at: str,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        confidence: float = 1.0,
        idempotency_key: Optional[str] = None,
    ) -> MemoryRecord:
        self._ensure_open()
        return await self._store.supersede(
            self._scope,
            memory_id,
            content,
            effective_at=effective_at,
            reason=reason,
            metadata=metadata,
            confidence=confidence,
            idempotency_key=idempotency_key,
        )

    async def explain(
        self,
        memory_id: str,
        *,
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
    ) -> MemoryReceipt:
        self._ensure_open()
        return await self._store.explain(
            self._scope,
            memory_id,
            valid_at=valid_at,
            known_at=known_at,
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

    async def compile_context(
        self,
        query: str,
        *,
        token_budget: int = 1200,
        include_ancestors: bool = False,
    ) -> ContextPack:
        """Compile exact-scope recall into a bounded, receipt-bearing briefing."""
        self._ensure_open()
        return await _compile_context(
            self._store,
            self._scope,
            query,
            token_budget=token_budget,
            include_ancestors=include_ancestors,
        )

    async def list(
        self,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[MemoryRecord]:
        self._ensure_open()
        if type(offset) is int and offset == 0:
            return await self._store.list(
                self._scope,
                kind=kind,
                status=status,
                limit=limit,
            )
        return await self._store.list(
            self._scope,
            kind=kind,
            status=status,
            limit=limit,
            offset=offset,
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

    def remember_many(
        self, writes: Sequence[MemoryWrite]
    ) -> builtins.list[MemoryRecord]:
        """Synchronously persist a remember-only batch in one transaction."""
        return self._run(lambda: self._client.remember_many(writes))

    def remember_many_with_active_sources(
        self,
        writes: Sequence[MemoryWrite],
        *,
        source_memory_ids: Sequence[str],
    ) -> builtins.list[MemoryRecord]:
        """Synchronously run an atomic active-source batch write."""
        return self._run(
            lambda: self._client.remember_many_with_active_sources(
                writes, source_memory_ids=source_memory_ids
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
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
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
                valid_at=valid_at,
                known_at=known_at,
            )
        )

    def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        effective_at: str,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        confidence: float = 1.0,
        idempotency_key: Optional[str] = None,
    ) -> MemoryRecord:
        return self._run(
            lambda: self._client.supersede(
                memory_id,
                content,
                effective_at=effective_at,
                reason=reason,
                metadata=metadata,
                confidence=confidence,
                idempotency_key=idempotency_key,
            )
        )

    def explain(
        self,
        memory_id: str,
        *,
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
    ) -> MemoryReceipt:
        return self._run(
            lambda: self._client.explain(
                memory_id,
                valid_at=valid_at,
                known_at=known_at,
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

    def compile_context(
        self,
        query: str,
        *,
        token_budget: int = 1200,
        include_ancestors: bool = False,
    ) -> ContextPack:
        return self._run(
            lambda: self._client.compile_context(
                query,
                token_budget=token_budget,
                include_ancestors=include_ancestors,
            )
        )

    def list(
        self,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[MemoryRecord]:
        return self._run(
            lambda: self._client.list(
                kind=kind, status=status, limit=limit, offset=offset
            )
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
        self._final_episode: Optional[MemoryRecord] = None

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
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
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
            valid_at=valid_at,
            known_at=known_at,
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

    async def compile_context(
        self,
        query: str,
        *,
        token_budget: int = 1200,
        include_ancestors: bool = True,
    ) -> ContextPack:
        """Compile session and permitted ancestor memory into a mission briefing."""
        return await _compile_context(
            self._store,
            self._scope,
            query,
            token_budget=token_budget,
            include_ancestors=include_ancestors,
        )

    async def list(
        self,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[MemoryRecord]:
        if type(offset) is int and offset == 0:
            return await self._store.list(
                self._scope,
                kind=kind,
                status=status,
                limit=limit,
            )
        return await self._store.list(
            self._scope,
            kind=kind,
            status=status,
            limit=limit,
            offset=offset,
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
        return await self.remember(content, kind=kind, metadata=metadata, **kwargs)

    async def finish(
        self,
        outcome: str = "success",
        reward: float = 0.0,
        notes: Optional[str] = None,
    ) -> MemoryRecord:
        """Close the session by recording an EPISODE memory with outcome/reward.

        The EPISODE is stored in the *session* scope.
        """
        if self._final_episode is not None:
            return self._final_episode
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
        self._final_episode = rec
        self._finished = True
        return rec


class SyncMemorySession:
    """Explicit synchronous wrapper around a MemorySession (reuses parent's loop)."""

    def __init__(self, session: MemorySession, loop: asyncio.AbstractEventLoop) -> None:
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
        valid_at: Optional[str] = None,
        known_at: Optional[str] = None,
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
                valid_at=valid_at,
                known_at=known_at,
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

    def compile_context(
        self,
        query: str,
        *,
        token_budget: int = 1200,
        include_ancestors: bool = True,
    ) -> ContextPack:
        return self._run(
            lambda: self._session.compile_context(
                query,
                token_budget=token_budget,
                include_ancestors=include_ancestors,
            )
        )

    def list(
        self,
        *,
        kind: Optional[MemoryKind] = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[MemoryRecord]:
        return self._run(
            lambda: self._session.list(
                kind=kind, status=status, limit=limit, offset=offset
            )
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
