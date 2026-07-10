import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mnemocore.agent_memory import (
    AgentMemory,
    AgentMemoryError,
    ClosedStoreError,
    MemoryEventType,
    MemoryKind,
    MemoryNotFoundError,
    MemoryScope,
    MemorySession,
    MemoryStatus,
    SyncAgentMemory,
    SyncMemorySession,
)


@pytest.mark.asyncio
async def test_async_public_round_trip(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore")
    memory = await AgentMemory.open(tmp_path / "memory.db", scope=scope)

    async with memory:
        stored = await memory.remember(
            "Prefer minimal public APIs",
            kind=MemoryKind.PREFERENCE,
            metadata={"source": "test"},
            idempotency_key="preference-1",
            confidence=0.9,
            observed_at="2026-07-10T10:00:00Z",
            valid_from="2026-07-10T09:00:00Z",
            valid_to="2026-07-11T09:00:00Z",
        )
        assert await memory.rebuild() == 1
        recalled = await memory.recall(
            "minimal APIs",
            kinds=(MemoryKind.PREFERENCE,),
            limit=1,
            as_of="2026-07-10T12:00:00Z",
        )

        assert recalled[0].memory.id == stored.id
        assert await memory.get(stored.id) == stored
        assert await memory.list(kind=MemoryKind.PREFERENCE, limit=1) == [stored]
        assert [entry.action for entry in await memory.history(stored.id)] == [
            MemoryEventType.REMEMBERED
        ]

        forgotten = await memory.forget(stored.id, reason="obsolete")
        assert forgotten.status is MemoryStatus.FORGOTTEN
        assert await memory.get(stored.id, include_forgotten=True) == forgotten

    with pytest.raises(ClosedStoreError):
        await memory.list()


@pytest.mark.asyncio
async def test_async_client_always_uses_its_exact_default_scope(tmp_path):
    path = tmp_path / "memory.db"
    local_scope = MemoryScope(user_id="robin", agent_id="codex")
    foreign_scope = MemoryScope(user_id="other", agent_id="codex")

    async with await AgentMemory.open(path, scope=local_scope) as local:
        local_record = await local.remember("Local scoped memory")
    async with await AgentMemory.open(path, scope=foreign_scope) as foreign:
        foreign_record = await foreign.remember("Foreign scoped memory")
        with pytest.raises(MemoryNotFoundError):
            await foreign.get(local_record.id)
        assert await foreign.list() == [foreign_record]


@pytest.mark.asyncio
async def test_async_close_is_idempotent_and_all_operations_reject_after_close(tmp_path):
    memory = await AgentMemory.open(
        tmp_path / "memory.db",
        scope=MemoryScope(user_id="robin", agent_id="codex"),
    )
    await memory.close()
    await memory.close()

    operations = (
        lambda: memory.remember("closed"),
        lambda: memory.recall("closed"),
        lambda: memory.get("missing"),
        lambda: memory.list(),
        lambda: memory.history("missing"),
        lambda: memory.forget("missing"),
        lambda: memory.rebuild(),
    )
    for operation in operations:
        with pytest.raises(ClosedStoreError):
            await operation()


def test_sync_public_round_trip_reuses_one_private_loop(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex")

    with SyncAgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        stored = memory.remember(
            "Sync wrapper stays explicit",
            kind=MemoryKind.PROCEDURE,
            idempotency_key="sync-1",
        )
        assert memory.rebuild() == 1
        assert memory.get(stored.id).content == stored.content
        assert memory.list(kind=MemoryKind.PROCEDURE) == [stored]
        assert memory.recall("wrapper explicit")[0].memory.id == stored.id
        assert [entry.action for entry in memory.history(stored.id)] == [
            MemoryEventType.REMEMBERED
        ]
        assert memory.forget(stored.id, reason="done").status is MemoryStatus.FORGOTTEN
        assert memory.get(stored.id, include_forgotten=True).id == stored.id

    with pytest.raises(ClosedStoreError):
        memory.list()
    memory.close()


def test_sync_client_refuses_calls_from_async_code_without_becoming_unusable(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex")
    memory = SyncAgentMemory.open(tmp_path / "memory.db", scope=scope)
    stored = memory.remember("Remain usable after rejected async call")

    async def call_sync_client():
        with pytest.raises(AgentMemoryError, match="^Use AgentMemory inside async code$"):
            memory.get(stored.id)

    asyncio.run(call_sync_client())
    assert memory.get(stored.id) == stored
    memory.close()


def test_sync_open_refuses_running_event_loop(tmp_path):
    async def open_sync_client():
        with pytest.raises(AgentMemoryError, match="^Use AgentMemory inside async code$"):
            SyncAgentMemory.open(
                tmp_path / "memory.db",
                scope=MemoryScope(user_id="robin", agent_id="codex"),
            )

    asyncio.run(open_sync_client())


def test_public_import_keeps_legacy_and_optional_dependencies_unloaded():
    script = """
import sys

blocked = ("mnemocore.core.engine", "qdrant_client", "redis", "fastapi", "faiss")
assert not any(name in sys.modules for name in blocked)
import mnemocore.agent_memory
assert not any(name in sys.modules for name in blocked)
"""

    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH")) if part
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


def test_lazy_root_memory_export_remains_compatible():
    script = """
from mnemocore import Memory

memory = Memory()
assert repr(memory) == "<Memory (lite)>"
"""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH")) if part
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


def test_subprocess_memory_persists_across_restarts(tmp_path):
    path = tmp_path / "memory.db"
    script = r"""
import asyncio
import sys

from mnemocore.agent_memory import AgentMemory, MemoryScope


async def main():
    path = sys.argv[1]
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore")
    async with await AgentMemory.open(path, scope=scope) as memory:
        await memory.remember("Persistent across restarts")
        await memory.remember("A second durable memory")
    async with await AgentMemory.open(path, scope=scope) as memory:
        results = await memory.recall("persistent restarts", limit=1)
        print(results[0].memory.content)


asyncio.run(main())
"""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = source_root

    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout == "Persistent across restarts\n"
    assert completed.stderr == ""


@pytest.mark.asyncio
async def test_compile_context_is_bounded_and_receipted(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="core")

    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        await memory.remember(
            "Prefer concise retrieval updates", kind=MemoryKind.PREFERENCE
        )
        await memory.remember(
            "Use FTS retrieval before reranking", kind=MemoryKind.PROCEDURE
        )

        pack = await memory.compile_context("retrieval", token_budget=20)

    assert pack.estimated_tokens <= 20
    assert pack.core[0].receipt.kind is MemoryKind.PREFERENCE
    assert pack.procedural[0].receipt.evidence_ids


@pytest.mark.asyncio
async def test_session_start_finish_roundtrip_and_session_scoped_recall(tmp_path):
    """Roundtrip for the Session API: start, remember (DECISION), recall, finish(EPISODE)."""
    base_scope = MemoryScope(
        user_id="robin", agent_id="codex", project_id="mnemocore"
    )
    path = tmp_path / "memory.db"
    memory = await AgentMemory.open(path, scope=base_scope)

    async with memory:
        session = await memory.start_session(goal="Fix bug in retrieval fusion")
        assert isinstance(session, MemorySession)
        assert session.goal == "Fix bug in retrieval fusion"
        assert session.session_id is not None
        assert session.session_id == session._scope.session_id  # internal for test

        # remember in session with DECISION kind
        rec = await session.remember(
            "Never use pure HDV for semantic; always union with FTS first",
            kind=MemoryKind.DECISION,
        )
        assert rec.kind == MemoryKind.DECISION
        assert rec.scope.session_id == session.session_id
        assert rec.scope.project_id == base_scope.project_id  # inherits base

        # recall within session sees the session-scoped item
        # The default provider is intentionally lexical; use terms present in
        # the remembered item so this test isolates session-scope behavior.
        ctx = await session.recall("semantic FTS", limit=5)
        assert len(ctx) >= 1
        assert ctx[0].memory.scope.session_id == session.session_id
        assert "FTS first" in ctx[0].memory.content

        # also test observe alias
        obs = await session.observe("Observed mid-session work")
        assert obs.kind == MemoryKind.OBSERVATION
        assert obs.scope.session_id == session.session_id

        # finish writes EPISODE in session scope
        episode = await session.finish(
            outcome="success", reward=0.8, notes="demo roundtrip"
        )
        assert episode.kind == MemoryKind.EPISODE
        assert episode.scope.session_id == session.session_id
        assert episode.metadata["outcome"] == "success"
        assert episode.metadata["reward"] == 0.8
        assert episode.metadata["goal"] == "Fix bug in retrieval fusion"

        # post-finish recall in session still works and includes episode
        after = await session.recall("success", kinds=[MemoryKind.EPISODE], limit=1)
        assert len(after) == 1
        assert after[0].memory.id == episode.id

    # direct remember on base memory still works (backward compat)
    async with await AgentMemory.open(path, scope=base_scope) as mem2:
        base_rec = await mem2.remember("base level fact", kind=MemoryKind.FACT)
        assert base_rec.scope.session_id is None
        # session data not visible in base (no cross)
        base_results = await mem2.recall("retrieval", limit=3)
        assert all(r.memory.scope.session_id is None for r in base_results)


@pytest.mark.asyncio
async def test_session_finish_is_idempotent(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex")
    async with await AgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        session = await memory.start_session(goal="Complete a task")

        first = await session.finish(outcome="success", reward=1.0)
        second = await session.finish(outcome="success", reward=1.0)

        assert second == first
        assert await session.list(kind=MemoryKind.EPISODE) == [first]


def test_sync_session_start_finish(tmp_path):
    """Sync counterpart for MemorySession via SyncAgentMemory."""
    base_scope = MemoryScope(user_id="robin", agent_id="codex")
    with SyncAgentMemory.open(tmp_path / "syncmem.db", scope=base_scope) as mem:
        sess = mem.start_session(goal="sync goal", session_id="fixed-sync-123")
        assert isinstance(sess, SyncMemorySession)
        assert sess.session_id == "fixed-sync-123"
        assert sess.goal == "sync goal"

        rec = sess.remember("Sync decision in session", kind=MemoryKind.DECISION)
        assert rec.scope.session_id == "fixed-sync-123"

        results = sess.recall("decision", limit=1)
        assert results[0].memory.scope.session_id == "fixed-sync-123"

        brief = sess.compile_context("decision", token_budget=20)
        assert brief.working[0].receipt.memory_id == rec.id

        ep = sess.finish(outcome="partial", reward=0.5)
        assert ep.kind == MemoryKind.EPISODE
        assert ep.metadata["reward"] == 0.5

        # observe also
        o = sess.observe("sync observe")
        assert o.scope.session_id == "fixed-sync-123"
