import asyncio
import inspect
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
    ValidationError,
)


VALID_BEFORE = "2026-07-11T09:30:00.000000Z"
EFFECTIVE_AT = "2026-07-11T09:30:00.000001Z"


def _known_at(record):
    return record.updated_at.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _without_self(method):
    signature = inspect.signature(method)
    return signature.replace(parameters=tuple(signature.parameters.values())[1:])


def _public_callable_names(client_type):
    return {
        name
        for name, value in inspect.getmembers(client_type)
        if not name.startswith("_") and callable(value)
    }


@pytest.mark.parametrize(
    ("async_type", "sync_type", "factory_exceptions"),
    [
        (AgentMemory, SyncAgentMemory, {"open", "start_session"}),
        (MemorySession, SyncMemorySession, set()),
    ],
)
def test_public_callable_name_sets_have_async_sync_parity(
    async_type,
    sync_type,
    factory_exceptions,
):
    async_names = _public_callable_names(async_type)
    sync_names = _public_callable_names(sync_type)

    assert async_names == sync_names
    assert factory_exceptions <= async_names


@pytest.mark.parametrize(
    ("async_name", "sync_name"),
    [("__aenter__", "__enter__"), ("__aexit__", "__exit__")],
)
def test_client_lifecycle_name_differences_are_explicit(async_name, sync_name):
    assert callable(getattr(AgentMemory, async_name))
    assert callable(getattr(SyncAgentMemory, sync_name))


@pytest.mark.parametrize(
    ("async_type", "sync_type", "method_name"),
    [
        *(
            (AgentMemory, SyncAgentMemory, method_name)
            for method_name in (
                "remember",
                "recall",
                "supersede",
                "explain",
                "get",
                "compile_context",
                "list",
                "history",
                "forget",
                "rebuild",
                "start_session",
                "close",
            )
        ),
        *(
            (MemorySession, SyncMemorySession, method_name)
            for method_name in (
                "remember",
                "recall",
                "get",
                "compile_context",
                "list",
                "history",
                "forget",
                "observe",
                "finish",
            )
        ),
    ],
)
def test_public_operational_async_sync_methods_have_parameter_parity(
    async_type,
    sync_type,
    method_name,
):
    async_signature = _without_self(getattr(async_type, method_name))
    sync_signature = _without_self(getattr(sync_type, method_name))

    assert tuple(async_signature.parameters.values()) == tuple(
        sync_signature.parameters.values()
    )


@pytest.mark.parametrize(
    ("async_type", "sync_type", "method_name"),
    [
        (AgentMemory, SyncAgentMemory, "open"),
        (AgentMemory, SyncAgentMemory, "start_session"),
    ],
)
def test_async_sync_factories_preserve_parameters_with_mapped_return_types(
    async_type,
    sync_type,
    method_name,
):
    async_signature = inspect.signature(getattr(async_type, method_name))
    sync_signature = inspect.signature(getattr(sync_type, method_name))

    assert tuple(async_signature.parameters.values()) == tuple(
        sync_signature.parameters.values()
    )
    assert async_signature.return_annotation != sync_signature.return_annotation


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
async def test_async_timeline_facade_has_supersede_recall_and_explain_parity(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="timeline")
    memory = await AgentMemory.open(tmp_path / "memory.db", scope=scope)

    async with memory:
        source = await memory.remember(
            "Launch date is July 20",
            kind=MemoryKind.FACT,
            valid_from="2026-07-01T00:00:00Z",
        )
        replacement = await memory.supersede(
            source.id,
            "Launch date is July 27",
            effective_at=EFFECTIVE_AT,
            reason="vendor correction",
            metadata={"source": "vendor"},
            confidence=0.95,
            idempotency_key="launch-date-v2",
        )
        known_at = _known_at(replacement)
        before = await memory.recall(
            "launch date",
            valid_at=VALID_BEFORE,
            known_at=known_at,
        )
        exact = await memory.recall(
            "launch date",
            valid_at=EFFECTIVE_AT,
            known_at=known_at,
        )
        receipt = await memory.explain(
            replacement.id,
            valid_at=EFFECTIVE_AT,
            known_at=known_at,
        )

        assert [result.memory.id for result in before] == [source.id]
        assert [result.memory.id for result in exact] == [replacement.id]
        assert receipt.memory == replacement
        assert receipt.evidence_memory_ids == (source.id,)
        assert [relation.target_id for relation in receipt.relations] == [source.id]
        with pytest.raises(ValidationError, match="as_of and valid_at"):
            await memory.recall(
                "launch date",
                as_of=VALID_BEFORE,
                valid_at=EFFECTIVE_AT,
            )


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
        lambda: memory.recall(
            "closed", valid_at=VALID_BEFORE, known_at=EFFECTIVE_AT
        ),
        lambda: memory.supersede(
            "missing", "closed", effective_at=EFFECTIVE_AT
        ),
        lambda: memory.explain("missing"),
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


def test_sync_timeline_facade_matches_async_and_reuses_one_private_loop(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="sync-timeline")

    with SyncAgentMemory.open(tmp_path / "memory.db", scope=scope) as memory:
        private_loop = memory._loop
        source = memory.remember(
            "Launch date is July 20",
            kind=MemoryKind.FACT,
            valid_from="2026-07-01T00:00:00Z",
        )
        replacement = memory.supersede(
            source.id,
            "Launch date is July 27",
            effective_at=EFFECTIVE_AT,
            reason="vendor correction",
            metadata={"source": "vendor"},
            confidence=0.95,
            idempotency_key="sync-launch-date-v2",
        )
        known_at = _known_at(replacement)
        before = memory.recall(
            "launch date",
            valid_at=VALID_BEFORE,
            known_at=known_at,
        )
        exact = memory.recall(
            "launch date",
            valid_at=EFFECTIVE_AT,
            known_at=known_at,
        )
        receipt = memory.explain(
            replacement.id,
            valid_at=EFFECTIVE_AT,
            known_at=known_at,
        )

        assert [result.memory.id for result in before] == [source.id]
        assert [result.memory.id for result in exact] == [replacement.id]
        assert receipt.memory == replacement
        assert receipt.evidence_memory_ids == (source.id,)
        assert [relation.target_id for relation in receipt.relations] == [source.id]
        assert memory._loop is private_loop
        assert not private_loop.is_closed()
        with pytest.raises(ValidationError, match="as_of and valid_at"):
            memory.recall(
                "launch date",
                as_of=VALID_BEFORE,
                valid_at=EFFECTIVE_AT,
            )

    operations = (
        lambda: memory.recall("closed", valid_at=VALID_BEFORE),
        lambda: memory.supersede(
            "missing", "closed", effective_at=EFFECTIVE_AT
        ),
        lambda: memory.explain("missing"),
    )
    for operation in operations:
        with pytest.raises(ClosedStoreError):
            operation()


def test_sync_client_refuses_calls_from_async_code_without_becoming_unusable(tmp_path):
    scope = MemoryScope(user_id="robin", agent_id="codex")
    memory = SyncAgentMemory.open(tmp_path / "memory.db", scope=scope)
    stored = memory.remember("Remain usable after rejected async call")

    async def call_sync_client():
        operations = (
            lambda: memory.get(stored.id),
            lambda: memory.recall("usable", valid_at=VALID_BEFORE),
            lambda: memory.supersede(
                stored.id, "rejected", effective_at=EFFECTIVE_AT
            ),
            lambda: memory.explain(stored.id),
        )
        for operation in operations:
            with pytest.raises(AgentMemoryError, match="^Use AgentMemory inside async code$"):
                operation()

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
def test_subprocess_v1_timeline_survives_migration_and_restart(tmp_path):
    path = tmp_path / "v1-timeline.db"
    script = r'''
import asyncio
import json
import sqlite3
import sys

from mnemocore.agent_memory import AgentMemory, MemoryScope
from mnemocore.agent_memory.schema import _AUXILIARY_SCHEMA_V1, _TABLE_SCHEMA_V1


PATH = sys.argv[1]
SCOPE = MemoryScope(user_id="robin", agent_id="codex", project_id="timeline")
MEMORY_ID = "foundation-fact"
REMEMBERED_AT = "2026-07-10T10:00:00.000000Z"
VALID_BEFORE = "2026-07-11T09:30:00.000000Z"
EFFECTIVE_AT = "2026-07-11T09:30:00.000001Z"


def create_v1_database():
    observation = {
        "memory_id": MEMORY_ID,
        "observation": {
            "kind": "fact",
            "content": "Launch date is July 20",
            "metadata": {},
            "status": "active",
            "confidence": 1.0,
            "observed_at": REMEMBERED_AT,
            "valid_from": "2026-07-01T00:00:00.000000Z",
            "valid_to": None,
            "created_at": REMEMBERED_AT,
            "updated_at": REMEMBERED_AT,
        },
    }
    with sqlite3.connect(PATH) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(_TABLE_SCHEMA_V1)
        connection.executescript(_AUXILIARY_SCHEMA_V1)
        scope_values = (
            SCOPE.scope_key,
            SCOPE.tenant_id,
            SCOPE.user_id,
            SCOPE.agent_id,
            SCOPE.project_id,
            SCOPE.session_id,
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
                "foundation-fact:remembered",
                MEMORY_ID,
                *scope_values,
                "remembered",
                json.dumps(observation, separators=(",", ":"), sort_keys=True),
                None,
                REMEMBERED_AT,
                REMEMBERED_AT,
            ),
        )
        connection.execute(
            """
            INSERT INTO memories (
                id, scope_key, tenant_id, user_id, agent_id, project_id,
                session_id, kind, content, metadata_json, status, confidence,
                observed_at, valid_from, valid_to, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                MEMORY_ID,
                *scope_values,
                "fact",
                "Launch date is July 20",
                "{}",
                "active",
                1.0,
                REMEMBERED_AT,
                "2026-07-01T00:00:00.000000Z",
                None,
                REMEMBERED_AT,
                REMEMBERED_AT,
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_history (
                id, memory_id, event_id, action, status, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "foundation-fact:remembered:history",
                MEMORY_ID,
                "foundation-fact:remembered",
                "remembered",
                "active",
                "{}",
                REMEMBERED_AT,
            ),
        )
        connection.execute(
            "INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)",
            (MEMORY_ID, "Launch date is July 20"),
        )
        connection.execute("PRAGMA user_version=1")


async def main():
    create_v1_database()
    async with await AgentMemory.open(PATH, scope=SCOPE) as memory:
        replacement = await memory.supersede(
            MEMORY_ID,
            "Launch date is July 27",
            effective_at=EFFECTIVE_AT,
            reason="vendor correction",
        )
    known_at = replacement.updated_at.isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    async with await AgentMemory.open(PATH, scope=SCOPE) as memory:
        before = await memory.recall(
            "launch date", valid_at=VALID_BEFORE, known_at=known_at
        )
        exact = await memory.recall(
            "launch date", valid_at=EFFECTIVE_AT, known_at=known_at
        )
        receipt = await memory.explain(
            replacement.id, valid_at=EFFECTIVE_AT, known_at=known_at
        )
        assert [result.memory.id for result in before] == [MEMORY_ID]
        assert [result.memory.id for result in exact] == [replacement.id]
        assert receipt.memory == replacement
        assert receipt.evidence_memory_ids == (MEMORY_ID,)
        assert len(receipt.evidence_event_ids) == 2
        print("Truth timeline survives restart")


asyncio.run(main())
'''
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

    assert completed.stdout == "Truth timeline survives restart\n"
    assert completed.stderr == ""
