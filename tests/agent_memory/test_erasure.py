import sqlite3
from contextlib import closing

import pytest

from mnemocore.agent_memory import (
    MemoryConflictError,
    MemoryKind,
    MemoryNotFoundError,
    MemoryScope,
)
from mnemocore.agent_memory.sqlite_store import SQLiteMemoryStore


@pytest.mark.asyncio
async def test_erase_physically_removes_exact_memory_and_preserves_other_scopes(
    tmp_path,
):
    path = tmp_path / "memory.db"
    local = MemoryScope(user_id="local", agent_id="codex")
    foreign = MemoryScope(user_id="foreign", agent_id="codex")
    store = await SQLiteMemoryStore.open(path)
    removed = await store.remember(local, "secret phrase unique-erasure-token")
    retained = await store.remember(foreign, "retained phrase")

    receipt = await store.erase(local, [removed.id])

    assert receipt.memory_ids == (removed.id,)
    assert receipt.scope_key == local.scope_key
    assert not hasattr(receipt, "content")
    with pytest.raises(MemoryNotFoundError):
        await store.get(local, removed.id, include_forgotten=True)
    assert (await store.get(foreign, retained.id)).content == "retained phrase"
    await store.close()

    raw = path.read_bytes()
    assert b"unique-erasure-token" not in raw
    with closing(sqlite3.connect(path)) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM memory_events WHERE memory_id = ?", (removed.id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM memory_fts WHERE memory_id = ?", (removed.id,)
            ).fetchone()[0]
            == 0
        )
    reopened = await SQLiteMemoryStore.open(path)
    assert (await reopened.get(foreign, retained.id)).content == "retained phrase"
    await reopened.close()


@pytest.mark.asyncio
async def test_erase_rejects_foreign_owned_id_without_changing_database(tmp_path):
    path = tmp_path / "memory.db"
    local = MemoryScope(user_id="local", agent_id="codex")
    foreign = MemoryScope(user_id="foreign", agent_id="codex")
    store = await SQLiteMemoryStore.open(path)
    record = await store.remember(foreign, "must remain")

    with pytest.raises(MemoryNotFoundError):
        await store.erase(local, [record.id])

    assert (await store.get(foreign, record.id)).content == "must remain"
    await store.close()


@pytest.mark.asyncio
async def test_erase_rejects_partial_supersession_component_unless_cascaded(tmp_path):
    path = tmp_path / "memory.db"
    scope = MemoryScope(user_id="local", agent_id="codex")
    store = await SQLiteMemoryStore.open(path)
    source = await store.remember(scope, "old fact", kind=MemoryKind.FACT)
    replacement = await store.supersede(
        scope, source.id, "new fact", effective_at="2026-07-13T10:00:00Z"
    )

    with pytest.raises(MemoryConflictError, match="component"):
        await store.erase(scope, [source.id])

    receipt = await store.erase(scope, [source.id], cascade=True)
    assert set(receipt.memory_ids) == {source.id, replacement.id}
    await store.close()


@pytest.mark.asyncio
async def test_erase_removes_all_rows_depending_on_erased_event(tmp_path):
    path = tmp_path / "memory.db"
    scope = MemoryScope(user_id="local", agent_id="codex")
    store = await SQLiteMemoryStore.open(path)
    erased = await store.remember(scope, "erase event owner")
    retained_a = await store.remember(scope, "retained a")
    retained_b = await store.remember(scope, "retained b")
    await store.close()

    with closing(sqlite3.connect(path)) as connection:
        erased_event = connection.execute(
            "SELECT id FROM memory_events WHERE memory_id = ?", (erased.id,)
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO memory_relations (
                id, scope_key, source_id, target_id, relation_type,
                confidence, event_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "dependent-relation",
                scope.scope_key,
                retained_a.id,
                retained_b.id,
                "related",
                1.0,
                erased_event,
                "2026-07-13T10:00:00.000000Z",
            ),
        )
        connection.commit()

    store = await SQLiteMemoryStore.open(path)
    await store.erase(scope, [erased.id])
    await store.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            connection.execute(
                "SELECT count(*) FROM memory_relations WHERE id = 'dependent-relation'"
            ).fetchone()[0]
            == 0
        )
