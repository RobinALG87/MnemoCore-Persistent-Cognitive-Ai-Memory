import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from mnemocore.agent_memory import (
    MemoryConflictError,
    MemoryKind,
    MemoryNotFoundError,
    MemoryScope,
    MemoryStatus,
    StorageError,
    ValidationError,
)
from mnemocore.agent_memory.sqlite_store import SQLiteMemoryStore


EFFECTIVE_AT = "2026-07-11T09:30:00.000001Z"
SOURCE_VALID_FROM = "2026-07-01T00:00:00Z"
SOURCE_VALID_TO = "2026-08-01T00:00:00Z"


@pytest.fixture
def scope():
    return MemoryScope(
        tenant_id="tenant-a",
        user_id="robin",
        agent_id="codex",
        project_id="timeline",
    )


async def _remember_fact(store, scope, *, content="The launch date is July 20"):
    return await store.remember(
        scope,
        content,
        kind=MemoryKind.FACT,
        metadata={"source": {"name": "brief", "pages": [1, 2]}},
        confidence=0.8,
        observed_at="2026-07-02T10:00:00+02:00",
        valid_from=SOURCE_VALID_FROM,
        valid_to=SOURCE_VALID_TO,
    )


def _database_snapshot(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        result = {}
        for table, order_by in (
            ("memory_events", "id"),
            ("memories", "id"),
            ("memory_lifecycle", "memory_id, known_from, status"),
            ("memory_history", "id"),
            ("memory_evidence", "memory_id, source_memory_id, event_id"),
            ("memory_relations", "id"),
            ("memory_fts", "memory_id, content"),
        ):
            result[table] = [
                tuple(row)
                for row in connection.execute(
                    f"SELECT * FROM {table} ORDER BY {order_by}"
                ).fetchall()
            ]
        return result


@pytest.mark.asyncio
async def test_supersede_atomically_creates_full_event_and_all_projections(tmp_path, scope):
    path = tmp_path / "timeline.db"
    store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(store, scope)

    replacement = await store.supersede(
        scope,
        source.id,
        "The launch date is July 27",
        effective_at=EFFECTIVE_AT,
        reason="  vendor confirmed delay  ",
        metadata={"source": {"name": "vendor", "pages": [3]}},
        confidence=0.95,
        idempotency_key="supersede-launch-date",
    )

    assert replacement.kind is MemoryKind.FACT
    assert replacement.status is MemoryStatus.ACTIVE
    assert replacement.valid_from.isoformat() == "2026-07-11T09:30:00.000001+00:00"
    assert replacement.valid_to.isoformat() == "2026-08-01T00:00:00+00:00"
    superseded = await store.get(scope, source.id, include_forgotten=True)
    assert superseded.status is MemoryStatus.SUPERSEDED
    assert superseded.valid_to == replacement.valid_from

    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        events = connection.execute(
            "SELECT * FROM memory_events WHERE event_type = 'superseded'"
        ).fetchall()
        assert len(events) == 1
        event = events[0]
        payload = json.loads(event["payload_json"])
        assert event["memory_id"] == source.id
        assert event["idempotency_key"] == "supersede-launch-date"
        assert payload["source"] == {
            "id": source.id,
            "scope": {
                "scope_key": scope.scope_key,
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "agent_id": scope.agent_id,
                "project_id": scope.project_id,
                "session_id": scope.session_id,
            },
            "kind": "fact",
            "content": source.content,
            "metadata": {"source": {"name": "brief", "pages": [1, 2]}},
            "status": "superseded",
            "confidence": 0.8,
            "observed_at": "2026-07-02T08:00:00.000000Z",
            "valid_from": "2026-07-01T00:00:00.000000Z",
            "valid_to": EFFECTIVE_AT,
            "created_at": superseded.created_at.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
            "updated_at": superseded.updated_at.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
        }
        assert payload["replacement"]["id"] == replacement.id
        assert payload["replacement"]["content"] == replacement.content
        assert payload["reason"] == "vendor confirmed delay"

        lifecycles = connection.execute(
            """
            SELECT memory_id, status, known_from, known_to, valid_from, valid_to, event_id
            FROM memory_lifecycle
            WHERE memory_id IN (?, ?)
            ORDER BY known_from, memory_id, status
            """,
            (source.id, replacement.id),
        ).fetchall()
        assert len(lifecycles) == 3
        old_active = next(
            row for row in lifecycles if row["memory_id"] == source.id and row["status"] == "active"
        )
        old_closed = next(
            row
            for row in lifecycles
            if row["memory_id"] == source.id and row["status"] == "superseded"
        )
        new_active = next(row for row in lifecycles if row["memory_id"] == replacement.id)
        assert old_active["known_to"] == event["occurred_at"]
        assert old_closed["known_from"] == event["occurred_at"]
        assert old_closed["known_to"] is None
        assert old_closed["valid_to"] == EFFECTIVE_AT
        assert new_active["known_from"] == event["occurred_at"]
        assert new_active["known_to"] is None
        assert new_active["valid_from"] == EFFECTIVE_AT

        history = connection.execute(
            "SELECT * FROM memory_history WHERE event_id = ? ORDER BY id", (event["id"],)
        ).fetchall()
        assert [row["id"] for row in history] == [
            f"{event['id']}:history:replacement",
            f"{event['id']}:history:source",
        ]
        assert {(row["memory_id"], row["status"]) for row in history} == {
            (source.id, "superseded"),
            (replacement.id, "active"),
        }
        assert all(row["action"] == "superseded" for row in history)

        evidence = connection.execute(
            "SELECT * FROM memory_evidence WHERE event_id = ?", (event["id"],)
        ).fetchall()
        assert [tuple(row) for row in evidence] == [
            (
                replacement.id,
                source.id,
                event["id"],
                scope.scope_key,
                "supersedes",
                event["created_at"],
            )
        ]
        relation = connection.execute(
            "SELECT * FROM memory_relations WHERE event_id = ?", (event["id"],)
        ).fetchone()
        assert tuple(relation) == (
            f"{event['id']}:relation:supersedes",
            scope.scope_key,
            replacement.id,
            source.id,
            "supersedes",
            EFFECTIVE_AT,
            None,
            replacement.confidence,
            event["id"],
            event["created_at"],
        )
        assert connection.execute("SELECT count(*) FROM memory_fts").fetchone()[0] == 2

    await store.close()


@pytest.mark.asyncio
async def test_supersede_retry_lookup_precedes_new_payload_validation_and_returns_replacement(
    tmp_path, scope
):
    path = tmp_path / "timeline.db"
    store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(store, scope)
    first = await store.supersede(
        scope,
        source.id,
        "The launch date is July 27",
        effective_at=EFFECTIVE_AT,
        idempotency_key="same-transition",
    )
    before = _database_snapshot(path)

    retried = await store.supersede(
        scope,
        "not-the-original-source",
        " ",
        effective_at="not-a-timestamp",
        metadata={"bad": object()},
        confidence=float("nan"),
        idempotency_key="same-transition",
    )

    assert retried == first
    assert _database_snapshot(path) == before
    await store.close()


@pytest.mark.asyncio
async def test_supersede_retry_returns_replacement_after_it_is_later_superseded(tmp_path, scope):
    path = tmp_path / "timeline.db"
    store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(store, scope)
    first = await store.supersede(
        scope,
        source.id,
        "first replacement",
        effective_at="2026-07-10T00:00:00Z",
        idempotency_key="first-transition",
    )
    await store.supersede(
        scope,
        first.id,
        "second replacement",
        effective_at=EFFECTIVE_AT,
        idempotency_key="second-transition",
    )
    before = _database_snapshot(path)

    retried = await store.supersede(
        scope,
        "invalid-source-is-ignored",
        " ",
        effective_at="invalid-time-is-ignored",
        idempotency_key="first-transition",
    )

    assert retried.id == first.id
    assert retried.status is MemoryStatus.SUPERSEDED
    assert _database_snapshot(path) == before
    await store.close()


@pytest.mark.asyncio
async def test_supersede_rejects_foreign_id_as_not_found_without_mutation(tmp_path, scope):
    path = tmp_path / "timeline.db"
    store = await SQLiteMemoryStore.open(path)
    foreign_scope = MemoryScope(user_id="mallory", agent_id="codex", project_id="timeline")
    foreign = await _remember_fact(store, foreign_scope)
    before = _database_snapshot(path)

    with pytest.raises(MemoryNotFoundError):
        await store.supersede(
            scope,
            foreign.id,
            "replacement",
            effective_at=EFFECTIVE_AT,
        )

    assert _database_snapshot(path) == before
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["type", "status", "before", "at-start", "at-end", "after"])
async def test_supersede_rejects_non_active_fact_or_invalid_effective_time_without_mutation(
    tmp_path, scope, case
):
    path = tmp_path / f"{case}.db"
    store = await SQLiteMemoryStore.open(path)
    if case == "type":
        source = await store.remember(scope, "an observation", kind=MemoryKind.OBSERVATION)
        effective_at = EFFECTIVE_AT
    else:
        source = await _remember_fact(store, scope)
        effective_at = {
            "status": EFFECTIVE_AT,
            "before": "2026-06-30T23:59:59.999999Z",
            "at-start": "2026-07-01T00:00:00Z",
            "at-end": "2026-08-01T00:00:00Z",
            "after": "2026-08-01T00:00:00.000001Z",
        }.get(case, EFFECTIVE_AT)
        if case == "status":
            await store.supersede(
                scope,
                source.id,
                "first replacement",
                effective_at=EFFECTIVE_AT,
                idempotency_key="first",
            )
    before = _database_snapshot(path)

    with pytest.raises(MemoryConflictError):
        await store.supersede(
            scope,
            source.id,
            "second replacement",
            effective_at=effective_at,
            idempotency_key="second",
        )

    assert _database_snapshot(path) == before
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": " "},
        {"metadata": {"bad": object()}},
        {"confidence": float("nan")},
        {"effective_at": "not-a-time"},
        {"reason": object()},
        {"idempotency_key": []},
    ],
)
async def test_supersede_validates_all_new_input_before_first_insert(tmp_path, scope, kwargs):
    path = tmp_path / "timeline.db"
    store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(store, scope)
    before = _database_snapshot(path)
    arguments = {
        "content": "replacement",
        "effective_at": EFFECTIVE_AT,
        "reason": "correction",
        "metadata": {},
        "confidence": 1.0,
        "idempotency_key": "new-transition",
    }
    arguments.update(kwargs)

    with pytest.raises(ValidationError):
        await store.supersede(scope, source.id, **arguments)

    assert _database_snapshot(path) == before
    await store.close()


ROLLBACK_TRIGGERS = [
    pytest.param(
        "CREATE TRIGGER fail_point AFTER INSERT ON memory_events WHEN NEW.event_type = 'superseded' BEGIN SELECT RAISE(ABORT, 'event'); END",
        id="event",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER UPDATE ON memories WHEN NEW.status = 'superseded' BEGIN SELECT RAISE(ABORT, 'source'); END",
        id="source-projection",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER INSERT ON memories WHEN NEW.status = 'active' AND NEW.kind = 'fact' BEGIN SELECT RAISE(ABORT, 'replacement'); END",
        id="replacement-projection",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER UPDATE ON memory_lifecycle WHEN NEW.known_to IS NOT NULL BEGIN SELECT RAISE(ABORT, 'lifecycle-close'); END",
        id="lifecycle-close",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER INSERT ON memory_lifecycle WHEN NEW.status = 'superseded' BEGIN SELECT RAISE(ABORT, 'source-lifecycle'); END",
        id="source-lifecycle",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER INSERT ON memory_lifecycle WHEN NEW.status = 'active' AND NEW.memory_id != (SELECT memory_id FROM memory_events WHERE event_type = 'remembered' ORDER BY created_at LIMIT 1) BEGIN SELECT RAISE(ABORT, 'replacement-lifecycle'); END",
        id="replacement-lifecycle",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER INSERT ON memory_history WHEN NEW.status = 'superseded' BEGIN SELECT RAISE(ABORT, 'source-history'); END",
        id="source-history",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER INSERT ON memory_history WHEN NEW.status = 'active' AND NEW.action = 'superseded' BEGIN SELECT RAISE(ABORT, 'replacement-history'); END",
        id="replacement-history",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER INSERT ON memory_evidence BEGIN SELECT RAISE(ABORT, 'evidence'); END",
        id="evidence",
    ),
    pytest.param(
        "CREATE TRIGGER fail_point AFTER INSERT ON memory_relations BEGIN SELECT RAISE(ABORT, 'relation'); END",
        id="relation",
    ),
    pytest.param("FAIL_REPLACEMENT_FTS_INSERT", id="fts"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ROLLBACK_TRIGGERS)
async def test_supersede_rolls_back_after_every_projection_boundary(tmp_path, scope, trigger):
    path = tmp_path / "rollback.db"
    store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(store, scope)
    with closing(sqlite3.connect(path)) as connection:
        if trigger == "FAIL_REPLACEMENT_FTS_INSERT":
            existing_fts = connection.execute(
                "SELECT memory_id, content FROM memory_fts"
            ).fetchall()
            connection.execute("DROP TABLE memory_fts")
            connection.execute(
                """
                CREATE TABLE memory_fts (
                    memory_id TEXT,
                    content TEXT CHECK(content != 'replacement')
                )
                """
            )
            connection.executemany(
                "INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)",
                existing_fts,
            )
        else:
            connection.execute(trigger)
        connection.commit()
    before = _database_snapshot(path)

    with pytest.raises(StorageError):
        await store.supersede(
            scope,
            source.id,
            "replacement",
            effective_at=EFFECTIVE_AT,
            idempotency_key="rollback-transition",
        )

    assert _database_snapshot(path) == before
    await store.close()


@pytest.mark.asyncio
async def test_supersede_same_idempotency_key_race_yields_one_replacement(tmp_path, scope):
    path = tmp_path / "same-key.db"
    first_store = await SQLiteMemoryStore.open(path)
    second_store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(first_store, scope)

    first, second = await asyncio.gather(
        first_store.supersede(
            scope, source.id, "replacement", effective_at=EFFECTIVE_AT, idempotency_key="race"
        ),
        second_store.supersede(
            scope,
            source.id,
            "different ignored payload",
            effective_at=EFFECTIVE_AT,
            idempotency_key="race",
        ),
    )

    assert first == second
    with closing(sqlite3.connect(path)) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM memory_events WHERE event_type = 'superseded'"
            ).fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT count(*) FROM memories").fetchone()[0] == 2
    await first_store.close()
    await second_store.close()


@pytest.mark.asyncio
async def test_supersede_different_idempotency_key_race_yields_success_and_conflict(
    tmp_path, scope
):
    path = tmp_path / "different-keys.db"
    first_store = await SQLiteMemoryStore.open(path)
    second_store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(first_store, scope)

    results = await asyncio.gather(
        first_store.supersede(
            scope,
            source.id,
            "first replacement",
            effective_at=EFFECTIVE_AT,
            idempotency_key="race-1",
        ),
        second_store.supersede(
            scope,
            source.id,
            "second replacement",
            effective_at=EFFECTIVE_AT,
            idempotency_key="race-2",
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, MemoryConflictError) for result in results) == 1
    with closing(sqlite3.connect(path)) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM memory_events WHERE event_type = 'superseded'"
            ).fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT count(*) FROM memories").fetchone()[0] == 2
    await first_store.close()
    await second_store.close()


def _canonical_timestamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@pytest.mark.asyncio
async def test_recall_uses_independent_valid_and_known_time_boundaries(tmp_path, scope):
    path = tmp_path / "timeline.db"
    store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(
        store,
        scope,
        content="Launch timeline says July 20",
    )
    replacement = await store.supersede(
        scope,
        source.id,
        "Launch timeline says July 27",
        effective_at=EFFECTIVE_AT,
        reason="vendor confirmed delay",
        confidence=0.95,
    )

    with closing(sqlite3.connect(path)) as connection:
        superseded_at = datetime.fromisoformat(
            connection.execute(
                "SELECT occurred_at FROM memory_events WHERE event_type = 'superseded'"
            ).fetchone()[0].replace("Z", "+00:00")
        )
    known_before = _canonical_timestamp(superseded_at - timedelta(microseconds=1))
    known_exact = _canonical_timestamp(superseded_at)
    known_after = _canonical_timestamp(superseded_at + timedelta(microseconds=1))
    valid_before = "2026-07-11T09:30:00.000000Z"
    valid_exact = EFFECTIVE_AT
    valid_after = "2026-07-11T09:30:00.000002Z"

    async def recalled_id(*, valid_at, known_at):
        results = await store.recall(
            scope,
            "launch timeline",
            valid_at=valid_at,
            known_at=known_at,
        )
        assert len(results) == 1
        return results[0]

    current_current = await recalled_id(valid_at=valid_after, known_at=known_after)
    current_past = await recalled_id(valid_at=valid_before, known_at=known_after)
    past_past = await recalled_id(valid_at=valid_before, known_at=known_before)
    past_different_validity = await recalled_id(
        valid_at=valid_after,
        known_at=known_before,
    )

    assert current_current.memory.id == replacement.id
    assert current_current.memory.status is MemoryStatus.ACTIVE
    assert current_past.memory.id == source.id
    assert current_past.memory.status is MemoryStatus.SUPERSEDED
    assert past_past.memory.id == source.id
    assert past_past.memory.status is MemoryStatus.ACTIVE
    assert past_past.memory.valid_to.isoformat() == "2026-08-01T00:00:00+00:00"
    assert past_different_validity.memory.id == source.id

    assert (await recalled_id(valid_at=valid_before, known_at=known_after)).memory.id == source.id
    assert (await recalled_id(valid_at=valid_exact, known_at=known_after)).memory.id == replacement.id
    assert (await recalled_id(valid_at=valid_after, known_at=known_after)).memory.id == replacement.id
    assert (await recalled_id(valid_at=valid_after, known_at=known_exact)).memory.id == replacement.id

    alias = await store.recall(
        scope,
        "launch timeline",
        as_of=valid_exact,
        known_at=known_after,
    )
    explicit = await store.recall(
        scope,
        "launch timeline",
        valid_at=valid_exact,
        known_at=known_after,
    )
    assert alias == explicit
    await store.close()


@pytest.mark.asyncio
async def test_recall_returns_complete_deduplicated_temporal_evidence_chain(tmp_path, scope):
    path = tmp_path / "timeline.db"
    store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(
        store,
        scope,
        content="Launch timeline says July 20",
    )
    replacement = await store.supersede(
        scope,
        source.id,
        "Launch timeline says July 27",
        effective_at=EFFECTIVE_AT,
    )

    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT id, event_type, occurred_at
            FROM memory_events
            WHERE scope_key = ?
            ORDER BY occurred_at, created_at, id
            """,
            (scope.scope_key,),
        ).fetchall()
    remembered_event_id = next(row[0] for row in rows if row[1] == "remembered")
    superseded_event_id = next(row[0] for row in rows if row[1] == "superseded")
    superseded_at = datetime.fromisoformat(
        next(row[2] for row in rows if row[1] == "superseded").replace("Z", "+00:00")
    )
    known_before = _canonical_timestamp(superseded_at - timedelta(microseconds=1))
    known_after = _canonical_timestamp(superseded_at + timedelta(microseconds=1))

    replacement_result = (
        await store.recall(
            scope,
            "launch timeline",
            valid_at="2026-07-11T09:30:00.000002Z",
            known_at=known_after,
        )
    )[0]
    current_source_result = (
        await store.recall(
            scope,
            "launch timeline",
            valid_at="2026-07-11T09:30:00.000000Z",
            known_at=known_after,
        )
    )[0]
    historical_source_result = (
        await store.recall(
            scope,
            "launch timeline",
            valid_at="2026-07-11T09:30:00.000000Z",
            known_at=known_before,
        )
    )[0]

    assert replacement_result.memory.id == replacement.id
    assert replacement_result.evidence_ids == (remembered_event_id, superseded_event_id)
    assert current_source_result.memory.id == source.id
    assert current_source_result.evidence_ids == (remembered_event_id, superseded_event_id)
    assert historical_source_result.evidence_ids == (remembered_event_id,)
    assert len(replacement_result.evidence_ids) == len(set(replacement_result.evidence_ids))
    await store.close()


@pytest.mark.asyncio
async def test_explain_returns_deterministic_same_scope_supersession_receipt(tmp_path, scope):
    path = tmp_path / "timeline.db"
    store = await SQLiteMemoryStore.open(path)
    source = await _remember_fact(
        store,
        scope,
        content="Launch timeline says July 20",
    )
    replacement = await store.supersede(
        scope,
        source.id,
        "Launch timeline says July 27",
        effective_at=EFFECTIVE_AT,
        reason="vendor confirmed delay",
        metadata={"private": "do not expose this payload"},
        confidence=0.95,
    )

    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        events = connection.execute(
            """
            SELECT * FROM memory_events
            WHERE scope_key = ?
            ORDER BY occurred_at, created_at, id
            """,
            (scope.scope_key,),
        ).fetchall()
    remembered_event = next(row for row in events if row["event_type"] == "remembered")
    superseded_event = next(row for row in events if row["event_type"] == "superseded")
    known_after = _canonical_timestamp(
        datetime.fromisoformat(superseded_event["occurred_at"].replace("Z", "+00:00"))
        + timedelta(microseconds=1)
    )

    receipt = await store.explain(
        scope,
        replacement.id,
        valid_at="2026-07-11T09:30:00.000002Z",
        known_at=known_after,
    )

    assert receipt.memory == replacement
    assert receipt.memory.confidence == 0.95
    assert receipt.memory.valid_from.isoformat() == "2026-07-11T09:30:00.000001+00:00"
    assert receipt.evidence_event_ids == (remembered_event["id"], superseded_event["id"])
    assert receipt.evidence_memory_ids == (source.id,)
    assert len(receipt.relations) == 1
    relation = receipt.relations[0]
    assert relation.scope == scope
    assert relation.source_id == replacement.id
    assert relation.target_id == source.id
    assert relation.relation_type == "supersedes"
    assert relation.confidence == 0.95
    assert relation.valid_from == replacement.valid_from
    assert relation.event_id == superseded_event["id"]
    assert {
        (entry.memory_id, entry.event_id, entry.status)
        for entry in receipt.history
        if entry.event_id == superseded_event["id"]
    } == {
        (source.id, superseded_event["id"], MemoryStatus.SUPERSEDED),
        (replacement.id, superseded_event["id"], MemoryStatus.ACTIVE),
    }
    assert receipt.explanation == (
        f"Memory {replacement.id} supersedes memory {source.id} from {EFFECTIVE_AT} "
        f"via event {superseded_event['id']}; confidence 0.950000."
    )
    assert "vendor confirmed delay" not in receipt.explanation
    assert "do not expose" not in receipt.explanation
    assert await store.explain(
        scope,
        replacement.id,
        valid_at="2026-07-11T09:30:00.000002Z",
        known_at=known_after,
    ) == receipt
    await store.close()


@pytest.mark.asyncio
async def test_explain_hides_foreign_scope_ids_like_missing_ids(tmp_path, scope):
    store = await SQLiteMemoryStore.open(tmp_path / "timeline.db")
    foreign_scope = MemoryScope(user_id="mallory", agent_id="codex", project_id="timeline")
    foreign = await _remember_fact(store, foreign_scope)

    errors = []
    for memory_id in (foreign.id, "random-missing-id"):
        with pytest.raises(MemoryNotFoundError) as caught:
            await store.explain(scope, memory_id)
        errors.append(str(caught.value))

    assert errors == [
        f"Memory {foreign.id!r} was not found in this scope",
        "Memory 'random-missing-id' was not found in this scope",
    ]
    await store.close()
