import json
from datetime import datetime, timedelta, timezone

import pytest

from mnemocore.agent_memory import (
    MemoryEvent,
    MemoryEventType,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    StorageError,
    ValidationError,
)
from mnemocore.agent_memory.timeline import (
    _contains_half_open,
    build_superseded_payload,
    normalize_timeline_query,
    parse_superseded_payload,
)


NOW = datetime(2026, 7, 10, 12, 0, 0, 123456, tzinfo=timezone.utc)
BOUNDARY = datetime(2026, 7, 11, 9, 30, 0, 1, tzinfo=timezone.utc)


def _scope(**overrides):
    values = {
        "tenant_id": "tenant-a",
        "user_id": "robin",
        "agent_id": "codex",
        "project_id": "timeline",
        "session_id": None,
    }
    values.update(overrides)
    return MemoryScope(**values)


def _record(memory_id, *, status, valid_from, valid_to, updated_at, **overrides):
    values = {
        "id": memory_id,
        "scope": _scope(),
        "kind": MemoryKind.FACT,
        "content": f"content for {memory_id}",
        "metadata": {"nested": {"values": [1, 2]}},
        "status": status,
        "confidence": 0.875,
        "observed_at": NOW - timedelta(days=2),
        "valid_from": valid_from,
        "valid_to": valid_to,
        "created_at": NOW - timedelta(days=2),
        "updated_at": updated_at,
    }
    values.update(overrides)
    return MemoryRecord(**values)


def _supersession_fixture():
    source = _record(
        "source-1",
        status=MemoryStatus.SUPERSEDED,
        valid_from=NOW - timedelta(days=10),
        valid_to=BOUNDARY,
        updated_at=NOW,
    )
    replacement = _record(
        "replacement-1",
        status=MemoryStatus.ACTIVE,
        valid_from=BOUNDARY,
        valid_to=NOW + timedelta(days=30),
        updated_at=NOW,
        observed_at=NOW,
        created_at=NOW,
    )
    payload = build_superseded_payload(
        source,
        replacement,
        reason="  corrected after verification  ",
        relation_id="relation-1",
    )
    event = MemoryEvent(
        id="event-1",
        memory_id=source.id,
        scope=source.scope,
        event_type=MemoryEventType.SUPERSEDED,
        payload=payload,
        occurred_at=NOW,
        created_at=NOW,
    )
    return source, replacement, payload, event


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-07-10T14:00:00.12+02:00", "2026-07-10T12:00:00.120000Z"),
        ("2026-07-10T12:00:00Z", "2026-07-10T12:00:00.000000Z"),
        ("2026-07-10T12:00:00.123456+00:00", "2026-07-10T12:00:00.123456Z"),
    ],
)
def test_supersede_timeline_query_normalizes_aware_inputs_to_canonical_utc(value, expected):
    valid_at, known_at = normalize_timeline_query(
        as_of=None,
        valid_at=value,
        known_at=value,
        now=NOW,
    )

    assert (valid_at, known_at) == (expected, expected)


def test_supersede_timeline_query_uses_as_of_as_valid_at_alias_and_now_defaults():
    valid_at, known_at = normalize_timeline_query(
        as_of="2026-07-11T11:30:00.000001+02:00",
        valid_at=None,
        known_at=None,
        now=NOW,
    )

    assert valid_at == "2026-07-11T09:30:00.000001Z"
    assert known_at == "2026-07-10T12:00:00.123456Z"


def test_supersede_timeline_query_rejects_as_of_with_valid_at():
    with pytest.raises(ValidationError, match="as_of and valid_at"):
        normalize_timeline_query(
            as_of="2026-07-10T12:00:00Z",
            valid_at="2026-07-10T12:00:00Z",
            known_at=None,
            now=NOW,
        )


@pytest.mark.parametrize("field", ["as_of", "valid_at", "known_at"])
def test_supersede_timeline_query_rejects_naive_or_non_string_inputs(field):
    values = {"as_of": None, "valid_at": None, "known_at": None, "now": NOW}
    values[field] = "2026-07-10T12:00:00" if field != "known_at" else object()

    with pytest.raises(ValidationError, match=field):
        normalize_timeline_query(**values)


def test_supersede_half_open_boundaries_select_the_replacement_at_exact_microsecond():
    before = BOUNDARY - timedelta(microseconds=1)

    assert _contains_half_open(before, NOW - timedelta(days=10), BOUNDARY)
    assert not _contains_half_open(BOUNDARY, NOW - timedelta(days=10), BOUNDARY)
    assert _contains_half_open(BOUNDARY, BOUNDARY, None)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (None, None),
        ("   ", None),
        ("  corrected after verification  ", "corrected after verification"),
    ],
)
def test_supersede_payload_normalizes_reason_and_contains_complete_snapshots(reason, expected):
    source, replacement, _, _ = _supersession_fixture()

    payload = build_superseded_payload(
        source,
        replacement,
        reason=reason,
        relation_id="relation-1",
    )

    assert payload["effective_at"] == "2026-07-11T09:30:00.000001Z"
    assert payload["reason"] == expected
    assert payload["source_memory_id"] == source.id
    assert payload["replacement_memory_id"] == replacement.id
    assert payload["scope_key"] == source.scope.scope_key
    assert payload["relation"] == {
        "id": "relation-1",
        "relation_type": "supersedes",
        "source_id": replacement.id,
        "target_id": source.id,
    }
    assert payload["evidence"] == {
        "memory_id": replacement.id,
        "source_memory_id": source.id,
        "relation": "supersedes",
    }
    for name, record in (("source", source), ("replacement", replacement)):
        snapshot = payload[name]
        assert snapshot["id"] == record.id
        assert snapshot["scope"] == {
            "scope_key": record.scope.scope_key,
            "tenant_id": "tenant-a",
            "user_id": "robin",
            "agent_id": "codex",
            "project_id": "timeline",
            "session_id": None,
        }
        assert set(snapshot) == {
            "id",
            "scope",
            "kind",
            "content",
            "metadata",
            "status",
            "confidence",
            "observed_at",
            "valid_from",
            "valid_to",
            "created_at",
            "updated_at",
        }


def test_supersede_payload_parser_detaches_and_validates_complete_payload():
    source, replacement, payload, event = _supersession_fixture()
    mutable_payload = json.loads(json.dumps(payload))
    mutable_event = MemoryEvent(
        id=event.id,
        memory_id=event.memory_id,
        scope=event.scope,
        event_type=event.event_type,
        payload=mutable_payload,
        occurred_at=event.occurred_at,
        created_at=event.created_at,
    )

    replay = parse_superseded_payload(mutable_event, path="memory.db")
    mutable_payload["source"]["metadata"]["nested"]["values"].append(99)

    assert replay.source == source
    assert replay.replacement == replacement
    assert replay.effective_at == BOUNDARY
    assert replay.reason == "corrected after verification"
    assert replay.relation_id == "relation-1"
    assert replay.relation_type == "supersedes"
    assert replay.evidence_memory_id == replacement.id
    assert replay.evidence_source_memory_id == source.id


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.pop("replacement"), "replacement"),
        (lambda payload: payload["source"].pop("content"), "source"),
        (lambda payload: payload["source"]["scope"].update(user_id="mallory"), "scope"),
        (lambda payload: payload["source"]["scope"].update(user_id=" robin "), "scope"),
        (lambda payload: payload.update(scope_key="foreign"), "scope"),
        (lambda payload: payload.update(source_memory_id="other"), "source_memory_id"),
        (
            lambda payload: payload["replacement"].update(valid_from="2026-07-11T09:30:00.000002Z"),
            "boundary",
        ),
        (
            lambda payload: payload["source"].update(valid_to="2026-07-11T09:30:00.000002Z"),
            "boundary",
        ),
        (lambda payload: payload["relation"].update(relation_type="related_to"), "relation"),
        (lambda payload: payload["evidence"].update(memory_id="other"), "evidence"),
    ],
)
def test_supersede_payload_parser_rejects_incomplete_or_denormalized_values(mutation, match):
    _, _, payload, event = _supersession_fixture()
    corrupt = json.loads(json.dumps(payload))
    mutation(corrupt)
    corrupt_event = MemoryEvent(
        id=event.id,
        memory_id=event.memory_id,
        scope=event.scope,
        event_type=event.event_type,
        payload=corrupt,
        occurred_at=event.occurred_at,
        created_at=event.created_at,
    )

    with pytest.raises(StorageError, match=match):
        parse_superseded_payload(corrupt_event, path="memory.db")
