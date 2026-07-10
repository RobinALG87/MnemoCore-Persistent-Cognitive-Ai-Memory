from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from mnemocore.agent_memory import (
    AgentMemoryError,
    ClosedStoreError,
    MemoryConflictError,
    MemoryEvent,
    MemoryEventType,
    MemoryHistoryEntry,
    MemoryKind,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryReceipt,
    MemoryRelation,
    MemoryScope,
    MemoryStatus,
    RecallResult,
    ScopeError,
    StorageError,
    ValidationError,
    utc_now,
)


def make_record(**overrides):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    values = {
        "id": "memory-1",
        "scope": MemoryScope(user_id="robin", agent_id="codex"),
        "kind": MemoryKind.OBSERVATION,
        "content": "Prefer compact public APIs",
        "metadata": {"source": "test"},
        "status": MemoryStatus.ACTIVE,
        "confidence": 0.9,
        "observed_at": now,
        "valid_from": now,
        "valid_to": now + timedelta(days=1),
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return MemoryRecord(**values)


def make_event(**overrides):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    values = {
        "id": "event-1",
        "scope": MemoryScope(user_id="robin", agent_id="codex"),
        "event_type": MemoryEventType.REMEMBERED,
        "payload": {"content": "Prefer compact public APIs"},
        "occurred_at": now,
        "created_at": now,
    }
    values.update(overrides)
    return MemoryEvent(**values)


def make_history(**overrides):
    values = {
        "id": "history-1",
        "memory_id": "memory-1",
        "event_id": "event-1",
        "action": MemoryEventType.REMEMBERED,
        "status": MemoryStatus.ACTIVE,
        "created_at": datetime(2026, 7, 10, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return MemoryHistoryEntry(**values)


def make_relation(**overrides):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    values = {
        "id": "relation-1",
        "scope": MemoryScope(user_id="robin", agent_id="codex"),
        "source_id": "memory-1",
        "target_id": "memory-2",
        "relation_type": "supersedes",
        "valid_from": now,
        "valid_to": now + timedelta(days=1),
        "confidence": 0.9,
        "event_id": "event-1",
        "created_at": now,
    }
    values.update(overrides)
    return MemoryRelation(**values)


def test_scope_requires_user_and_agent():
    with pytest.raises(ScopeError):
        MemoryScope(user_id="", agent_id="codex")
    with pytest.raises(ScopeError):
        MemoryScope(user_id="robin", agent_id="")


def test_scope_key_is_stable_and_unambiguous():
    scope = MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore")
    assert scope.scope_key == '["local","robin","codex","mnemocore",null]'
    assert scope != MemoryScope(user_id="robin", agent_id="codex", project_id="other")


def test_scope_normalizes_identifiers():
    scope = MemoryScope(
        user_id=" robin ",
        agent_id=" codex ",
        tenant_id=" tenant ",
        project_id=" project ",
        session_id=" session ",
    )

    assert (
        scope.tenant_id,
        scope.user_id,
        scope.agent_id,
        scope.project_id,
        scope.session_id,
    ) == ("tenant", "robin", "codex", "project", "session")


@pytest.mark.parametrize("value", ["bad\nvalue", "bad\x7fvalue", "x" * 257])
def test_scope_rejects_invalid_identifiers(value):
    with pytest.raises(ScopeError):
        MemoryScope(user_id=value, agent_id="codex")


def test_scope_is_immutable_and_slotted():
    scope = MemoryScope(user_id="robin", agent_id="codex")
    with pytest.raises(FrozenInstanceError):
        scope.user_id = "other"
    assert not hasattr(scope, "__dict__")


def test_enum_values_are_stable():
    assert [kind.value for kind in MemoryKind] == [
        "observation",
        "fact",
        "episode",
        "procedure",
        "preference",
        "summary",
    ]
    assert [status.value for status in MemoryStatus] == [
        "active",
        "superseded",
        "contradicted",
        "forgotten",
    ]
    assert [event_type.value for event_type in MemoryEventType] == [
        "remembered",
        "reinforced",
        "superseded",
        "contradicted",
        "forgotten",
        "restored",
    ]


def test_memory_record_has_the_public_fields_and_is_immutable():
    record = make_record()

    assert [field.name for field in fields(record)] == [
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
    ]
    with pytest.raises(FrozenInstanceError):
        record.content = "changed"
    assert not hasattr(record, "__dict__")


@pytest.mark.parametrize(
    "content",
    ["", "   ", "x" * 100_001],
    ids=["empty", "whitespace", "too-long"],
)
def test_memory_record_validates_content(content):
    with pytest.raises(ValidationError):
        make_record(content=content)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_memory_record_validates_confidence(confidence):
    with pytest.raises(ValidationError):
        make_record(confidence=confidence)


def test_memory_record_accepts_confidence_boundaries():
    assert make_record(confidence=0).confidence == 0
    assert make_record(confidence=1).confidence == 1


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(seconds=-1)])
def test_memory_record_requires_valid_to_after_valid_from(offset):
    valid_from = datetime(2026, 7, 10, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        make_record(valid_from=valid_from, valid_to=valid_from + offset)


def test_memory_event_payload_rejects_direct_mutation():
    event = make_event(payload={"direct": 1})

    assert isinstance(event.payload, MappingProxyType)
    with pytest.raises(TypeError):
        event.payload["direct"] = 2


def test_memory_event_payload_rejects_nested_mutation():
    event = make_event(payload={"nested": {"items": [1, {"value": 2}]}})

    with pytest.raises(TypeError):
        event.payload["nested"]["items"][1]["value"] = 3
    with pytest.raises(AttributeError):
        event.payload["nested"]["items"].append(3)


def test_memory_event_payload_is_detached_from_the_original_mapping():
    original = {"nested": {"items": ["original"]}}

    event = make_event(payload=original)
    original["nested"]["items"].append("changed")
    original["nested"]["new"] = "changed"
    original["top"] = "changed"

    assert event.payload == {"nested": {"items": ("original",)}}


def test_models_normalize_aware_datetimes_to_utc():
    eastern = timezone(timedelta(hours=2))
    local_time = datetime(2026, 7, 10, 14, 30, tzinfo=eastern)
    expected = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)

    record = make_record(
        observed_at=local_time,
        valid_from=local_time,
        valid_to=local_time + timedelta(hours=1),
        created_at=local_time,
        updated_at=local_time,
    )
    event = make_event(occurred_at=local_time, created_at=local_time)
    history = make_history(created_at=local_time)

    assert record.observed_at == expected
    assert record.valid_from == expected
    assert record.valid_to == expected + timedelta(hours=1)
    assert record.created_at == expected
    assert record.updated_at == expected
    assert event.occurred_at == expected
    assert event.created_at == expected
    assert history.created_at == expected
    for value in (
        record.observed_at,
        record.valid_from,
        record.valid_to,
        record.created_at,
        record.updated_at,
        event.occurred_at,
        event.created_at,
        history.created_at,
    ):
        assert value.tzinfo is timezone.utc


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (make_record, "observed_at"),
        (make_record, "valid_from"),
        (make_record, "valid_to"),
        (make_record, "created_at"),
        (make_record, "updated_at"),
        (make_event, "occurred_at"),
        (make_event, "created_at"),
        (make_history, "created_at"),
    ],
    ids=[
        "record-observed-at",
        "record-valid-from",
        "record-valid-to",
        "record-created-at",
        "record-updated-at",
        "event-occurred-at",
        "event-created-at",
        "history-created-at",
    ],
)
@pytest.mark.parametrize("invalid", [datetime(2026, 7, 10), "2026-07-10T00:00:00Z"])
def test_models_reject_naive_and_non_datetime_values(factory, field_name, invalid):
    with pytest.raises(ValidationError):
        factory(**{field_name: invalid})


def test_event_history_and_recall_models_are_frozen_and_slotted():
    now = utc_now()
    scope = MemoryScope(user_id="robin", agent_id="codex")
    record = make_record(scope=scope)
    event = MemoryEvent(
        id="event-1",
        scope=scope,
        event_type=MemoryEventType.REMEMBERED,
        payload={"content": record.content},
        occurred_at=now,
        created_at=now,
        memory_id=record.id,
        idempotency_key="request-1",
    )
    history = MemoryHistoryEntry(
        id="history-1",
        memory_id=record.id,
        event_id=event.id,
        action=MemoryEventType.REMEMBERED,
        status=MemoryStatus.ACTIVE,
        created_at=now,
        details={"reason": "test"},
    )
    result = RecallResult(
        memory=record,
        score=0.75,
        score_components={"bm25_rank": 1.0},
        reason="lexical match",
        evidence_ids=(event.id,),
    )

    assert event.memory_id == history.memory_id == result.memory.id
    assert result.evidence_ids == ("event-1",)
    for model, attribute in ((event, "id"), (history, "id"), (result, "reason")):
        assert not hasattr(model, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(model, attribute, "changed")


def test_public_models_recursively_detach_and_freeze_nested_values():
    record_metadata = {"nested": {"items": [1, {"name": "original"}]}}
    history_details = {"nested": {"reasons": ["first"]}}
    score_components = {"lexical": {"weights": [0.75]}}
    evidence_ids = ["event-1"]

    record = make_record(metadata=record_metadata)
    history = make_history(details=history_details)
    result = RecallResult(
        memory=record,
        score=0.75,
        score_components=score_components,
        evidence_ids=evidence_ids,
    )

    record_metadata["nested"]["items"][1]["name"] = "mutated"
    history_details["nested"]["reasons"].append("mutated")
    score_components["lexical"]["weights"].append(0.25)
    evidence_ids.append("event-2")

    assert record.metadata["nested"]["items"][1]["name"] == "original"
    assert history.details["nested"]["reasons"] == ("first",)
    assert result.score_components["lexical"]["weights"] == (0.75,)
    assert result.evidence_ids == ("event-1",)
    with pytest.raises(TypeError):
        record.metadata["nested"]["changed"] = True
    with pytest.raises(TypeError):
        history.details["nested"]["changed"] = True
    with pytest.raises(TypeError):
        result.score_components["lexical"]["changed"] = True


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (make_record, {"metadata": {"bad": float("nan")}}),
        (make_event, {"payload": {"bad": float("inf")}}),
        (make_history, {"details": {"bad": float("-inf")}}),
        (
            lambda **values: RecallResult(memory=make_record(), **values),
            {"score": float("nan")},
        ),
        (
            lambda **values: RecallResult(memory=make_record(), score=1.0, **values),
            {"score_components": {"bad": float("inf")}},
        ),
    ],
)
def test_public_models_reject_non_finite_floats(factory, kwargs):
    with pytest.raises(ValidationError):
        factory(**kwargs)


def test_utc_now_returns_an_aware_utc_datetime():
    result = utc_now()

    assert result.tzinfo is timezone.utc
    assert abs(datetime.now(timezone.utc) - result) < timedelta(seconds=1)


def test_memory_relation_has_the_public_fields_and_is_immutable():
    relation = make_relation()

    assert [field.name for field in fields(relation)] == [
        "id",
        "scope",
        "source_id",
        "target_id",
        "relation_type",
        "valid_from",
        "valid_to",
        "confidence",
        "event_id",
        "created_at",
    ]
    assert not hasattr(relation, "__dict__")
    with pytest.raises(FrozenInstanceError):
        relation.relation_type = "changed"


@pytest.mark.parametrize(
    "field_name",
    ["id", "source_id", "target_id", "relation_type", "event_id"],
)
def test_memory_relation_rejects_blank_identifiers_and_type(field_name):
    with pytest.raises(ValidationError):
        make_relation(**{field_name: "   "})


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.01, 1.01])
def test_memory_relation_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        make_relation(confidence=confidence)


def test_memory_relation_normalizes_timestamps_and_rejects_invalid_interval():
    local_timezone = timezone(timedelta(hours=2))
    valid_from = datetime(2026, 7, 10, 14, tzinfo=local_timezone)
    relation = make_relation(
        valid_from=valid_from,
        valid_to=valid_from + timedelta(hours=1),
        created_at=valid_from,
    )

    assert relation.valid_from == datetime(2026, 7, 10, 12, tzinfo=timezone.utc)
    assert relation.valid_to == datetime(2026, 7, 10, 13, tzinfo=timezone.utc)
    assert relation.created_at == datetime(2026, 7, 10, 12, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        make_relation(valid_to=make_relation().valid_from)


def test_memory_receipt_normalizes_and_detaches_collections():
    record = make_record()
    relation = make_relation()
    history = make_history()
    event_ids = ["event-1"]
    memory_ids = ["source-1"]
    relations = [relation]
    history_entries = [history]

    receipt = MemoryReceipt(
        memory=record,
        evidence_event_ids=event_ids,
        evidence_memory_ids=memory_ids,
        relations=relations,
        history=history_entries,
        explanation="This fact supersedes source-1 at 2026-07-10T12:00:00Z.",
    )
    event_ids.append("event-2")
    memory_ids.append("source-2")
    relations.clear()
    history_entries.clear()

    assert [field.name for field in fields(receipt)] == [
        "memory",
        "evidence_event_ids",
        "evidence_memory_ids",
        "relations",
        "history",
        "explanation",
    ]
    assert receipt.evidence_event_ids == ("event-1",)
    assert receipt.evidence_memory_ids == ("source-1",)
    assert receipt.relations == (relation,)
    assert receipt.history == (history,)
    assert not hasattr(receipt, "__dict__")
    with pytest.raises(FrozenInstanceError):
        receipt.explanation = "changed"
    with pytest.raises(AttributeError):
        receipt.relations.append(relation)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("evidence_event_ids", [""]),
        ("evidence_memory_ids", ["   "]),
        ("explanation", "   "),
    ],
)
def test_memory_receipt_rejects_blank_public_values(field_name, value):
    values = {
        "memory": make_record(),
        "evidence_event_ids": ["event-1"],
        "evidence_memory_ids": ["source-1"],
        "relations": [make_relation()],
        "history": [make_history()],
        "explanation": "Because the source was superseded.",
    }
    values[field_name] = value

    with pytest.raises(ValidationError):
        MemoryReceipt(**values)


def test_exception_hierarchy_is_public_and_stable():
    assert issubclass(ValidationError, AgentMemoryError)
    assert issubclass(ScopeError, ValidationError)
    assert issubclass(StorageError, AgentMemoryError)
    assert issubclass(MemoryNotFoundError, AgentMemoryError)
    assert issubclass(MemoryConflictError, AgentMemoryError)
    assert issubclass(ClosedStoreError, StorageError)
