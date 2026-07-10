from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone

import pytest

from mnemocore.agent_memory import (
    AgentMemoryError,
    ClosedStoreError,
    MemoryEvent,
    MemoryEventType,
    MemoryHistoryEntry,
    MemoryKind,
    MemoryNotFoundError,
    MemoryRecord,
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


def test_utc_now_returns_an_aware_utc_datetime():
    result = utc_now()

    assert result.tzinfo is timezone.utc
    assert abs(datetime.now(timezone.utc) - result) < timedelta(seconds=1)


def test_exception_hierarchy_is_public_and_stable():
    assert issubclass(ValidationError, AgentMemoryError)
    assert issubclass(ScopeError, ValidationError)
    assert issubclass(StorageError, AgentMemoryError)
    assert issubclass(MemoryNotFoundError, AgentMemoryError)
    assert issubclass(ClosedStoreError, StorageError)
