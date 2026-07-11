import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemocore.agent_memory import (
    MemoryEventType,
    MemoryKind,
    MemoryScope,
    MemoryStatus,
    StorageError,
    ValidationError,
)
from mnemocore.agent_memory.sqlite_codecs import (
    _row_to_event,
    _row_to_history,
    _row_to_lifecycle_record,
    _row_to_record,
    _row_to_relation,
    _scope_from_row,
    _timestamp_from_input,
    _timestamp_from_storage,
    _timestamp_to_storage,
)


PATH = Path("codec-test.db")
STAMP = "2026-07-12T10:00:00.000000Z"


def _scope_values(**overrides):
    values = {
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "agent_id": "agent-a",
        "project_id": "project-a",
        "session_id": None,
    }
    values.update(overrides)
    return values


def _record_row(**overrides):
    values = {
        "id": "memory-a",
        **_scope_values(),
        "kind": "fact",
        "content": "Generic fact",
        "metadata_json": '{"source":"fixture"}',
        "status": "active",
        "confidence": 0.75,
        "observed_at": STAMP,
        "valid_from": "2026-07-01T00:00:00+02:00",
        "valid_to": None,
        "created_at": STAMP,
        "updated_at": STAMP,
    }
    values.update(overrides)
    return values


def test_timestamp_codecs_normalize_offsets_to_utc():
    parsed = _timestamp_from_storage("2026-07-12T12:00:00+02:00", "stored_at")

    assert parsed == datetime(2026, 7, 12, 10, tzinfo=timezone.utc)
    assert _timestamp_to_storage(parsed) == STAMP
    assert _timestamp_from_input("2026-07-12T05:00:00-05:00", "input_at") == parsed
    assert _timestamp_from_input(None, "input_at") is None


@pytest.mark.parametrize("value", [123, "2026-07-12T10:00:00"])
def test_storage_timestamp_rejects_non_string_or_naive_values(value):
    with pytest.raises((TypeError, ValueError)):
        _timestamp_from_storage(value, "stored_at")


@pytest.mark.parametrize("value", [123, "2026-07-12T10:00:00", "not-a-date"])
def test_input_timestamp_wraps_invalid_values(value):
    with pytest.raises(
        ValidationError,
        match="input_at must be a valid timezone-aware ISO-8601 string|input_at must be an ISO-8601 string",
    ):
        _timestamp_from_input(value, "input_at")


def test_scope_and_record_hydration_preserve_enums_metadata_and_utc():
    row = _record_row()

    scope = _scope_from_row(row)
    record = _row_to_record(PATH, row)

    assert scope == MemoryScope(**_scope_values())
    assert record.scope == scope
    assert record.kind is MemoryKind.FACT
    assert record.status is MemoryStatus.ACTIVE
    assert record.metadata == {"source": "fixture"}
    assert record.valid_from == datetime(2026, 6, 30, 22, tzinfo=timezone.utc)


@pytest.mark.parametrize("metadata_json", ["{", json.dumps(["not", "a", "mapping"])])
def test_record_hydration_wraps_malformed_or_non_mapping_json_with_exact_cause(metadata_json):
    with pytest.raises(StorageError, match=r"^Failed to hydrate memory row from codec-test\.db:") as caught:
        _row_to_record(PATH, _record_row(metadata_json=metadata_json))

    if metadata_json == "{":
        assert isinstance(caught.value.__cause__, json.JSONDecodeError)
    else:
        assert isinstance(caught.value.__cause__, ValidationError)


def test_event_and_history_hydration_validate_enum_and_status():
    event = _row_to_event(
        PATH,
        {
            "id": "event-a",
            "memory_id": "memory-a",
            **_scope_values(),
            "event_type": "remembered",
            "payload_json": '{"memory_id":"memory-a"}',
            "idempotency_key": None,
            "occurred_at": STAMP,
            "created_at": STAMP,
        },
    )
    history = _row_to_history(
        PATH,
        {
            "id": "history-a",
            "memory_id": "memory-a",
            "event_id": "event-a",
            "action": "remembered",
            "status": "active",
            "details_json": "{}",
            "created_at": STAMP,
        },
    )

    assert event.event_type is MemoryEventType.REMEMBERED
    assert history.action is MemoryEventType.REMEMBERED
    assert history.status is MemoryStatus.ACTIVE
    with pytest.raises(StorageError) as caught:
        _row_to_history(PATH, {**history.__dict__} if hasattr(history, "__dict__") else {
            "id": "history-a", "memory_id": "memory-a", "event_id": "event-a",
            "action": "remembered", "status": "unknown", "details_json": "{}",
            "created_at": STAMP,
        })
    assert isinstance(caught.value.__cause__, ValueError)


def test_relation_uses_explicit_authorized_scope_and_lifecycle_overlays_projection():
    authorized_scope = MemoryScope(**_scope_values(session_id="authorized-session"))
    relation = _row_to_relation(
        PATH,
        {
            "id": "relation-a",
            "source_id": "source-a",
            "target_id": "target-a",
            "relation_type": "supersedes",
            "valid_from": STAMP,
            "valid_to": None,
            "confidence": 1.0,
            "event_id": "event-a",
            "created_at": STAMP,
        },
        authorized_scope,
    )
    lifecycle = _row_to_lifecycle_record(
        PATH,
        {
            **_record_row(status="active", valid_from="2026-07-01T00:00:00Z"),
            "lifecycle_status": "superseded",
            "lifecycle_valid_from": "2026-07-02T00:00:00Z",
            "lifecycle_valid_to": "2026-07-03T00:00:00Z",
        },
    )

    assert relation.scope is authorized_scope
    assert lifecycle.status is MemoryStatus.SUPERSEDED
    assert lifecycle.valid_from == datetime(2026, 7, 2, tzinfo=timezone.utc)
    assert lifecycle.valid_to == datetime(2026, 7, 3, tzinfo=timezone.utc)


def test_lifecycle_preserves_record_storage_error_without_double_wrapping():
    with pytest.raises(StorageError, match="Failed to hydrate memory row") as caught:
        _row_to_lifecycle_record(
            PATH,
            {
                **_record_row(metadata_json="{"),
                "lifecycle_status": "active",
                "lifecycle_valid_from": None,
                "lifecycle_valid_to": None,
            },
        )

    assert "Failed to hydrate memory lifecycle" not in str(caught.value)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)
