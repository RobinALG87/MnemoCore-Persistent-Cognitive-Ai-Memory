"""
Tests for Phase 5.0 Provenance Tracking Module.
Tests ProvenanceRecord, ProvenanceOrigin, and LineageEvent.
"""

import pytest
from datetime import datetime, timezone

from mnemocore.core.provenance import (
    ProvenanceRecord,
    ProvenanceOrigin,
    LineageEvent,
    ORIGIN_TYPES,
)


# ------------------------------------------------------------------ #
#  LineageEvent                                                       #
# ------------------------------------------------------------------ #

class TestLineageEvent:
    def test_basic_creation(self):
        evt = LineageEvent(event="created", timestamp="2026-02-21T20:00:00+00:00")
        assert evt.event == "created"
        assert evt.actor is None
        assert evt.source_memories == []

    def test_to_dict_minimal(self):
        evt = LineageEvent(event="archived", timestamp="2026-02-21T20:00:00+00:00")
        d = evt.to_dict()
        assert d["event"] == "archived"
        assert "actor" not in d
        assert "source_memories" not in d

    def test_to_dict_full(self):
        evt = LineageEvent(
            event="consolidated",
            timestamp="2026-02-21T20:00:00+00:00",
            actor="worker",
            source_memories=["mem_a", "mem_b"],
            outcome=True,
            notes="cluster_7",
        )
        d = evt.to_dict()
        assert d["actor"] == "worker"
        assert d["source_memories"] == ["mem_a", "mem_b"]
        assert d["outcome"] is True

    def test_roundtrip(self):
        evt = LineageEvent(
            event="verified",
            timestamp="2026-02-21T20:00:00+00:00",
            actor="system",
            outcome=True,
        )
        restored = LineageEvent.from_dict(evt.to_dict())
        assert restored.event == "verified"
        assert restored.outcome is True


# ------------------------------------------------------------------ #
#  ProvenanceOrigin                                                   #
# ------------------------------------------------------------------ #

class TestProvenanceOrigin:
    def test_creation(self):
        origin = ProvenanceOrigin(type="observation", agent_id="agent-001")
        assert origin.type == "observation"
        assert origin.agent_id == "agent-001"

    def test_to_dict_omits_nones(self):
        origin = ProvenanceOrigin(type="dream")
        d = origin.to_dict()
        assert "agent_id" not in d
        assert "source_url" not in d

    def test_roundtrip(self):
        origin = ProvenanceOrigin(
            type="external_sync",
            source_url="https://example.com/feed",
            session_id="sess_abc",
        )
        restored = ProvenanceOrigin.from_dict(origin.to_dict())
        assert restored.type == "external_sync"
        assert restored.source_url == "https://example.com/feed"


# ------------------------------------------------------------------ #
#  ProvenanceRecord                                                   #
# ------------------------------------------------------------------ #

class TestProvenanceRecord:
    def test_new_creates_created_event(self):
        rec = ProvenanceRecord.new(origin_type="observation", agent_id="agent-42")
        assert rec.origin.type == "observation"
        assert rec.origin.agent_id == "agent-42"
        assert len(rec.lineage) == 1
        assert rec.lineage[0].event == "created"
        assert rec.version == 2  # starts at 1, incremented once

    def test_add_event_bumps_version(self):
        rec = ProvenanceRecord.new()
        v0 = rec.version
        rec.add_event("accessed", actor="query_engine")
        assert rec.version == v0 + 1
        assert rec.lineage[-1].event == "accessed"

    def test_mark_consolidated(self):
        rec = ProvenanceRecord.new()
        rec.mark_consolidated(["mem_a", "mem_b"])
        evt = rec.lineage[-1]
        assert evt.event == "consolidated"
        assert "mem_a" in evt.source_memories

    def test_mark_verified_successful(self):
        rec = ProvenanceRecord.new()
        rec.mark_verified(success=True)
        assert rec.is_verified()

    def test_mark_verified_failed_not_is_verified(self):
        rec = ProvenanceRecord.new()
        rec.mark_verified(success=False)
        assert not rec.is_verified()

    def test_mark_contradicted(self):
        rec = ProvenanceRecord.new()
        rec.mark_contradicted(contradiction_group_id="cg_001")
        assert rec.is_contradicted()

    def test_is_contradicted_false_when_clean(self):
        rec = ProvenanceRecord.new()
        assert not rec.is_contradicted()

    def test_last_event(self):
        rec = ProvenanceRecord.new()
        rec.add_event("verified", outcome=True)
        rec.add_event("archived")
        assert rec.last_event.event == "archived"

    def test_created_at_property(self):
        rec = ProvenanceRecord.new()
        # Should not raise and should return a parseable ISO string
        ts_str = rec.created_at
        dt = datetime.fromisoformat(ts_str)
        assert dt.tzinfo is not None

    def test_serialization_roundtrip(self):
        rec = ProvenanceRecord.new(
            origin_type="inference",
            agent_id="agent-99",
            session_id="s123",
        )
        rec.add_event("accessed")
        rec.mark_consolidated(["x", "y"])

        serialized = rec.to_dict()
        restored = ProvenanceRecord.from_dict(serialized)

        assert restored.origin.type == "inference"
        assert restored.version == rec.version
        assert len(restored.lineage) == len(rec.lineage)
        assert restored.lineage[0].event == "created"
        assert restored.lineage[-1].event == "consolidated"

    def test_unknown_origin_type_defaults_to_observation(self):
        rec = ProvenanceRecord.new(origin_type="INVALID_TYPE")
        assert rec.origin.type == "observation"

    def test_empty_lineage_last_event_is_none(self):
        rec = ProvenanceRecord(
            origin=ProvenanceOrigin(type="observation"),
            lineage=[],
        )
        assert rec.last_event is None

    def test_confidence_source_default(self):
        rec = ProvenanceRecord.new()
        assert rec.confidence_source == "bayesian_ltp"

    def test_repr(self):
        rec = ProvenanceRecord.new(origin_type="dream")
        r = repr(rec)
        assert "dream" in r
        assert "version" in r
