"""
Tests for Phase 5.0 Confidence Calibration Module.
Tests ConfidenceEnvelopeGenerator for all confidence levels and edge cases.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from mnemocore.core.confidence import (
    ConfidenceEnvelopeGenerator,
    build_confidence_envelope,
    LEVEL_HIGH,
    LEVEL_MEDIUM,
    LEVEL_LOW,
    LEVEL_CONTRADICTED,
    LEVEL_STALE,
)
from mnemocore.core.provenance import ProvenanceRecord


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_node(
    ltp_strength: float = 0.9,
    access_count: int = 10,
    days_old: float = 5.0,
    bayes_mean: float | None = None,
):
    """Create a minimal mock MemoryNode."""
    node = MagicMock()
    node.ltp_strength = ltp_strength
    node.access_count = access_count
    now = datetime.now(timezone.utc)
    node.last_accessed = now - timedelta(days=days_old)
    # Optionally inject a Bayesian state mock
    if bayes_mean is not None:
        bayes = MagicMock()
        bayes.mean = bayes_mean
        node._bayes = bayes
    else:
        # Remove _bayes attribute so hasattr returns False
        del node._bayes
    return node


# ------------------------------------------------------------------ #
#  ConfidenceEnvelopeGenerator                                        #
# ------------------------------------------------------------------ #

class TestConfidenceEnvelopeGenerator:
    def test_high_confidence(self):
        node = _make_node(ltp_strength=0.92, access_count=8, days_old=3)
        prov = ProvenanceRecord.new(origin_type="observation")
        env = ConfidenceEnvelopeGenerator.build(node, prov)
        assert env["level"] == LEVEL_HIGH
        assert env["reliability"] >= 0.80

    def test_medium_confidence_low_reliability(self):
        node = _make_node(ltp_strength=0.65, access_count=3, days_old=5)
        env = ConfidenceEnvelopeGenerator.build(node)
        assert env["level"] == LEVEL_MEDIUM

    def test_low_confidence_insufficient_access(self):
        node = _make_node(ltp_strength=0.75, access_count=1, days_old=2)
        env = ConfidenceEnvelopeGenerator.build(node)
        assert env["level"] == LEVEL_LOW

    def test_low_confidence_poor_reliability(self):
        node = _make_node(ltp_strength=0.35, access_count=10, days_old=2)
        env = ConfidenceEnvelopeGenerator.build(node)
        assert env["level"] == LEVEL_LOW

    def test_stale_overrides_high_reliability(self):
        """A memory last verified 40 days ago should be STALE even if reliable."""
        node = _make_node(ltp_strength=0.98, access_count=20, days_old=40)
        env = ConfidenceEnvelopeGenerator.build(node)
        assert env["level"] == LEVEL_STALE
        assert env["staleness_days"] >= 30

    def test_contradicted_overrides_everything(self):
        """Contradicted memories take priority over any reliability score."""
        node = _make_node(ltp_strength=0.99, access_count=100, days_old=1)
        prov = ProvenanceRecord.new(origin_type="observation")
        prov.mark_contradicted("cg_999")
        env = ConfidenceEnvelopeGenerator.build(node, prov)
        assert env["level"] == LEVEL_CONTRADICTED
        assert env["is_contradicted"] is True

    def test_source_type_observation(self):
        node = _make_node(ltp_strength=0.9, access_count=6)
        prov = ProvenanceRecord.new(origin_type="observation")
        env = ConfidenceEnvelopeGenerator.build(node, prov)
        assert env["source_type"] == "observation"
        assert env["source_trust"] == 1.0

    def test_source_type_dream_lower_trust(self):
        node = _make_node(ltp_strength=0.9, access_count=6)
        prov = ProvenanceRecord.new(origin_type="dream")
        env = ConfidenceEnvelopeGenerator.build(node, prov)
        assert env["source_type"] == "dream"
        # Dream trust is 0.6, should not reach HIGH level
        assert env["level"] != LEVEL_HIGH

    def test_verified_event_resets_staleness(self):
        """A fresh verification event should make staleness very short."""
        node = _make_node(ltp_strength=0.9, access_count=8, days_old=50)
        prov = ProvenanceRecord.new(origin_type="observation")
        prov.mark_verified(success=True)
        env = ConfidenceEnvelopeGenerator.build(node, prov)
        # Verified just now → staleness should be near 0
        assert env["staleness_days"] < 1.0
        assert env["level"] != LEVEL_STALE

    def test_bayesian_state_used_over_ltp(self):
        """If node has _bayes, reliability = bayes.mean, not ltp_strength."""
        node = _make_node(ltp_strength=0.3, access_count=6, bayes_mean=0.95)
        env = ConfidenceEnvelopeGenerator.build(node)
        assert env["reliability"] == pytest.approx(0.95, abs=0.01)

    def test_no_provenance_uses_last_accessed(self):
        node = _make_node(ltp_strength=0.85, access_count=7, days_old=5)
        env = ConfidenceEnvelopeGenerator.build(node, provenance=None)
        assert env["source_type"] == "unknown"
        assert "level" in env

    def test_envelope_keys_present(self):
        node = _make_node()
        env = build_confidence_envelope(node)
        expected_keys = {
            "level", "reliability", "access_count",
            "staleness_days", "source_type", "source_trust", "is_contradicted",
        }
        assert expected_keys.issubset(env.keys())

    def test_module_shortcut_same_result(self):
        node = _make_node(ltp_strength=0.88, access_count=6)
        prov = ProvenanceRecord.new()
        r1 = ConfidenceEnvelopeGenerator.build(node, prov)
        r2 = build_confidence_envelope(node, prov)
        assert r1["level"] == r2["level"]
        assert r1["reliability"] == r2["reliability"]
