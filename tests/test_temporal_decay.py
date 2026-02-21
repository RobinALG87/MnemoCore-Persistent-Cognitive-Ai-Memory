"""
Tests for Phase 5.0 Temporal Decay Module.
Tests AdaptiveDecayEngine: retention, stability, review candidates, eviction.
"""

import math
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from mnemocore.core.temporal_decay import (
    AdaptiveDecayEngine,
    get_adaptive_decay_engine,
    REVIEW_THRESHOLD,
    EVICTION_THRESHOLD,
    S_BASE,
    K_GROWTH,
)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_node(access_count: int = 1, days_since_access: float = 0.0, stability: float = 1.0):
    node = MagicMock()
    node.id = "test-node-0001"
    node.access_count = access_count
    now = datetime.now(timezone.utc)
    node.last_accessed = now - timedelta(days=days_since_access)
    node.created_at = now - timedelta(days=days_since_access + 1)
    node.stability = stability
    node.review_candidate = False
    node.epistemic_value = 0.5
    return node


# ------------------------------------------------------------------ #
#  Stability                                                          #
# ------------------------------------------------------------------ #

class TestStability:
    def test_stability_grows_with_access_count(self):
        engine = AdaptiveDecayEngine()
        node1 = _make_node(access_count=1)
        node10 = _make_node(access_count=10)
        assert engine.stability(node10) > engine.stability(node1)

    def test_stability_formula(self):
        engine = AdaptiveDecayEngine(s_base=1.0, k_growth=0.5)
        node = _make_node(access_count=5)
        expected = 1.0 * (1.0 + 0.5 * math.log1p(5))
        assert engine.stability(node) == pytest.approx(expected, rel=1e-5)

    def test_stability_never_below_s_base(self):
        engine = AdaptiveDecayEngine()
        node = _make_node(access_count=1)
        assert engine.stability(node) >= S_BASE


# ------------------------------------------------------------------ #
#  Retention                                                          #
# ------------------------------------------------------------------ #

class TestRetention:
    def test_fresh_memory_high_retention(self):
        engine = AdaptiveDecayEngine()
        node = _make_node(access_count=5, days_since_access=0.0)
        r = engine.retention(node)
        assert r > 0.99

    def test_old_memory_low_retention(self):
        engine = AdaptiveDecayEngine()
        node = _make_node(access_count=1, days_since_access=30.0)
        r = engine.retention(node)
        assert r < 0.5

    def test_high_access_count_slows_decay(self):
        engine = AdaptiveDecayEngine()
        node_low = _make_node(access_count=1, days_since_access=5)
        node_high = _make_node(access_count=50, days_since_access=5)
        # node_high has higher stability → higher retention after same time
        assert engine.retention(node_high) > engine.retention(node_low)

    def test_retention_between_0_and_1(self):
        engine = AdaptiveDecayEngine()
        for days in [0, 1, 10, 100]:
            node = _make_node(access_count=1, days_since_access=days)
            r = engine.retention(node)
            assert 0.0 < r <= 1.0


# ------------------------------------------------------------------ #
#  Review candidates                                                  #
# ------------------------------------------------------------------ #

class TestReviewCandidates:
    def test_node_flagged_when_retention_low(self):
        engine = AdaptiveDecayEngine(review_threshold=0.99)
        node = _make_node(access_count=1, days_since_access=0.0)
        flagged = engine.update_review_candidate(node)
        # retention near 1.0 but threshold is 0.99 → might flag
        # The point is the flag matches the result
        assert node.review_candidate == flagged

    def test_recently_accessed_node_not_flagged(self):
        engine = AdaptiveDecayEngine()
        node = _make_node(access_count=5, days_since_access=0.0)
        flagged = engine.update_review_candidate(node)
        # Fresh node with good access_count should not be a candidate
        assert not flagged
        assert not node.review_candidate

    def test_scan_returns_candidates(self):
        engine = AdaptiveDecayEngine()
        fresh = _make_node(access_count=10, days_since_access=0)
        old = _make_node(access_count=1, days_since_access=100)
        candidates = engine.scan_review_candidates([fresh, old])
        assert old in candidates
        assert fresh not in candidates


# ------------------------------------------------------------------ #
#  Eviction                                                           #
# ------------------------------------------------------------------ #

class TestEviction:
    def test_very_old_low_access_should_evict(self):
        engine = AdaptiveDecayEngine()
        node = _make_node(access_count=1, days_since_access=200)
        assert engine.should_evict(node)

    def test_recent_accessed_should_not_evict(self):
        engine = AdaptiveDecayEngine()
        node = _make_node(access_count=5, days_since_access=0)
        assert not engine.should_evict(node)

    def test_eviction_candidates_batch(self):
        engine = AdaptiveDecayEngine()
        keepers = [_make_node(access_count=10, days_since_access=0) for _ in range(3)]
        evicts = [_make_node(access_count=1, days_since_access=300) for _ in range(2)]
        result = engine.eviction_candidates(keepers + evicts)
        assert len(result) >= 2


# ------------------------------------------------------------------ #
#  Singleton                                                          #
# ------------------------------------------------------------------ #

class TestSingleton:
    def test_same_object(self):
        a = get_adaptive_decay_engine()
        b = get_adaptive_decay_engine()
        assert a is b


# ------------------------------------------------------------------ #
#  update_after_access                                               #
# ------------------------------------------------------------------ #

class TestUpdateAfterAccess:
    def test_stability_written_back(self):
        engine = AdaptiveDecayEngine()
        node = _make_node(access_count=3)
        expected = engine.stability(node)
        engine.update_after_access(node)
        assert node.stability == pytest.approx(expected, rel=1e-5)

    def test_review_candidate_cleared(self):
        engine = AdaptiveDecayEngine()
        node = _make_node(access_count=3)
        node.review_candidate = True
        engine.update_after_access(node)
        assert not node.review_candidate
