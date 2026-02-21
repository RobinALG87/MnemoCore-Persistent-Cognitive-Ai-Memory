"""
Tests for Phase 5.0 Emotional Tagging Module.
"""

import pytest
from mnemocore.core.emotional_tag import (
    EmotionalTag,
    attach_emotional_tag,
    get_emotional_tag,
)
from unittest.mock import MagicMock


# ------------------------------------------------------------------ #
#  EmotionalTag construction and clamping                           #
# ------------------------------------------------------------------ #

class TestEmotionalTag:
    def test_default_neutral(self):
        t = EmotionalTag()
        assert t.valence == 0.0
        assert t.arousal == 0.0

    def test_clamping_valence_upper(self):
        t = EmotionalTag(valence=2.0)
        assert t.valence == 1.0

    def test_clamping_valence_lower(self):
        t = EmotionalTag(valence=-5.0)
        assert t.valence == -1.0

    def test_clamping_arousal_lower(self):
        t = EmotionalTag(arousal=-0.5)
        assert t.arousal == 0.0

    def test_clamping_arousal_upper(self):
        t = EmotionalTag(arousal=99.0)
        assert t.arousal == 1.0

    def test_salience_zero_for_neutral(self):
        t = EmotionalTag.neutral()
        assert t.salience() == pytest.approx(0.0)

    def test_salience_calculation(self):
        t = EmotionalTag(valence=0.8, arousal=0.5)
        expected = abs(0.8) * 0.5
        assert t.salience() == pytest.approx(expected)

    def test_salience_negative_valence(self):
        t = EmotionalTag(valence=-1.0, arousal=1.0)
        assert t.salience() == pytest.approx(1.0)

    def test_is_emotionally_significant(self):
        t = EmotionalTag(valence=0.8, arousal=0.9)
        assert t.is_emotionally_significant(threshold=0.3)

    def test_not_significant_when_calm(self):
        t = EmotionalTag(valence=0.5, arousal=0.1)
        # 0.5 * 0.1 = 0.05 < 0.3
        assert not t.is_emotionally_significant(threshold=0.3)

    def test_high_positive_factory(self):
        t = EmotionalTag.high_positive()
        assert t.valence == 1.0
        assert t.arousal == 1.0
        assert t.salience() == pytest.approx(1.0)

    def test_high_negative_factory(self):
        t = EmotionalTag.high_negative()
        assert t.valence == -1.0
        assert t.salience() == pytest.approx(1.0)

    def test_to_metadata_dict_keys(self):
        t = EmotionalTag(valence=0.6, arousal=0.7)
        d = t.to_metadata_dict()
        assert "emotional_valence" in d
        assert "emotional_arousal" in d
        assert "emotional_salience" in d

    def test_from_metadata_roundtrip(self):
        t = EmotionalTag(valence=-0.5, arousal=0.8)
        d = t.to_metadata_dict()
        restored = EmotionalTag.from_metadata(d)
        assert restored.valence == pytest.approx(-0.5, abs=0.001)
        assert restored.arousal == pytest.approx(0.8, abs=0.001)

    def test_repr(self):
        t = EmotionalTag(valence=0.7, arousal=0.4)
        r = repr(t)
        assert "valence" in r
        assert "arousal" in r


# ------------------------------------------------------------------ #
#  Node helpers                                                      #
# ------------------------------------------------------------------ #

class TestNodeHelpers:
    def test_attach_writes_metadata(self):
        node = MagicMock()
        node.metadata = {}
        tag = EmotionalTag(valence=0.9, arousal=0.8)
        attach_emotional_tag(node, tag)
        assert node.metadata["emotional_valence"] == pytest.approx(0.9)
        assert node.metadata["emotional_arousal"] == pytest.approx(0.8)

    def test_get_returns_neutral_when_empty(self):
        node = MagicMock()
        node.metadata = {}
        tag = get_emotional_tag(node)
        assert tag.valence == 0.0
        assert tag.arousal == 0.0

    def test_get_reads_existing_metadata(self):
        node = MagicMock()
        node.metadata = {"emotional_valence": -0.3, "emotional_arousal": 0.6}
        tag = get_emotional_tag(node)
        assert tag.valence == pytest.approx(-0.3)
        assert tag.arousal == pytest.approx(0.6)
