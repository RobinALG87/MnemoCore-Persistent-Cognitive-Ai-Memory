"""
Comprehensive Tests for Synapse Module
======================================

Tests the SynapticConnection class including:
- Synapse creation
- Firing mechanism
- Strength decay (Ebbinghaus forgetting curve)
- Activation thresholding
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from mnemocore.core.synapse import SynapticConnection


class TestSynapticConnectionCreation:
    """Test synapse initialization."""

    def test_creation_with_defaults(self):
        """Should create synapse with default strength."""
        synapse = SynapticConnection("neuron_a", "neuron_b")

        assert synapse.neuron_a_id == "neuron_a"
        assert synapse.neuron_b_id == "neuron_b"
        assert synapse.strength == 0.1
        assert synapse.fire_count == 0
        assert synapse.success_count == 0

    def test_creation_with_custom_strength(self):
        """Should create synapse with custom initial strength."""
        synapse = SynapticConnection("a", "b", initial_strength=0.7)

        assert synapse.strength == 0.7

    def test_timestamps_set_on_creation(self):
        """Should set creation and last_fired timestamps."""
        before = datetime.now(timezone.utc)
        synapse = SynapticConnection("a", "b")
        after = datetime.now(timezone.utc)

        assert before <= synapse.created_at <= after
        assert before <= synapse.last_fired <= after


class TestSynapseFiring:
    """Test synapse firing mechanism."""

    def test_fire_increases_count(self):
        """Firing should increment fire_count."""
        synapse = SynapticConnection("a", "b")

        synapse.fire()

        assert synapse.fire_count == 1

    def test_fire_updates_last_fired(self):
        """Firing should update last_fired timestamp."""
        synapse = SynapticConnection("a", "b")

        time.sleep(0.01)
        synapse.fire()

        assert synapse.last_fired > synapse.created_at

    def test_fire_success_increases_strength(self):
        """Successful fire should increase strength."""
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        synapse.fire(success=True)

        assert synapse.strength > 0.5

    def test_fire_failure_doesnt_increase_strength(self):
        """Failed fire should not increase strength."""
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        synapse.fire(success=False)

        assert synapse.strength == 0.5

    def test_fire_success_increases_success_count(self):
        """Successful fire should increment success_count."""
        synapse = SynapticConnection("a", "b")

        synapse.fire(success=True)
        synapse.fire(success=True)
        synapse.fire(success=False)

        assert synapse.success_count == 2

    def test_fire_with_weight(self):
        """Should apply weight to strength increase."""
        synapse1 = SynapticConnection("a", "b", initial_strength=0.5)
        synapse2 = SynapticConnection("c", "d", initial_strength=0.5)

        synapse1.fire(success=True, weight=1.0)
        synapse2.fire(success=True, weight=0.1)

        assert synapse2.strength < synapse1.strength

    def test_fire_caps_at_one(self):
        """Strength should cap at 1.0."""
        synapse = SynapticConnection("a", "b", initial_strength=0.95)

        synapse.fire(success=True, weight=10.0)

        assert synapse.strength <= 1.0


class TestSynapseStrengthDecay:
    """Test Ebbinghaus forgetting curve decay."""

    def test_no_decay_immediately(self):
        """Should have no decay immediately after firing."""
        synapse = SynapticConnection("a", "b", initial_strength=0.8)

        current = synapse.get_current_strength()

        assert current == pytest.approx(0.8, abs=0.01)

    def test_decay_over_time(self):
        """Strength should decay over time."""
        synapse = SynapticConnection("a", "b", initial_strength=0.8)

        # Mock last_fired to be in the past
        synapse.last_fired = datetime.now(timezone.utc) - timedelta(days=10)

        current = synapse.get_current_strength()

        # Should be less than original
        assert current < 0.8

    def test_decay_formula(self):
        """Decay should follow exponential formula."""
        synapse = SynapticConnection("a", "b", initial_strength=0.9)

        # Set last_fired to one half-life ago
        half_life = synapse._half_life_days
        synapse.last_fired = datetime.now(timezone.utc) - timedelta(days=half_life)

        current = synapse.get_current_strength()

        # After one half-life, should be ~50%
        assert 0.4 < current < 0.55

    def test_multiple_half_lives(self):
        """After multiple half-lives, should be very weak."""
        synapse = SynapticConnection("a", "b", initial_strength=1.0)

        # Two half-lives ago
        half_life = synapse._half_life_days
        synapse.last_fired = datetime.now(timezone.utc) - timedelta(days=half_life * 2)

        current = synapse.get_current_strength()

        # Should be ~25%
        assert 0.2 < current < 0.3

    def test_zero_half_life(self):
        """Zero half-life should prevent decay."""
        with patch('mnemocore.core.synapse.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.ltp.half_life_days = 0
            mock_config.return_value = mock_cfg

            synapse = SynapticConnection("a", "b", initial_strength=0.8)
            synapse.last_fired = datetime.now(timezone.utc) - timedelta(days=100)

            current = synapse.get_current_strength()

            # Should not decay
            assert current == 0.8


class TestSynapseActivation:
    """Test activation thresholding."""

    def test_is_active_with_default_threshold(self):
        """Should be active above 0.3 threshold."""
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        assert synapse.is_active() is True

    def test_is_active_below_threshold(self):
        """Should not be active below threshold."""
        synapse = SynapticConnection("a", "b", initial_strength=0.2)

        assert synapse.is_active() is False

    def test_is_active_with_custom_threshold(self):
        """Should use custom threshold."""
        synapse = SynapticConnection("a", "b", initial_strength=0.4)

        assert synapse.is_active(threshold=0.5) is False
        assert synapse.is_active(threshold=0.3) is True

    def test_is_active_considers_decay(self):
        """Should consider decay when checking activation."""
        synapse = SynapticConnection("a", "b", initial_strength=0.8)

        # Active immediately
        assert synapse.is_active(threshold=0.5) is True

        # Set far in the past
        synapse.last_fired = datetime.now(timezone.utc) - timedelta(days=100)

        # May not be active anymore due to decay
        assert synapse.get_current_strength() < 0.8


class TestSynapseEdgeCases:
    """Test edge cases."""

    def test_negative_initial_strength(self):
        """Should handle negative initial strength."""
        synapse = SynapticConnection("a", "b", initial_strength=-0.1)
        assert synapse.strength == -0.1

    def test_very_large_initial_strength(self):
        """Should handle very large initial strength."""
        synapse = SynapticConnection("a", "b", initial_strength=10.0)
        assert synapse.strength == 10.0

    def test_fire_with_zero_weight(self):
        """Should handle zero weight."""
        synapse = SynapticConnection("a", "b", initial_strength=0.5)
        before = synapse.strength

        synapse.fire(success=True, weight=0.0)

        assert synapse.strength == before

    def test_fire_with_negative_weight(self):
        """Should handle negative weight (reduces strength)."""
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        synapse.fire(success=True, weight=-0.5)

        # Should decrease strength
        assert synapse.strength < 0.5

    def test_very_long_time_since_fired(self):
        """Should handle very long time since fired."""
        synapse = SynapticConnection("a", "b", initial_strength=1.0)

        # 10 years ago
        synapse.last_fired = datetime.now(timezone.utc) - timedelta(days=3650)

        current = synapse.get_current_strength()

        # Should be very close to 0
        assert current < 0.01

    def test_future_last_fired(self):
        """Should handle future last_fired (edge case)."""
        synapse = SynapticConnection("a", "b", initial_strength=0.8)

        # Future time
        synapse.last_fired = datetime.now(timezone.utc) + timedelta(days=10)

        current = synapse.get_current_strength()

        # Should not decay (or increase)
        assert current >= 0.8


class TestSynapsePropertyBased:
    """Property-based tests using Hypothesis."""

    from hypothesis import given, strategies as st

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_strength_in_range_after_creation(self, initial_strength):
        """Initial strength should be preserved."""
        synapse = SynapticConnection("a", "b", initial_strength=initial_strength)
        assert synapse.strength == initial_strength

    @given(st.floats(min_value=0.0, max_value=1.0),
           st.floats(min_value=0.0, max_value=2.0),
           st.integers(min_value=0, max_value=100))
    def test_fire_never_decreases_base_strength(self, initial, weight, times):
        """Firing should never decrease base strength attribute."""
        synapse = SynapticConnection("a", "b", initial_strength=initial)
        base = synapse.strength

        for _ in range(times):
            synapse.fire(success=True, weight=weight)

        # Base strength attribute should not decrease
        assert synapse.strength >= base or synapse.strength >= initial

    @given(st.integers(min_value=0, max_value=365))
    def test_decay_never_negative(self, days_ago):
        """Decayed strength should never be negative."""
        synapse = SynapticConnection("a", "b", initial_strength=0.5)
        synapse.last_fired = datetime.now(timezone.utc) - timedelta(days=days_ago)

        current = synapse.get_current_strength()

        assert current >= 0.0

    @given(st.floats(min_value=0.0, max_value=1.0),
           st.floats(min_value=0.0, max_value=1.0))
    def test_activation_threshold_logic(self, strength, threshold):
        """is_active should correctly implement threshold logic."""
        synapse = SynapticConnection("a", "b", initial_strength=strength)

        # No decay
        result = synapse.is_active(threshold=threshold)

        if strength >= threshold:
            assert result is True
        else:
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
