"""
Comprehensive Tests for Bayesian LTP Module
===========================================

Tests the Bayesian Long-Term Potentiation feedback loop including:
- Beta distribution calculations
- BayesianState management
- Synapse observation and updates
- MemoryNode observation and updates
- Serialization/deserialization
"""

import pytest
import math
from unittest.mock import MagicMock

from mnemocore.core.bayesian_ltp import (
    _beta_mean,
    _beta_variance,
    _beta_std,
    _beta_upper_credible,
    BayesianState,
    BayesianLTPUpdater,
    get_bayesian_updater,
)


class TestBetaDistribution:
    """Test Beta distribution helper functions."""

    def test_beta_mean_basic(self):
        """Mean should be alpha / (alpha + beta)."""
        assert _beta_mean(1.0, 1.0) == 0.5
        assert _beta_mean(3.0, 1.0) == 0.75
        assert _beta_mean(1.0, 3.0) == 0.25

    def test_beta_mean_zero_total(self):
        """Mean should return 0.5 when alpha + beta <= 0."""
        assert _beta_mean(0.0, 0.0) == 0.5
        assert _beta_mean(-1.0, -1.0) == 0.5

    def test_beta_variance_basic(self):
        """Variance calculation should be correct."""
        var = _beta_variance(1.0, 1.0)
        # Beta(1,1) is uniform, variance = 1/12 ≈ 0.0833
        assert abs(var - 0.0833) < 0.001

    def test_beta_variance_zero_total(self):
        """Variance should return 0.25 (maximum) when alpha + beta <= 0."""
        assert _beta_variance(0.0, 0.0) == 0.25
        assert _beta_variance(-1.0, -1.0) == 0.25

    def test_beta_std(self):
        """Standard deviation should be sqrt of variance."""
        mean = _beta_mean(2.0, 2.0)
        var = _beta_variance(2.0, 2.0)
        std = _beta_std(2.0, 2.0)
        assert abs(std - math.sqrt(var)) < 1e-10

    def test_beta_upper_credible(self):
        """Upper credible bound should be >= mean."""
        for alpha in [1.0, 2.0, 5.0, 10.0]:
            for beta in [1.0, 2.0, 5.0, 10.0]:
                mean = _beta_mean(alpha, beta)
                ucb = _beta_upper_credible(alpha, beta)
                assert ucb >= mean
                assert ucb <= 1.0

    def test_beta_upper_credible_custom_z(self):
        """Custom z-score should affect the bound."""
        ucb_1 = _beta_upper_credible(2.0, 2.0, z=1.0)
        ucb_2 = _beta_upper_credible(2.0, 2.0, z=2.0)
        assert ucb_2 > ucb_1

    def test_beta_upper_credible_clips_at_one(self):
        """Upper credible should not exceed 1.0."""
        # Even with very high z, should clip at 1.0
        ucb = _beta_upper_credible(100.0, 1.0, z=10.0)
        assert ucb <= 1.0


class TestBayesianState:
    """Test BayesianState dataclass."""

    def test_default_initialization(self):
        """Default state should have uniform prior (Beta(1,1))."""
        state = BayesianState()
        assert state.alpha == 1.0
        assert state.beta_count == 1.0
        assert state.mean == 0.5

    def test_custom_initialization(self):
        """Custom initialization should set alpha and beta."""
        state = BayesianState(alpha=5.0, beta_count=3.0)
        assert state.alpha == 5.0
        assert state.beta_count == 3.0

    def test_observe_success(self):
        """Observing success should increment alpha."""
        state = BayesianState(alpha=1.0, beta_count=1.0)
        state.observe(success=True, strength=1.0)
        assert state.alpha == 2.0
        assert state.beta_count == 1.0

    def test_observe_failure(self):
        """Observing failure should increment beta."""
        state = BayesianState(alpha=1.0, beta_count=1.0)
        state.observe(success=False, strength=1.0)
        assert state.alpha == 1.0
        assert state.beta_count == 2.0

    def test_observe_with_strength(self):
        """Observation strength should weight the update."""
        state = BayesianState(alpha=1.0, beta_count=1.0)
        state.observe(success=True, strength=0.5)
        assert state.alpha == 1.5

    def test_mean_property(self):
        """Mean should be alpha / (alpha + beta)."""
        state = BayesianState(alpha=3.0, beta_count=1.0)
        assert state.mean == 0.75

    def test_uncertainty_property(self):
        """Uncertainty should be standard deviation."""
        state = BayesianState(alpha=2.0, beta_count=2.0)
        var = (2.0 * 2.0) / (4.0 * 4.0 * 5.0)  # αβ / ((α+β)²(α+β+1))
        expected_std = math.sqrt(var)
        assert abs(state.uncertainty - expected_std) < 1e-10

    def test_upper_credible_property(self):
        """Upper credible should include exploration bonus."""
        state = BayesianState(alpha=2.0, beta_count=2.0)
        ucb = state.upper_credible
        assert ucb >= state.mean

    def test_total_observations(self):
        """Total observations should exclude priors."""
        state = BayesianState(alpha=1.0, beta_count=1.0)
        assert state.total_observations == 0.0

        state.observe(success=True, strength=1.0)
        assert state.total_observations == 1.0

        state.observe(success=False, strength=1.0)
        assert state.total_observations == 2.0

    def test_to_dict(self):
        """Serialization should produce correct dict."""
        state = BayesianState(alpha=3.0, beta_count=2.0)
        d = state.to_dict()
        assert d == {"alpha": 3.0, "beta": 2.0}

    def test_from_dict(self):
        """Deserialization should recreate state."""
        d = {"alpha": 4.0, "beta": 3.0}
        state = BayesianState.from_dict(d)
        assert state.alpha == 4.0
        assert state.beta_count == 3.0

    def test_from_dict_defaults(self):
        """Deserialization should use defaults for missing keys."""
        state = BayesianState.from_dict({})
        assert state.alpha == 1.0
        assert state.beta_count == 1.0

    def test_consecutive_observations(self):
        """Multiple observations should correctly update state."""
        state = BayesianState(alpha=1.0, beta_count=1.0)

        # 5 successes, 2 failures
        for _ in range(5):
            state.observe(success=True)
        for _ in range(2):
            state.observe(success=False)

        # alpha = 1 + 5 = 6, beta = 1 + 2 = 3
        assert state.alpha == 6.0
        assert state.beta_count == 3.0
        assert state.mean == pytest.approx(6.0 / 9.0)


class TestBayesianLTPUpdater:
    """Test BayesianLTPUpdater class."""

    def test_updater_initialization(self):
        """Updater should initialize without errors."""
        updater = BayesianLTPUpdater()
        assert updater._ATTR == "_bayes"

    def test_get_synapse_state_creates_new(self):
        """Getting synapse state should create BayesianState if missing."""
        from mnemocore.core.synapse import SynapticConnection

        updater = BayesianLTPUpdater()
        synapse = SynapticConnection("neuron_a", "neuron_b", initial_strength=0.7)
        # Set some counts
        synapse.fire_count = 10
        synapse.success_count = 7

        state = updater.get_synapse_state(synapse)

        assert hasattr(synapse, "_bayes")
        assert isinstance(state, BayesianState)
        # Should bootstrap from existing values
        assert state.alpha > 1.0  # 1.0 + success_count

    def test_get_synapse_state_returns_existing(self):
        """Getting synapse state should return existing if present."""
        from mnemocore.core.synapse import SynapticConnection

        updater = BayesianLTPUpdater()
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        # First call creates state
        state1 = updater.get_synapse_state(synapse)
        # Second call returns same state
        state2 = updater.get_synapse_state(synapse)

        assert state1 is state2

    def test_observe_synapse_success(self):
        """Observing synapse success should update state and strength."""
        from mnemocore.core.synapse import SynapticConnection

        updater = BayesianLTPUpdater()
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        updater.observe_synapse(synapse, success=True, weight=1.0)

        state = updater.get_synapse_state(synapse)
        assert state.alpha > 1.0
        # Strength should be updated to posterior mean
        assert synapse.strength == state.mean

    def test_observe_synapse_failure(self):
        """Observing synapse failure should update beta."""
        from mnemocore.core.synapse import SynapticConnection

        updater = BayesianLTPUpdater()
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        updater.observe_synapse(synapse, success=False, weight=1.0)

        state = updater.get_synapse_state(synapse)
        assert state.beta_count > 1.0
        assert synapse.strength < 0.5  # Should decrease

    def test_observe_synapse_with_weight(self):
        """Observation weight should affect update magnitude."""
        from mnemocore.core.synapse import SynapticConnection

        updater = BayesianLTPUpdater()
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        updater.observe_synapse(synapse, success=True, weight=0.3)

        state = updater.get_synapse_state(synapse)
        assert state.alpha == pytest.approx(1.3)  # 1.0 + 0.3

    def test_synapse_strength_ucb(self):
        """UCB strength should include exploration bonus."""
        from mnemocore.core.synapse import SynapticConnection

        updater = BayesianLTPUpdater()
        synapse = SynapticConnection("a", "b", initial_strength=0.5)

        ucb = updater.synapse_strength_ucb(synapse)

        # UCB should be >= base strength
        assert ucb >= synapse.strength

    def test_get_node_state_creates_new(self):
        """Getting node state should create BayesianState if missing."""
        from mnemocore.core.node import MemoryNode

        updater = BayesianLTPUpdater()
        node = MemoryNode("test_id", "test content", BinaryHDV.random(1024))

        state = updater.get_node_state(node)

        assert hasattr(node, "_bayes")
        assert isinstance(state, BayesianState)

    def test_get_node_state_no_attributes(self):
        """Should handle nodes without epistemic/pragmatic values."""
        from mnemocore.core.node import MemoryNode

        updater = BayesianLTPUpdater()
        node = MemoryNode("test_id", "test content", BinaryHDV.random(1024))

        # Node has default epistemic_value but may not have pragmatic_value
        state = updater.get_node_state(node)
        assert isinstance(state, BayesianState)

    def test_observe_node_retrieval_helpful(self):
        """Helpful retrieval should increase LTP strength."""
        updater = BayesianLTPUpdater()
        node = MagicMock()
        node.epistemic_value = 0.5
        node.pragmatic_value = 0.5
        node.access_count = 1
        node.ltp_strength = 0.5

        new_strength = updater.observe_node_retrieval(
            node, helpful=True, eig_signal=1.0
        )

        assert new_strength > 0.5
        assert node.ltp_strength == new_strength

    def test_observe_node_retrieval_not_helpful(self):
        """Unhelpful retrieval should decrease LTP strength."""
        updater = BayesianLTPUpdater()
        node = MagicMock()
        node.epistemic_value = 0.5
        node.pragmatic_value = 0.5
        node.access_count = 1
        node.ltp_strength = 0.8

        new_strength = updater.observe_node_retrieval(
            node, helpful=False, eig_signal=1.0
        )

        assert new_strength < 0.8

    def test_observe_node_with_eig_signal(self):
        """EIG signal should weight the update."""
        updater = BayesianLTPUpdater()
        node = MagicMock()
        node.epistemic_value = 0.5
        node.pragmatic_value = 0.5
        node.access_count = 1
        node.ltp_strength = 0.5

        # High EIG should have stronger effect
        updater.observe_node_retrieval(node, helpful=True, eig_signal=0.9)
        high_eig_strength = node.ltp_strength

        # Reset
        node.ltp_strength = 0.5
        updater.get_node_state(node)  # Recreate state

        # Low EIG should have weaker effect
        updater.observe_node_retrieval(node, helpful=True, eig_signal=0.1)
        low_eig_strength = node.ltp_strength

        assert high_eig_strength > low_eig_strength

    def test_node_ltp_ucb(self):
        """Node UCB should include exploration bonus."""
        updater = BayesianLTPUpdater()
        node = MagicMock()
        node.epistemic_value = 0.5
        node.pragmatic_value = 0.5
        node.access_count = 1

        ucb = updater.node_ltp_ucb(node)
        state = updater.get_node_state(node)

        assert ucb >= state.mean

    def test_synapse_to_dict(self):
        """Serialization should extract Bayesian state."""
        updater = BayesianLTPUpdater()
        synapse = MagicMock()
        synapse.strength = 0.5
        synapse.fire_count = 1
        synapse.success_count = 0

        updater.observe_synapse(synapse, success=True)
        d = updater.synapse_to_dict(synapse)

        assert "alpha" in d
        assert "beta" in d

    def test_synapse_from_dict(self):
        """Deserialization should restore Bayesian state."""
        updater = BayesianLTPUpdater()
        synapse = MagicMock()
        synapse.strength = 0.5

        d = {"alpha": 5.0, "beta": 2.0}
        updater.synapse_from_dict(synapse, d)

        state = updater.get_synapse_state(synapse)
        assert state.alpha == 5.0
        assert state.beta_count == 2.0
        assert synapse.strength == pytest.approx(5.0 / 7.0)


class TestBayesianLTPUpdaterIntegration:
    """Integration tests for Bayesian LTP."""

    def test_synapse_learning_sequence(self):
        """Synapse should learn from repeated observations."""
        updater = BayesianLTPUpdater()
        synapse = MagicMock()
        synapse.strength = 0.5
        synapse.fire_count = 0
        synapse.success_count = 0

        # Sequence of successes and failures
        outcomes = [True, True, True, False, True, True, False]
        for outcome in outcomes:
            updater.observe_synapse(synapse, success=outcome)

        # Final strength should reflect success rate
        # 5 successes, 2 failures = 5/7 ≈ 0.714
        assert 0.6 < synapse.strength < 0.8

    def test_uncertainty_decreases_with_evidence(self):
        """Uncertainty should decrease as evidence accumulates."""
        updater = BayesianLTPUpdater()
        synapse = MagicMock()
        synapse.strength = 0.5
        synapse.fire_count = 0
        synapse.success_count = 0

        state1 = updater.get_synapse_state(synapse)
        unc1 = state1.uncertainty

        # Add many observations
        for _ in range(100):
            updater.observe_synapse(synapse, success=True)

        state2 = updater.get_synapse_state(synapse)
        unc2 = state2.uncertainty

        assert unc2 < unc1

    def test_ucb_exploration_bonus(self):
        """UCB should prefer uncertain synapses."""
        updater = BayesianLTPUpdater()

        # Certain synapse (many observations)
        certain_synapse = MagicMock()
        certain_synapse.strength = 0.7
        certain_synapse.fire_count = 100
        certain_synapse.success_count = 70

        # Uncertain synapse (few observations but same mean)
        uncertain_synapse = MagicMock()
        uncertain_synapse.strength = 0.7
        uncertain_synapse.fire_count = 2
        uncertain_synapse.success_count = 1

        ucb_certain = updater.synapse_strength_ucb(certain_synapse)
        ucb_uncertain = updater.synapse_strength_ucb(uncertain_synapse)

        # Uncertain should get exploration bonus
        assert ucb_uncertain > ucb_certain

    def test_node_retrieval_learning(self):
        """Node LTP should adapt from retrieval feedback."""
        updater = BayesianLTPUpdater()
        node = MagicMock()
        node.epistemic_value = 0.5
        node.pragmatic_value = 0.5
        node.access_count = 1
        node.ltp_strength = 0.5

        # Series of helpful retrievals
        for _ in range(5):
            updater.observe_node_retrieval(node, helpful=True, eig_signal=0.8)

        assert node.ltp_strength > 0.6

        # Series of unhelpful retrievals
        for _ in range(5):
            updater.observe_node_retrieval(node, helpful=False, eig_signal=0.8)

        assert node.ltp_strength < 0.6


class TestBayesianLTPSingleton:
    """Test module-level singleton."""

    def test_get_bayesian_updater_singleton(self):
        """Should return same instance on repeated calls."""
        import mnemocore.core.bayesian_ltp as ltp_module
        ltp_module._UPDATER = None

        updater1 = get_bayesian_updater()
        updater2 = get_bayesian_updater()

        assert updater1 is updater2

    def test_get_bayesian_updater_creates_if_none(self):
        """Should create new instance if none exists."""
        import mnemocore.core.bayesian_ltp as ltp_module
        ltp_module._UPDATER = None

        updater = get_bayesian_updater()
        assert isinstance(updater, BayesianLTPUpdater)


class TestBayesianLTPPropertyBased:
    """Property-based tests using Hypothesis."""

    from hypothesis import given, strategies as st, assume

    @given(st.floats(min_value=0.0, max_value=10.0),
           st.floats(min_value=0.0, max_value=10.0))
    def test_beta_mean_in_range(self, alpha, beta):
        """Beta mean should always be in [0, 1]."""
        assume(alpha + beta > 0)
        mean = _beta_mean(alpha, beta)
        assert 0.0 <= mean <= 1.0

    @given(st.floats(min_value=0.0, max_value=10.0),
           st.floats(min_value=0.0, max_value=10.0))
    def test_beta_variance_non_negative(self, alpha, beta):
        """Beta variance should always be non-negative."""
        var = _beta_variance(alpha, beta)
        assert var >= 0.0

    @given(st.floats(min_value=0.0, max_value=10.0),
           st.floats(min_value=0.0, max_value=10.0))
    def test_beta_upper_credible_bounds(self, alpha, beta):
        """Upper credible bound should be in [0, 1]."""
        ucb = _beta_upper_credible(alpha, beta, z=1.65)
        assert 0.0 <= ucb <= 1.0

    @given(st.floats(min_value=0.0, max_value=1.0),
           st.floats(min_value=0.0, max_value=2.0))
    def test_bayesian_state_mean_after_observe(self, initial_alpha, initial_beta):
        """After observation, mean should move appropriately."""
        assume(initial_alpha > 0 or initial_beta > 0)

        state = BayesianState(alpha=max(1.0, initial_alpha), beta_count=max(1.0, initial_beta))
        mean_before = state.mean

        state.observe(success=True, strength=1.0)
        mean_after = state.mean

        # After success, mean should increase or stay same
        assert mean_after >= mean_before

    @given(st.lists(st.booleans(), min_size=0, max_size=50))
    def test_synapse_learning_converges(self, outcomes):
        """After many observations, strength should converge to success rate."""
        updater = BayesianLTPUpdater()
        synapse = MagicMock()
        synapse.strength = 0.5
        synapse.fire_count = 0
        synapse.success_count = 0

        if not outcomes:
            return

        for outcome in outcomes:
            updater.observe_synapse(synapse, success=outcome)

        success_rate = sum(outcomes) / len(outcomes)
        # Allow some tolerance due to priors
        assert abs(synapse.strength - success_rate) < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
