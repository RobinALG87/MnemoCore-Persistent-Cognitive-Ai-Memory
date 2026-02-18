"""
Bayesian Long-Term Potentiation (LTP) Feedback Loop (Phase 4.0)
================================================================
Replaces the simple Hebbian LTP update with a Bayesian reliability model.

Core idea:
  Each synaptic connection and each memory node maintains a Beta distribution
  over its "true reliability" p ~ Beta(α, β).

  - α = accumulated success evidence (hits, correct retrievals)
  - β = accumulated failure evidence (misses, wrong retrievals, decay)

  The posterior mean  E[p] = α / (α + β)  is used as the reliability estimate.
  The posterior variance Var[p] = αβ / ((α+β)²(α+β+1)) reflects uncertainty.

  On each firing:
    success → α += 1  (evidence for reliability)
    failure → β += 1  (evidence for unreliability)

  The LTP update on MemoryNode follows the same Beta model, where:
    - "success" events: retrieval that helped produce a good answer
    - "failure" events: retrieval miss, low-EIG storage, or forced decay

Benefits over plain Hebbian:
  - Uncertainty-aware: new synapses have wide credible intervals → exploration bonus
  - Natural regularization: α and β act as pseudo-counts preventing overconfidence
  - Compatible with existing strength/ltp_strength fields (posterior mean replaces raw strength)

Public API:
    updater = BayesianLTPUpdater()
    updater.observe_synapse(synapse, success=True)
    strength = updater.posterior_mean(synapse)
    uncertainty = updater.posterior_uncertainty(synapse)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from loguru import logger


# ------------------------------------------------------------------ #
#  Beta distribution helpers                                          #
# ------------------------------------------------------------------ #

def _beta_mean(alpha: float, beta: float) -> float:
    """E[p] = α / (α + β)."""
    total = alpha + beta
    if total <= 0:
        return 0.5
    return alpha / total


def _beta_variance(alpha: float, beta: float) -> float:
    """Var[p] = αβ / ((α+β)²(α+β+1))."""
    total = alpha + beta
    if total <= 0:
        return 0.25   # Maximum variance of Beta(1,1)
    return (alpha * beta) / (total * total * (total + 1.0))


def _beta_std(alpha: float, beta: float) -> float:
    return math.sqrt(_beta_variance(alpha, beta))


def _beta_upper_credible(alpha: float, beta: float, z: float = 1.65) -> float:
    """
    Approximate upper credible bound using normal approximation.
    z=1.65 ≈ 90th percentile.  Used for UCB-style exploration bonus.
    """
    return min(1.0, _beta_mean(alpha, beta) + z * _beta_std(alpha, beta))


# ------------------------------------------------------------------ #
#  Mixin state stored alongside SynapticConnection / MemoryNode      #
# ------------------------------------------------------------------ #

@dataclass
class BayesianState:
    """
    Lightweight Beta distribution state for Bayesian LTP.
    Stored as extra fields; zero overhead when not used.

    alpha_prior / beta_prior: informative priors (default: uninformative Beta(1,1))
    """
    alpha: float = 1.0   # success pseudo-count
    beta_count: float = 1.0  # failure pseudo-count  (renamed to avoid clash with scipy.beta)

    def observe(self, success: bool, strength: float = 1.0) -> None:
        """
        Update posterior given an observation.

        Args:
            success: True → α += strength, False → β += strength
            strength: Fractional evidence weight (default 1.0).
        """
        if success:
            self.alpha += strength
        else:
            self.beta_count += strength

    @property
    def mean(self) -> float:
        return _beta_mean(self.alpha, self.beta_count)

    @property
    def uncertainty(self) -> float:
        """Standard deviation of the posterior."""
        return _beta_std(self.alpha, self.beta_count)

    @property
    def upper_credible(self) -> float:
        """90th percentile upper bound (UCB exploration bonus)."""
        return _beta_upper_credible(self.alpha, self.beta_count)

    @property
    def total_observations(self) -> float:
        # Subtract initial priors so total_observations = 0 when untouched
        return (self.alpha - 1.0) + (self.beta_count - 1.0)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta_count}

    @classmethod
    def from_dict(cls, d: dict) -> "BayesianState":
        return cls(alpha=d.get("alpha", 1.0), beta_count=d.get("beta", 1.0))


# ------------------------------------------------------------------ #
#  Core updater                                                       #
# ------------------------------------------------------------------ #

class BayesianLTPUpdater:
    """
    Manages Bayesian LTP state for synapses and memory nodes.

    Attach BayesianState to objects lazily to avoid changing data-class
    signatures across the codebase.
    """

    _ATTR = "_bayes"   # attribute name injected onto target objects

    # ---- Synapse helpers ------------------------------------------ #

    def get_synapse_state(self, synapse) -> BayesianState:
        """Get (or create) BayesianState for a SynapticConnection."""
        if not hasattr(synapse, self._ATTR):
            # Bootstrap from existing strength as evidence ratio
            s = synapse.strength
            # Seed: alpha ∝ successes, beta ∝ failures, total = fire_count
            fc = max(synapse.fire_count, 1)
            sc = max(synapse.success_count, 0)
            alpha = 1.0 + sc
            beta_count = 1.0 + (fc - sc)
            object.__setattr__(synapse, self._ATTR, BayesianState(alpha=alpha, beta_count=beta_count))
        return getattr(synapse, self._ATTR)

    def observe_synapse(self, synapse, success: bool, weight: float = 1.0) -> None:
        """
        Update Bayesian posterior for a synapse and synchronize back to
        the SynapticConnection.strength field (as posterior mean).
        """
        state = self.get_synapse_state(synapse)
        state.observe(success=success, strength=weight)
        # Write posterior mean back to the canonical `.strength` field
        synapse.strength = state.mean
        logger.debug(
            f"Synapse ({synapse.neuron_a_id[:8]}↔{synapse.neuron_b_id[:8]}) "
            f"Bayesian update — success={success} "
            f"α={state.alpha:.2f} β={state.beta_count:.2f} "
            f"→ p_mean={state.mean:.4f} ± {state.uncertainty:.4f}"
        )

    def synapse_strength_ucb(self, synapse) -> float:
        """
        Return the UCB (Upper Credible Bound) strength for exploration.
        Prefer under-explored synapses during associative spreading.
        """
        state = self.get_synapse_state(synapse)
        return state.upper_credible

    # ---- MemoryNode helpers --------------------------------------- #

    def get_node_state(self, node) -> BayesianState:
        """Get (or create) BayesianState for a MemoryNode."""
        if not hasattr(node, self._ATTR):
            # Bootstrap from epistemic + pragmatic values
            ev = getattr(node, "epistemic_value", 0.5)
            pv = getattr(node, "pragmatic_value", 0.0)
            combined = (ev + pv) / 2.0
            ac = max(getattr(node, "access_count", 1), 1)
            alpha = 1.0 + combined * ac
            beta_count = 1.0 + (1.0 - combined) * ac
            object.__setattr__(node, self._ATTR, BayesianState(alpha=alpha, beta_count=beta_count))
        return getattr(node, self._ATTR)

    def observe_node_retrieval(
        self, node, helpful: bool, eig_signal: float = 1.0
    ) -> float:
        """
        Record a retrieval outcome for a MemoryNode.

        Args:
            node: MemoryNode instance.
            helpful: Was this retrieval actually useful?
            eig_signal: Epistemic Information Gain from context (0–1).
                        Used as evidence weight: higher EIG → stronger update.

        Returns:
            Updated posterior mean LTP strength.
        """
        state = self.get_node_state(node)
        state.observe(success=helpful, strength=eig_signal)
        # Synchronize back to node.ltp_strength
        node.ltp_strength = state.mean
        logger.debug(
            f"Node {node.id[:8]} Bayesian retrieval update — helpful={helpful} "
            f"eig={eig_signal:.3f} → ltp={node.ltp_strength:.4f}"
        )
        return node.ltp_strength

    def node_ltp_ucb(self, node) -> float:
        """UCB estimate for node retrieval priority (exploration bonus)."""
        state = self.get_node_state(node)
        return state.upper_credible

    # ---- Serialization helpers ----------------------------------- #

    def synapse_to_dict(self, synapse) -> dict:
        """Serialize Bayesian state for persistence."""
        state = self.get_synapse_state(synapse)
        return state.to_dict()

    def synapse_from_dict(self, synapse, d: dict) -> None:
        """Restore Bayesian state from persisted dict."""
        state = BayesianState.from_dict(d)
        object.__setattr__(synapse, self._ATTR, state)
        synapse.strength = state.mean


# Module-level singleton
_UPDATER: BayesianLTPUpdater | None = None


def get_bayesian_updater() -> BayesianLTPUpdater:
    """Get the global Bayesian LTP updater singleton."""
    global _UPDATER
    if _UPDATER is None:
        _UPDATER = BayesianLTPUpdater()
    return _UPDATER
