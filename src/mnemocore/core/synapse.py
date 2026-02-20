from datetime import datetime, timezone
import math

from .config import get_config


class SynapticConnection:
    """Biologically-inspired synapse with decay"""

    def __init__(
        self,
        neuron_a_id: str,
        neuron_b_id: str,
        initial_strength: float = 0.1
    ):
        self.neuron_a_id = neuron_a_id
        self.neuron_b_id = neuron_b_id
        self.strength = initial_strength
        self.created_at = datetime.now(timezone.utc)
        self.last_fired = datetime.now(timezone.utc)
        self.fire_count = 0
        self.success_count = 0  # For Hebbian learning
        # Cache half_life at creation to avoid get_config() on every strength call.
        self._half_life_days: float = get_config().ltp.half_life_days

    def fire(self, success: bool = True, weight: float = 1.0):
        """Activate synapse (strengthen if successful)"""
        self.last_fired = datetime.now(timezone.utc)
        self.fire_count += 1

        if success:
            # Phase 12.1: Allow aggressive weight multiplier for co-occurrence
            self.strength += (0.1 * weight) * (1 - self.strength)
            if self.strength > 1.0:
                self.strength = 1.0
            self.success_count += 1

    def get_current_strength(self) -> float:
        """
        Ebbinghaus forgetting curve
        Returns decayed strength based on age and cached half-life.
        """
        age_seconds = (datetime.now(timezone.utc) - self.last_fired).total_seconds()
        age_days = age_seconds / 86400.0

        # Exponential decay: exp(-λ * t)
        # Half-life formula: N(t) = N0 * (1/2)^(t / t_half)
        # Which is equivalent to N0 * exp(-ln(2) * t / t_half)
        half_life = self._half_life_days

        if half_life <= 0:
            return self.strength  # No decay

        decay = math.exp(-(math.log(2) / half_life) * age_days)

        return self.strength * decay

    def is_active(self, threshold: float = 0.3) -> bool:
        """Check if synapse is above activation threshold"""
        return self.get_current_strength() >= threshold
