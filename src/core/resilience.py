import logging
import pybreaker
from .config import get_config

logger = logging.getLogger(__name__)

class LogListener(pybreaker.CircuitBreakerListener):
    def state_change(self, cb, old_state, new_state):
        logger.warning(f"Circuit Breaker '{cb.name}' changed state: {old_state} -> {new_state}")

# Configuration from settings if available, else defaults
REDIS_FAIL_MAX = 5
REDIS_RESET_TIMEOUT = 60
VECTOR_FAIL_MAX = 3
VECTOR_RESET_TIMEOUT = 30

# Initialize Breakers
storage_circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=REDIS_FAIL_MAX,
    reset_timeout=REDIS_RESET_TIMEOUT,
    listeners=[LogListener()],
    name="storage_circuit_breaker"
)

vector_circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=VECTOR_FAIL_MAX,
    reset_timeout=VECTOR_RESET_TIMEOUT,
    listeners=[LogListener()],
    name="vector_circuit_breaker"
)

def circuit_breaker(breaker):
    """Decorator for synchronous functions."""
    return breaker.decorate

def async_circuit_breaker(breaker):
    """Decorator for asynchronous functions."""
    # pybreaker.CircuitBreaker.call_async is the method, 
    # but for decorators we use the same mechanism or custom wrapper.
    # pybreaker's @breaker.decorate works for both sync and async in recent versions.
    return breaker.decorate
