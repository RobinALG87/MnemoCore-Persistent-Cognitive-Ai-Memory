"""
DEPRECATED: Legacy Circuit Breaker Implementation
==================================================

This module is DEPRECATED and will be removed in a future version.
Use `src.core.reliability` instead.

Migration:
    from src.core.resilience import storage_circuit_breaker, vector_circuit_breaker
    ->
    from src.core.reliability import storage_circuit_breaker, vector_circuit_breaker

    pybreaker.CircuitBreakerError
    ->
    from src.core.reliability import CircuitBreakerError

The new reliability module uses a native async-friendly implementation.
"""

import warnings

warnings.warn(
    "src.core.resilience is deprecated. Use src.core.reliability instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

import logging
import pybreaker

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
