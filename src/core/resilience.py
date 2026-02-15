"""
Resilience Patterns (Phase 3.5.4)
=================================
Circuit Breaker and Retry logic for external services (Redis, Qdrant).
"""

import time
import asyncio
import logging
import functools
import random
from typing import Callable, Any


logger = logging.getLogger(__name__)

class CircuitBreakerOpenException(Exception):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exceptions: tuple = (Exception,)):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED" # CLOSED, OPEN, HALF-OPEN

    def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
            else:
                raise CircuitBreakerOpenException(f"Circuit Open (last failure {time.time() - self.last_failure_time:.1f}s ago)")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except self.expected_exceptions as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise e

    async def call_async(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
            else:
                raise CircuitBreakerOpenException(f"Circuit Open (last failure {time.time() - self.last_failure_time:.1f}s ago)")

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except self.expected_exceptions as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise e


def circuit_breaker(curr_breaker: CircuitBreaker):
    """Decorator for circuit breaker."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return curr_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def async_circuit_breaker(curr_breaker: CircuitBreaker):
    """Decorator for async circuit breaker."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await curr_breaker.call_async(func, *args, **kwargs)
        return wrapper
    return decorator


def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """Simple exponential backoff retry decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x +
                             random.uniform(0, 1))
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator
