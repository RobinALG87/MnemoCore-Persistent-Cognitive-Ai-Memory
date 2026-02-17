import logging
import time
from typing import Optional, Callable, Any
from .config import get_config

logger = logging.getLogger(__name__)

# Constants for circuit breaker thresholds
REDIS_FAIL_THRESHOLD = 5
REDIS_RESET_TIMEOUT_SEC = 60
QDRANT_FAIL_THRESHOLD = 3
QDRANT_RESET_TIMEOUT_SEC = 30

class CircuitBreakerError(Exception):
    """Exception raised when a circuit breaker is open."""
    pass

class NativeCircuitBreaker:
    """Light-weight native implementation of a Circuit Breaker."""
    
    def __init__(self, fail_max: int, reset_timeout: int, name: str):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.name = name
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed" # closed, open, half-open

    def _check_state(self):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                logger.warning(f"Circuit Breaker {self.name} moving to half-open")
                self.state = "half-open"
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        self._check_state()
        
        if self.state == "open":
            raise CircuitBreakerError(f"Circuit Breaker {self.name} is OPEN")
            
        try:
            if hasattr(func, "__call__"):
                # Check if it's already an awaitable or a function returning awaitable
                import inspect
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
            else:
                # Direct awaitable? (not recommended for breaker logic)
                result = await func
                
            self.success()
            return result
        except Exception as e:
            self.fail()
            raise e

    def success(self):
        if self.state == "half-open":
            logger.info(f"Circuit Breaker {self.name} back to CLOSED")
            self.state = "closed"
        self.failures = 0

    def fail(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.fail_max:
            if self.state != "open":
                logger.critical(f"Circuit Breaker {self.name} OPENED after {self.failures} failures")
                self.state = "open"

class StorageCircuitBreaker:
    """Centralized management for native storage circuit breakers."""
    
    _redis_breaker = None
    _qdrant_breaker = None

    @classmethod
    def get_redis_breaker(cls):
        if cls._redis_breaker is None:
            cls._redis_breaker = NativeCircuitBreaker(
                fail_max=REDIS_FAIL_THRESHOLD,
                reset_timeout=REDIS_RESET_TIMEOUT_SEC,
                name="RedisBreaker"
            )
        return cls._redis_breaker

    @classmethod
    def get_qdrant_breaker(cls):
        if cls._qdrant_breaker is None:
            cls._qdrant_breaker = NativeCircuitBreaker(
                fail_max=QDRANT_FAIL_THRESHOLD,
                reset_timeout=QDRANT_RESET_TIMEOUT_SEC,
                name="QdrantBreaker"
            )
        return cls._qdrant_breaker

def is_storage_available(breaker_name: str) -> bool:
    """Helper to check if a circuit is currently open."""
    if breaker_name == "redis":
        return StorageCircuitBreaker.get_redis_breaker().state == "closed"
    elif breaker_name == "qdrant":
        return StorageCircuitBreaker.get_qdrant_breaker().state == "closed"
    return True
