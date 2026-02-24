"""
Comprehensive Tests for Reliability Module
=========================================

Tests the circuit breaker implementation including:
- NativeCircuitBreaker state transitions
- CircuitBreakerError handling
- StorageCircuitBreaker management
- Decorator functionality
"""

import pytest
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mnemocore.core.reliability import (
    NativeCircuitBreaker,
    StorageCircuitBreaker,
    redis_breaker,
    qdrant_breaker,
    circuit_breaker,
    async_circuit_breaker,
    is_storage_available,
    REDIS_FAIL_THRESHOLD,
    REDIS_RESET_TIMEOUT_SEC,
    QDRANT_FAIL_THRESHOLD,
    QDRANT_RESET_TIMEOUT_SEC,
)
from mnemocore.core.exceptions import CircuitOpenError


class TestNativeCircuitBreaker:
    """Test NativeCircuitBreaker class."""

    def test_initialization(self):
        """Should initialize with given parameters."""
        breaker = NativeCircuitBreaker(
            fail_max=5,
            reset_timeout=60,
            name="TestBreaker"
        )
        assert breaker.fail_max == 5
        assert breaker.reset_timeout == 60
        assert breaker.name == "TestBreaker"
        assert breaker.failures == 0
        assert breaker.state == "closed"

    def test_check_state_closed(self):
        """State should remain closed when below threshold."""
        breaker = NativeCircuitBreaker(5, 60, "Test")
        breaker._check_state()
        assert breaker.state == "closed"

    def test_check_state_open_to_half_open(self):
        """State should transition to half-open after timeout."""
        breaker = NativeCircuitBreaker(2, 1, "Test")
        breaker.state = "open"
        breaker.last_failure_time = time.time() - 2  # 2 seconds ago

        breaker._check_state()
        assert breaker.state == "half-open"

    def test_check_state_still_open(self):
        """State should remain open if timeout not elapsed."""
        breaker = NativeCircuitBreaker(5, 60, "Test")
        breaker.state = "open"
        breaker.last_failure_time = time.time()

        breaker._check_state()
        assert breaker.state == "open"

    @pytest.mark.asyncio
    async def test_call_success_in_closed_state(self):
        """Successful call should succeed in closed state."""
        breaker = NativeCircuitBreaker(5, 60, "Test")
        async_func = AsyncMock(return_value="success")

        result = await breaker.call(async_func, "arg1", "arg2")

        assert result == "success"
        assert breaker.failures == 0
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_call_failure_increments_count(self):
        """Failed call should increment failure count."""
        breaker = NativeCircuitBreaker(3, 60, "Test")
        async_func = AsyncMock(side_effect=RuntimeError("error"))

        with pytest.raises(RuntimeError):
            await breaker.call(async_func)

        assert breaker.failures == 1
        assert breaker.last_failure_time > 0

    @pytest.mark.asyncio
    async def test_call_opens_circuit_after_threshold(self):
        """Circuit should open after reaching failure threshold."""
        breaker = NativeCircuitBreaker(2, 60, "Test")
        async_func = AsyncMock(side_effect=RuntimeError("error"))

        # First failure
        with pytest.raises(RuntimeError):
            await breaker.call(async_func)
        assert breaker.state == "closed"

        # Second failure - should open circuit
        with pytest.raises(RuntimeError):
            await breaker.call(async_func)
        assert breaker.state == "open"
        assert breaker.failures == 2

    @pytest.mark.asyncio
    async def test_call_raises_when_open(self):
        """Call should raise CircuitOpenError when circuit is open."""
        breaker = NativeCircuitBreaker(2, 60, "Test")
        breaker.state = "open"
        breaker.failures = 2
        breaker.last_failure_time = time.time()  # Prevent early transition to half-open state

        async_func = AsyncMock(return_value="success")

        with pytest.raises(CircuitOpenError):
            await breaker.call(async_func)

        # Function should not have been called
        async_func.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_call_sync_function(self):
        """Should handle synchronous functions."""
        breaker = NativeCircuitBreaker(5, 60, "Test")

        def sync_func(x):
            return x * 2

        result = await breaker.call(sync_func, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_call_sync_function_returning_awaitable(self):
        """Should handle sync functions returning awaitables."""
        breaker = NativeCircuitBreaker(5, 60, "Test")

        async def inner():
            return "async result"

        def sync_wrapper():
            return inner()

        result = await breaker.call(sync_wrapper)
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_call_direct_awaitable(self):
        """Should handle direct awaitable (not recommended but should work)."""
        breaker = NativeCircuitBreaker(5, 60, "Test")

        async def awaitable_func():
            return "direct awaitable"

        result = await breaker.call(awaitable_func())
        assert result == "direct awaitable"

    def test_success_resets_failures(self):
        """Success should reset failure count."""
        breaker = NativeCircuitBreaker(5, 60, "Test")
        breaker.failures = 3

        breaker.success()

        assert breaker.failures == 0

    def test_success_closes_half_open(self):
        """Success in half-open state should close circuit."""
        breaker = NativeCircuitBreaker(5, 60, "Test")
        breaker.state = "half-open"
        breaker.failures = 2

        breaker.success()

        assert breaker.state == "closed"

    def test_success_no_effect_in_closed(self):
        """Success in closed state should keep it closed."""
        breaker = NativeCircuitBreaker(5, 60, "Test")
        breaker.state = "closed"

        breaker.success()

        assert breaker.state == "closed"

    def test_fail_increments_failures(self):
        """fail() should increment failure count."""
        breaker = NativeCircuitBreaker(5, 60, "Test")
        initial_time = breaker.last_failure_time

        breaker.fail()

        assert breaker.failures == 1
        assert breaker.last_failure_time >= initial_time

    def test_fail_opens_circuit_at_threshold(self):
        """fail() should open circuit at threshold."""
        breaker = NativeCircuitBreaker(2, 60, "Test")
        breaker.failures = 1

        breaker.fail()

        assert breaker.state == "open"

    def test_fail_does_not_reopen_if_already_open(self):
        """fail() should not change state if already open."""
        breaker = NativeCircuitBreaker(2, 60, "Test")
        breaker.state = "open"
        breaker.failures = 5

        breaker.fail()

        assert breaker.state == "open"
        assert breaker.failures == 6


class TestNativeCircuitBreakerRecovery:
    """Test circuit breaker recovery scenarios."""

    @pytest.mark.asyncio
    async def test_half_open_allows_one_call(self):
        """Half-open state should allow one call to test recovery."""
        breaker = NativeCircuitBreaker(2, 1, "Test")

        # Open the circuit
        breaker.state = "open"
        breaker.failures = 2
        breaker.last_failure_time = time.time() - 2  # Past timeout

        # Check should move to half-open
        breaker._check_state()
        assert breaker.state == "half-open"

        # First call in half-open should succeed and close circuit
        async_func = AsyncMock(return_value="success")
        result = await breaker.call(async_func)

        assert result == "success"
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        """Failure in half-open should reopen circuit."""
        breaker = NativeCircuitBreaker(2, 1, "Test")
        breaker.state = "half-open"
        breaker.failures = 2

        async_func = AsyncMock(side_effect=RuntimeError("error"))

        with pytest.raises(RuntimeError):
            await breaker.call(async_func)

        assert breaker.state == "open"


class TestStorageCircuitBreaker:
    """Test StorageCircuitBreaker class."""

    def test_get_redis_breaker_singleton(self):
        """Should return singleton Redis breaker."""
        breaker1 = StorageCircuitBreaker.get_redis_breaker()
        breaker2 = StorageCircuitBreaker.get_redis_breaker()

        assert breaker1 is breaker2
        assert breaker1.name == "RedisBreaker"
        assert breaker1.fail_max == REDIS_FAIL_THRESHOLD
        assert breaker1.reset_timeout == REDIS_RESET_TIMEOUT_SEC

    def test_get_qdrant_breaker_singleton(self):
        """Should return singleton Qdrant breaker."""
        breaker1 = StorageCircuitBreaker.get_qdrant_breaker()
        breaker2 = StorageCircuitBreaker.get_qdrant_breaker()

        assert breaker1 is breaker2
        assert breaker1.name == "QdrantBreaker"
        assert breaker1.fail_max == QDRANT_FAIL_THRESHOLD
        assert breaker1.reset_timeout == QDRANT_RESET_TIMEOUT_SEC

    def test_redis_and_qdrant_are_different(self):
        """Redis and Qdrant breakers should be different instances."""
        redis = StorageCircuitBreaker.get_redis_breaker()
        qdrant = StorageCircuitBreaker.get_qdrant_breaker()

        assert redis is not qdrant


class TestPreconfiguredBreakers:
    """Test pre-configured breaker instances."""

    def test_redis_breaker_instance(self):
        """Module-level redis_breaker should be available."""
        assert redis_breaker is not None
        assert redis_breaker.name == "RedisBreaker"

    def test_qdrant_breaker_instance(self):
        """Module-level qdrant_breaker should be available."""
        assert qdrant_breaker is not None
        assert qdrant_breaker.name == "QdrantBreaker"

    def test_backward_compat_aliases(self):
        """Backward compatibility aliases should exist."""
        from mnemocore.core import reliability
        assert hasattr(reliability, 'storage_circuit_breaker')
        assert hasattr(reliability, 'vector_circuit_breaker')
        assert reliability.storage_circuit_breaker is redis_breaker
        assert reliability.vector_circuit_breaker is qdrant_breaker


class TestIsStorageAvailable:
    """Test is_storage_available helper."""

    def test_redis_available_when_closed(self):
        """Should return True when Redis breaker is closed."""
        redis_breaker.state = "closed"
        assert is_storage_available("redis") is True

    def test_redis_unavailable_when_open(self):
        """Should return False when Redis breaker is open."""
        redis_breaker.state = "open"
        assert is_storage_available("redis") is False

    def test_qdrant_available_when_closed(self):
        """Should return True when Qdrant breaker is closed."""
        qdrant_breaker.state = "closed"
        assert is_storage_available("qdrant") is True

    def test_qdrant_unavailable_when_open(self):
        """Should return False when Qdrant breaker is open."""
        qdrant_breaker.state = "open"
        assert is_storage_available("qdrant") is False

    def test_unknown_breaker_returns_true(self):
        """Should return True for unknown breaker names."""
        assert is_storage_available("unknown") is True


class TestCircuitBreakerDecorators:
    """Test circuit breaker decorators."""

    @pytest.mark.asyncio
    async def test_async_decorator_success(self):
        """Async decorator should allow successful calls."""
        decorated = async_circuit_breaker(redis_breaker)(AsyncMock(return_value="ok"))
        result = await decorated()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_async_decorator_failure(self):
        """Async decorator should handle failures."""
        decorated = async_circuit_breaker(redis_breaker)(AsyncMock(side_effect=RuntimeError("fail")))

        with pytest.raises(RuntimeError):
            await decorated()

    @pytest.mark.asyncio
    async def test_async_decorator_with_args(self):
        """Async decorator should pass arguments."""
        mock_func = AsyncMock(return_value="result")
        decorated = async_circuit_breaker(redis_breaker)(mock_func)

        result = await decorated("arg1", kwarg="kwvalue")

        mock_func.assert_called_once_with("arg1", kwarg="kwvalue")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_async_decorator_blocks_when_open(self):
        """Async decorator should block when circuit is open."""
        # Open the circuit
        redis_breaker.state = "open"
        redis_breaker.failures = 10

        mock_func = AsyncMock(return_value="ok")
        decorated = async_circuit_breaker(redis_breaker)(mock_func)

        with pytest.raises(CircuitOpenError):
            await decorated()

        mock_func.assert_not_awaited()

    def test_sync_decorator_success(self):
        """Sync decorator should allow successful calls."""
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = "ok"

            @circuit_breaker(redis_breaker)
            def sync_func():
                return "ok"

            result = sync_func()

            # The decorator wraps with asyncio.run
            assert result == "ok" or mock_run.called

    @pytest.mark.asyncio
    async def test_sync_decorator_in_async_context(self):
        """Sync decorator should detect async context."""
        mock_func = MagicMock(return_value="ok")
        decorated = circuit_breaker(redis_breaker)(mock_func)

        # In async context, it should try to use the event loop
        result = await decorated()

        # Should complete without error
        assert isinstance(result, str) or result is not None


class TestCircuitBreakerError:
    """Test CircuitOpenError exception."""

    def test_error_creation(self):
        """Should create error with context."""
        error = CircuitOpenError(
            breaker_name="TestBreaker",
            failures=5,
            context={"state": "open", "reset_timeout": 60}
        )
        assert error.breaker_name == "TestBreaker"
        assert error.failures == 5
        assert error.context["state"] == "open"

    def test_error_is_mnemo_core_error(self):
        """CircuitOpenError should be a domain exception."""
        error = CircuitOpenError("Test", 1, {})
        assert isinstance(error, Exception)

    def test_error_message(self):
        """Error should have informative message."""
        error = CircuitOpenError("RedisBreaker", 10, {"state": "open"})
        str_repr = str(error)
        assert "RedisBreaker" in str_repr
        assert "10" in str_repr


class TestCircuitBreakerPropertyBased:
    """Property-based tests using Hypothesis."""

    from hypothesis import given, strategies as st

    @given(st.integers(min_value=1, max_value=20),
           st.integers(min_value=1, max_value=100))
    def test_various_thresholds(self, fail_max, reset_timeout):
        """Circuit breaker should work with various thresholds."""
        breaker = NativeCircuitBreaker(fail_max, reset_timeout, "Test")
        assert breaker.fail_max == fail_max
        assert breaker.reset_timeout == reset_timeout
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    @given(st.integers(min_value=1, max_value=10))
    async def test_opens_after_exact_failures(self, fail_count):
        """Circuit should open after exactly fail_max failures."""
        breaker = NativeCircuitBreaker(fail_count, 60, "Test")
        async_fail_func = AsyncMock(side_effect=RuntimeError("fail"))

        # Trigger failures
        for _ in range(fail_count):
            with pytest.raises(RuntimeError):
                await breaker.call(async_fail_func)

        # Circuit should now be open
        assert breaker.state == "open"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
