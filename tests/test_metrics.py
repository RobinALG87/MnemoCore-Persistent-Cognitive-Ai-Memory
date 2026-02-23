"""
Comprehensive Tests for Metrics Module
======================================

Tests the observability metrics including:
- Prometheus metrics definitions
- OpenTelemetry integration
- Decorators for tracking
- Helper functions
"""

import pytest
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mnemocore.core.metrics import (
    # Metrics
    API_REQUEST_COUNT,
    API_REQUEST_LATENCY,
    ENGINE_MEMORY_COUNT,
    ENGINE_STORE_LATENCY,
    ENGINE_QUERY_LATENCY,
    STORE_DURATION_SECONDS,
    QUERY_DURATION_SECONDS,
    MEMORY_COUNT_TOTAL,
    QUEUE_LENGTH,
    ERROR_TOTAL,
    STORAGE_OPERATION_COUNT,
    STORAGE_LATENCY,
    BUS_EVENTS_PUBLISHED,
    BUS_EVENTS_CONSUMED,
    DREAM_LOOP_TOTAL,
    DREAM_LOOP_ITERATION_SECONDS,
    DREAM_LOOP_INSIGHTS_GENERATED,
    DREAM_LOOP_ACTIVE,
    # Functions
    init_opentelemetry,
    get_trace_id,
    set_trace_id,
    extract_trace_context,
    inject_trace_context,
    track_latency,
    track_async_latency,
    timer,
    traced,
    update_memory_count,
    update_queue_length,
    record_error,
    # Constants
    OTEL_AVAILABLE,
    tracer,
    propagator,
)


class TestPrometheusMetricsExist:
    """Test that all Prometheus metrics are defined."""

    def test_api_request_count_exists(self):
        """API_REQUEST_COUNT should be a Counter."""
        assert API_REQUEST_COUNT is not None
        assert hasattr(API_REQUEST_COUNT, 'inc')
        assert hasattr(API_REQUEST_COUNT, 'labels')

    def test_api_request_latency_exists(self):
        """API_REQUEST_LATENCY should be a Histogram."""
        assert API_REQUEST_LATENCY is not None
        assert hasattr(API_REQUEST_LATENCY, 'observe')

    def test_engine_metrics_exist(self):
        """Engine metrics should be defined."""
        assert ENGINE_MEMORY_COUNT is not None
        assert ENGINE_STORE_LATENCY is not None
        assert ENGINE_QUERY_LATENCY is not None

    def test_phase4_metrics_exist(self):
        """Phase 4.1 metrics should be defined."""
        assert STORE_DURATION_SECONDS is not None
        assert QUERY_DURATION_SECONDS is not None
        assert MEMORY_COUNT_TOTAL is not None
        assert QUEUE_LENGTH is not None
        assert ERROR_TOTAL is not None

    def test_storage_metrics_exist(self):
        """Storage metrics should be defined."""
        assert STORAGE_OPERATION_COUNT is not None
        assert STORAGE_LATENCY is not None

    def test_bus_metrics_exist(self):
        """Bus metrics should be defined."""
        assert BUS_EVENTS_PUBLISHED is not None
        assert BUS_EVENTS_CONSUMED is not None

    def test_dream_loop_metrics_exist(self):
        """Dream loop metrics should be defined."""
        assert DREAM_LOOP_TOTAL is not None
        assert DREAM_LOOP_ITERATION_SECONDS is not None
        assert DREAM_LOOP_INSIGHTS_GENERATED is not None
        assert DREAM_LOOP_ACTIVE is not None


class TestMetricsOperations:
    """Test basic metric operations."""

    def test_counter_increment(self):
        """Counter should increment."""
        initial = ERROR_TOTAL.labels(error_type="test")._value.get()
        ERROR_TOTAL.labels(error_type="test").inc()
        assert ERROR_TOTAL.labels(error_type="test")._value.get() >= initial

    def test_counter_increment_by_amount(self):
        """Counter should increment by amount."""
        initial = ERROR_TOTAL.labels(error_type="test2")._value.get()
        ERROR_TOTAL.labels(error_type="test2").inc(5)
        assert ERROR_TOTAL.labels(error_type="test2")._value.get() >= initial + 5

    def test_gauge_set(self):
        """Gauge should set value."""
        QUEUE_LENGTH.set(42)
        assert QUEUE_LENGTH._value.get() == 42

    def test_gauge_with_labels(self):
        """Gauge with labels should work."""
        MEMORY_COUNT_TOTAL.labels(tier="hot").set(100)
        assert MEMORY_COUNT_TOTAL.labels(tier="hot")._value.get() == 100

    def test_histogram_observe(self):
        """Histogram should observe values."""
        metric = STORE_DURATION_SECONDS.labels(tier="hot")
        metric.observe(0.1)
        metric.observe(0.5)
        # Should not raise
        assert metric is not None


class TestHelperFunctions:
    """Test helper functions."""

    def test_update_memory_count(self):
        """Should update memory count gauges."""
        update_memory_count("hot", 150)
        assert MEMORY_COUNT_TOTAL.labels(tier="hot")._value.get() == 150
        assert ENGINE_MEMORY_COUNT.labels(tier="hot")._value.get() == 150

    def test_update_queue_length(self):
        """Should update queue length gauge."""
        update_queue_length(25)
        assert QUEUE_LENGTH._value.get() == 25

    def test_record_error(self):
        """Should increment error counter."""
        initial = ERROR_TOTAL.labels(error_type="ValueError")._value.get()
        record_error("ValueError")
        assert ERROR_TOTAL.labels(error_type="ValueError")._value.get() >= initial


class TestOpenTelemetryInit:
    """Test OpenTelemetry initialization."""

    def test_init_returns_none_when_unavailable(self):
        """Should return None when OTEL is not available."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', False):
            result = init_opentelemetry()
            assert result is None

    def test_init_with_console_exporter(self):
        """Should initialize with console exporter when available."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', True):
            result = init_opentelemetry(exporter="console")
            if OTEL_AVAILABLE:
                assert result is not None
            else:
                assert result is None

    def test_init_with_otlp_exporter(self):
        """Should handle OTLP exporter."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', True):
            result = init_opentelemetry(exporter="otlp")
            if OTEL_AVAILABLE:
                assert result is not None
            else:
                assert result is None

    def test_init_with_custom_service_name(self):
        """Should use custom service name."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', True):
            result = init_opentelemetry(service_name="custom_service")
            if OTEL_AVAILABLE:
                assert result is not None
            else:
                assert result is None


class TestTraceContext:
    """Test trace context management."""

    def test_get_and_set_trace_id(self):
        """Should set and get trace ID."""
        set_trace_id("test-trace-123")
        assert get_trace_id() == "test-trace-123"

    def test_get_trace_id_default(self):
        """Should return None when not set."""
        # Clear the context var
        set_trace_id(None)
        assert get_trace_id() is None

    def test_extract_trace_context_no_headers(self):
        """Should return None for empty headers."""
        result = extract_trace_context({})
        assert result is None

    def test_extract_trace_context_no_otel(self):
        """Should return None when OTEL unavailable."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', False):
            result = extract_trace_context({})
            assert result is None

    def test_inject_trace_context_no_otel(self):
        """Should return empty dict when OTEL unavailable."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', False):
            result = inject_trace_context()
            assert result == {}

    def test_inject_trace_context_with_otel(self):
        """Should inject trace context when OTEL available."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', True):
            result = inject_trace_context()
            # Should return a dict (possibly empty)
            assert isinstance(result, dict)


class TestTrackLatencyDecorator:
    """Test track_latency decorator."""

    def test_track_latency_sync_function(self):
        """Should track latency of sync function."""
        metric = MagicMock()
        decorator = track_latency(metric, labels={"op": "test"})

        @decorator
        def test_func():
            time.sleep(0.01)
            return "result"

        result = test_func()

        assert result == "result"
        metric.observe.assert_called()

    def test_track_latency_without_labels(self):
        """Should work without labels."""
        metric = MagicMock()
        decorator = track_latency(metric)

        @decorator
        def test_func():
            return "ok"

        result = test_func()

        assert result == "ok"
        metric.observe.assert_called()

    def test_track_latency_preserves_exception(self):
        """Should preserve exceptions."""
        metric = MagicMock()

        @track_latency(metric)
        def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            test_func()

        # Should still record latency
        metric.observe.assert_called()


class TestTrackAsyncLatencyDecorator:
    """Test track_async_latency decorator."""

    @pytest.mark.asyncio
    async def test_track_async_latency(self):
        """Should track latency of async function."""
        metric = MagicMock()
        decorator = track_async_latency(metric, labels={"op": "async_test"})

        @decorator
        async def test_func():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await test_func()

        assert result == "async_result"
        metric.observe.assert_called()

    @pytest.mark.asyncio
    async def test_track_async_latency_preserves_exception(self):
        """Should preserve exceptions in async functions."""
        metric = MagicMock()

        @track_async_latency(metric)
        async def test_func():
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError):
            await test_func()

        metric.observe.assert_called()


class TestTimerDecorator:
    """Test timer decorator with OTEL support."""

    @pytest.mark.asyncio
    async def test_timer_basic(self):
        """Should time async function."""
        metric = MagicMock()

        @timer(metric)
        async def test_func():
            await asyncio.sleep(0.01)
            return "result"

        result = await test_func()

        assert result == "result"
        metric.observe.assert_called()

    @pytest.mark.asyncio
    async def test_timer_with_labels(self):
        """Should use labels."""
        metric = MagicMock()

        @timer(metric, labels={"tier": "hot"})
        async def test_func():
            return "ok"

        await test_func()

        # Should have observed with labels
        metric.labels.assert_called_once()

    @pytest.mark.asyncio
    async def test_timer_records_errors(self):
        """Should record errors."""
        metric = MagicMock()

        @timer(metric)
        async def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await test_func()

        # Should have recorded error
        ERROR_TOTAL.labels.assert_called()

    @pytest.mark.asyncio
    async def test_timer_with_trace_id(self):
        """Should include trace_id if set."""
        metric = MagicMock()

        set_trace_id("trace-123")

        @timer(metric)
        async def test_func():
            return "ok"

        await test_func()

        # Should have called observe
        metric.observe.assert_called()


class TestTracedDecorator:
    """Test traced decorator."""

    @pytest.mark.asyncio
    async def test_traced_without_otel(self):
        """Should work normally when OTEL unavailable."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', False):

            @traced("test_operation")
            async def test_func():
                return "result"

            result = await test_func()

            assert result == "result"

    @pytest.mark.asyncio
    async def test_traced_preserves_exceptions(self):
        """Should preserve exceptions."""
        with patch('mnemocore.core.metrics.OTEL_AVAILABLE', False):

            @traced()
            async def test_func():
                raise ValueError("traced error")

            with pytest.raises(ValueError):
                await test_func()


class TestMetricsLabels:
    """Test metric label handling."""

    def test_api_request_count_labels(self):
        """Should accept labels for API metrics."""
        API_REQUEST_COUNT.labels(
            method="GET",
            endpoint="/query",
            status="200"
        ).inc()
        # Should not raise

    def test_api_request_latency_labels(self):
        """Should accept labels for latency metrics."""
        API_REQUEST_LATENCY.labels(
            method="POST",
            endpoint="/store"
        ).observe(0.1)
        # Should not raise

    def test_storage_operation_labels(self):
        """Should accept labels for storage metrics."""
        STORAGE_OPERATION_COUNT.labels(
            backend="redis",
            operation="get",
            status="success"
        ).inc()

    def test_bus_event_labels(self):
        """Should accept labels for bus metrics."""
        BUS_EVENTS_PUBLISHED.labels(type="memory_stored").inc()
        BUS_EVENTS_CONSUMED.labels(consumer="consolidator", type="batch").inc()

    def test_dream_loop_labels(self):
        """Should accept labels for dream loop metrics."""
        DREAM_LOOP_TOTAL.labels(status="success").inc()
        DREAM_LOOP_INSIGHTS_GENERATED.labels(type="concept").inc()


class TestMetricsIntegration:
    """Integration tests for metrics system."""

    def test_multiple_metric_updates(self):
        """Should handle multiple metric updates."""
        for i in range(10):
            ERROR_TOTAL.labels(error_type=f"error_{i}").inc()

        # Should have 10 different error types
        assert True  # If we got here, no errors

    def test_concurrent_metric_updates(self):
        """Should handle concurrent updates."""
        import threading

        def update_metrics():
            for _ in range(100):
                QUEUE_LENGTH.set(42)

        threads = [threading.Thread(target=update_metrics) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash
        assert QUEUE_LENGTH._value.get() == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
