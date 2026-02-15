"""
Observability Metrics (Phase 3.5.4)
===================================
Central definition of Prometheus metrics and utility decorators.
"""

import time
import functools
from prometheus_client import Counter, Histogram, Gauge

# --- Metrics Definitions ---
# API
API_REQUEST_COUNT = Counter(
    "haim_api_request_count", 
    "Total API requests", 
    ["method", "endpoint", "status"]
)
API_REQUEST_LATENCY = Histogram(
    "haim_api_request_latency_seconds", 
    "API request latency", 
    ["method", "endpoint"]
)

# Engine
ENGINE_MEMORY_COUNT = Gauge(
    "haim_engine_memory_total", 
    "Total memories in the system", 
    ["tier"]
)
ENGINE_STORE_LATENCY = Histogram(
    "haim_engine_store_seconds", 
    "Time taken to store memory"
)
ENGINE_QUERY_LATENCY = Histogram(
    "haim_engine_query_seconds", 
    "Time taken to query memories"
)

# Storage (Redis/Qdrant)
STORAGE_OPERATION_COUNT = Counter(
    "haim_storage_ops_total", 
    "Storage operations", 
    ["backend", "operation", "status"]
)
STORAGE_LATENCY = Histogram(
    "haim_storage_latency_seconds", 
    "Storage operation latency", 
    ["backend", "operation"]
)

# Bus
BUS_EVENTS_PUBLISHED = Counter(
    "haim_bus_events_published", 
    "Events published to bus", 
    ["type"]
)
BUS_EVENTS_CONSUMED = Counter(
    "haim_bus_events_consumed", 
    "Events consumed from bus", 
    ["consumer", "type"]
)


# --- Decorators ---

def track_latency(metric: Histogram, labels: dict = None):
    """Decorator to track function execution time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator


def track_async_latency(metric: Histogram, labels: dict = None):
    """Decorator to track async function execution time."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator
