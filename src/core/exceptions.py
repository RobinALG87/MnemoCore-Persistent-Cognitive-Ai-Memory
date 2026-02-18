"""
MnemoCore Domain-Specific Exceptions
=====================================

This module defines a hierarchy of exceptions for consistent error handling
across the MnemoCore system.

Exception Hierarchy:
    MnemoCoreError (base)
    ├── StorageError
    │   ├── StorageConnectionError
    │   ├── StorageTimeoutError
    │   └── DataCorruptionError
    ├── VectorError
    │   ├── DimensionMismatchError
    │   └── VectorOperationError
    ├── ConfigurationError
    ├── CircuitOpenError
    └── MemoryOperationError

Usage Guidelines:
    - Return None for "not found" scenarios (expected case, not an error)
    - Raise exceptions for actual errors (connection failures, validation, corruption)
    - Always include context in error messages
"""

from typing import Optional, Any


class MnemoCoreError(Exception):
    """Base exception for all MnemoCore errors."""

    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message


# =============================================================================
# Storage Errors
# =============================================================================

class StorageError(MnemoCoreError):
    """Base exception for storage-related errors."""
    pass


class StorageConnectionError(StorageError):
    """Raised when connection to storage backend fails."""

    def __init__(self, backend: str, message: str = "Connection failed", context: Optional[dict] = None):
        ctx = {"backend": backend}
        if context:
            ctx.update(context)
        super().__init__(f"[{backend}] {message}", ctx)
        self.backend = backend


class StorageTimeoutError(StorageError):
    """Raised when a storage operation times out."""

    def __init__(self, backend: str, operation: str, timeout_ms: Optional[int] = None, context: Optional[dict] = None):
        msg = f"[{backend}] Operation '{operation}' timed out"
        ctx = {"backend": backend, "operation": operation}
        if timeout_ms is not None:
            ctx["timeout_ms"] = timeout_ms
        if context:
            ctx.update(context)
        super().__init__(msg, ctx)
        self.backend = backend
        self.operation = operation


class DataCorruptionError(StorageError):
    """Raised when stored data is corrupt or cannot be deserialized."""

    def __init__(self, resource_id: str, reason: str = "Data corruption detected", context: Optional[dict] = None):
        ctx = {"resource_id": resource_id}
        if context:
            ctx.update(context)
        super().__init__(f"{reason} for resource '{resource_id}'", ctx)
        self.resource_id = resource_id


# =============================================================================
# Vector Errors
# =============================================================================

class VectorError(MnemoCoreError):
    """Base exception for vector/hyperdimensional operations."""
    pass


class DimensionMismatchError(VectorError):
    """Raised when vector dimensions do not match."""

    def __init__(self, expected: int, actual: int, operation: str = "operation", context: Optional[dict] = None):
        ctx = {"expected": expected, "actual": actual, "operation": operation}
        if context:
            ctx.update(context)
        super().__init__(
            f"Dimension mismatch in {operation}: expected {expected}, got {actual}",
            ctx
        )
        self.expected = expected
        self.actual = actual
        self.operation = operation


class VectorOperationError(VectorError):
    """Raised when a vector operation fails."""

    def __init__(self, operation: str, reason: str, context: Optional[dict] = None):
        ctx = {"operation": operation}
        if context:
            ctx.update(context)
        super().__init__(f"Vector operation '{operation}' failed: {reason}", ctx)
        self.operation = operation


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(MnemoCoreError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, config_key: str, reason: str, context: Optional[dict] = None):
        ctx = {"config_key": config_key}
        if context:
            ctx.update(context)
        super().__init__(f"Configuration error for '{config_key}': {reason}", ctx)
        self.config_key = config_key


# =============================================================================
# Circuit Breaker Errors
# =============================================================================

class CircuitOpenError(MnemoCoreError):
    """Raised when a circuit breaker is open and blocking requests."""

    def __init__(self, breaker_name: str, failures: int, context: Optional[dict] = None):
        ctx = {"breaker_name": breaker_name, "failures": failures}
        if context:
            ctx.update(context)
        super().__init__(
            f"Circuit breaker '{breaker_name}' is OPEN after {failures} failures",
            ctx
        )
        self.breaker_name = breaker_name
        self.failures = failures


# =============================================================================
# Memory Operation Errors
# =============================================================================

class MemoryOperationError(MnemoCoreError):
    """Raised when a memory operation (store, retrieve, delete) fails."""

    def __init__(self, operation: str, node_id: Optional[str], reason: str, context: Optional[dict] = None):
        ctx = {"operation": operation}
        if node_id:
            ctx["node_id"] = node_id
        if context:
            ctx.update(context)
        super().__init__(f"Memory {operation} failed for '{node_id}': {reason}", ctx)
        self.operation = operation
        self.node_id = node_id


# =============================================================================
# Utility Functions
# =============================================================================

def wrap_storage_exception(backend: str, operation: str, exc: Exception) -> StorageError:
    """
    Wrap a generic exception into an appropriate StorageError.

    Args:
        backend: Name of the storage backend (e.g., 'redis', 'qdrant')
        operation: Name of the operation that failed
        exc: The original exception

    Returns:
        An appropriate StorageError subclass
    """
    exc_name = type(exc).__name__
    exc_msg = str(exc)

    # Timeout detection
    if 'timeout' in exc_msg.lower() or 'Timeout' in exc_name:
        return StorageTimeoutError(backend, operation)

    # Connection error detection
    if any(x in exc_name.lower() for x in ['connection', 'connect', 'network']):
        return StorageConnectionError(backend, exc_msg)

    # Default to generic storage error
    return StorageError(
        f"[{backend}] {operation} failed: {exc_msg}",
        {"backend": backend, "operation": operation, "original_exception": exc_name}
    )
