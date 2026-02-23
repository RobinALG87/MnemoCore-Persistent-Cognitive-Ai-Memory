"""
MnemoCore Domain-Specific Exceptions
=====================================

This module defines a hierarchy of exceptions for consistent error handling
across the MnemoCore system.

Exception Hierarchy:
    MnemoCoreError (base)
    ├── RecoverableError (transient, retry possible)
    │   ├── StorageConnectionError
    │   ├── StorageTimeoutError
    │   └── CircuitOpenError
    ├── IrrecoverableError (permanent, requires intervention)
    │   ├── ConfigurationError
    │   ├── DataCorruptionError
    │   ├── ValidationError
    │   ├── NotFoundError
    │   └── UnsupportedOperationError
    └── Domain Errors (mixed recoverability)
        ├── StorageError
        ├── VectorError
        │   ├── DimensionMismatchError
        │   └── VectorOperationError
        └── MemoryOperationError

Usage Guidelines:
    - Return None for "not found" scenarios (expected case, not an error)
    - Raise exceptions for actual errors (connection failures, validation, corruption)
    - Always include context in error messages
    - Use error_code for API responses
"""

from typing import Optional, Any
from enum import Enum
import os


class ErrorCategory(Enum):
    """Categories for error classification."""
    STORAGE = "STORAGE"
    VECTOR = "VECTOR"
    CONFIG = "CONFIG"
    VALIDATION = "VALIDATION"
    MEMORY = "MEMORY"
    AGENT = "AGENT"
    PROVIDER = "PROVIDER"
    SYSTEM = "SYSTEM"


class MnemoCoreError(Exception):
    """
    Base exception for all MnemoCore errors.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code for API responses
        context: Additional context about the error
        recoverable: Whether the error is potentially recoverable
    """

    error_code: str = "MNEMO_CORE_ERROR"
    recoverable: bool = True
    category: ErrorCategory = ErrorCategory.SYSTEM

    def __init__(
        self,
        message: str,
        context: Optional[dict] = None,
        error_code: Optional[str] = None,
        recoverable: Optional[bool] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        if error_code is not None:
            self.error_code = error_code
        if recoverable is not None:
            self.recoverable = recoverable

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message

    def to_dict(self, include_traceback: bool = False) -> dict:
        """
        Convert exception to dictionary for JSON response.

        Args:
            include_traceback: Whether to include stack trace (only in DEBUG mode)
        """
        result = {
            "error": self.message,
            "code": self.error_code,
            "recoverable": self.recoverable,
        }

        if include_traceback:
            import traceback
            result["traceback"] = traceback.format_exc()

        if self.context:
            result["context"] = self.context

        return result


# =============================================================================
# Base Categories: Recoverable vs Irrecoverable
# =============================================================================

class RecoverableError(MnemoCoreError):
    """
    Base class for recoverable errors.

    These are transient errors that may succeed on retry:
    - Connection failures
    - Timeouts
    - Circuit breaker open
    - Rate limiting
    """
    recoverable = True


class IrrecoverableError(MnemoCoreError):
    """
    Base class for irrecoverable errors.

    These are permanent errors that require intervention:
    - Invalid configuration
    - Data corruption
    - Validation failures
    - Resource not found
    """
    recoverable = False


# =============================================================================
# Storage Errors
# =============================================================================

class StorageError(MnemoCoreError):
    """Base exception for storage-related errors."""
    error_code = "STORAGE_ERROR"
    category = ErrorCategory.STORAGE


class StorageConnectionError(RecoverableError, StorageError):
    """Raised when connection to storage backend fails."""
    error_code = "STORAGE_CONNECTION_ERROR"

    def __init__(self, backend: str, message: str = "Connection failed", context: Optional[dict] = None):
        ctx = {"backend": backend}
        if context:
            ctx.update(context)
        super().__init__(f"[{backend}] {message}", ctx)
        self.backend = backend


class StorageTimeoutError(RecoverableError, StorageError):
    """Raised when a storage operation times out."""
    error_code = "STORAGE_TIMEOUT_ERROR"

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


class DataCorruptionError(IrrecoverableError, StorageError):
    """Raised when stored data is corrupt or cannot be deserialized."""
    error_code = "DATA_CORRUPTION_ERROR"

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
    error_code = "VECTOR_ERROR"
    category = ErrorCategory.VECTOR


class DimensionMismatchError(IrrecoverableError, VectorError):
    """Raised when vector dimensions do not match."""
    error_code = "DIMENSION_MISMATCH_ERROR"

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


class VectorOperationError(IrrecoverableError, VectorError):
    """Raised when a vector operation fails."""
    error_code = "VECTOR_OPERATION_ERROR"

    def __init__(self, operation: str, reason: str, context: Optional[dict] = None):
        ctx = {"operation": operation}
        if context:
            ctx.update(context)
        super().__init__(f"Vector operation '{operation}' failed: {reason}", ctx)
        self.operation = operation


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(IrrecoverableError):
    """Raised when configuration is invalid or missing."""
    error_code = "CONFIGURATION_ERROR"
    category = ErrorCategory.CONFIG

    def __init__(self, config_key: str, reason: str, context: Optional[dict] = None):
        ctx = {"config_key": config_key}
        if context:
            ctx.update(context)
        super().__init__(f"Configuration error for '{config_key}': {reason}", ctx)
        self.config_key = config_key


# =============================================================================
# Circuit Breaker Errors
# =============================================================================

class CircuitOpenError(RecoverableError):
    """Raised when a circuit breaker is open and blocking requests."""
    error_code = "CIRCUIT_OPEN_ERROR"
    category = ErrorCategory.SYSTEM

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
    error_code = "MEMORY_OPERATION_ERROR"
    category = ErrorCategory.MEMORY

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
# Validation Errors
# =============================================================================

class ValidationError(IrrecoverableError):
    """Raised when input validation fails."""
    error_code = "VALIDATION_ERROR"
    category = ErrorCategory.VALIDATION

    def __init__(self, field: str, reason: str, value: Any = None, context: Optional[dict] = None):
        ctx = {"field": field}
        if value is not None:
            # Truncate large values
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            ctx["value"] = value_str
        if context:
            ctx.update(context)
        super().__init__(f"Validation error for '{field}': {reason}", ctx)
        self.field = field
        self.reason = reason


class MetadataValidationError(ValidationError):
    """Raised when metadata validation fails."""
    error_code = "METADATA_VALIDATION_ERROR"


class AttributeValidationError(ValidationError):
    """Raised when attribute validation fails."""
    error_code = "ATTRIBUTE_VALIDATION_ERROR"


# =============================================================================
# Not Found Errors
# =============================================================================

class NotFoundError(IrrecoverableError):
    """Raised when a requested resource is not found."""
    error_code = "NOT_FOUND_ERROR"
    category = ErrorCategory.SYSTEM

    def __init__(self, resource_type: str, resource_id: str, context: Optional[dict] = None):
        ctx = {"resource_type": resource_type, "resource_id": resource_id}
        if context:
            ctx.update(context)
        super().__init__(f"{resource_type} '{resource_id}' not found", ctx)
        self.resource_type = resource_type
        self.resource_id = resource_id


class AgentNotFoundError(NotFoundError):
    """Raised when an agent is not found."""
    error_code = "AGENT_NOT_FOUND_ERROR"
    category = ErrorCategory.AGENT

    def __init__(self, agent_id: str, context: Optional[dict] = None):
        super().__init__("Agent", agent_id, context)
        self.agent_id = agent_id


class MemoryNotFoundError(NotFoundError):
    """Raised when a memory is not found."""
    error_code = "MEMORY_NOT_FOUND_ERROR"
    category = ErrorCategory.MEMORY

    def __init__(self, memory_id: str, context: Optional[dict] = None):
        super().__init__("Memory", memory_id, context)
        self.memory_id = memory_id


# =============================================================================
# Provider Errors
# =============================================================================

class ProviderError(MnemoCoreError):
    """Base exception for provider-related errors."""
    error_code = "PROVIDER_ERROR"
    category = ErrorCategory.PROVIDER


class UnsupportedProviderError(IrrecoverableError, ProviderError):
    """Raised when an unsupported provider is requested."""
    error_code = "UNSUPPORTED_PROVIDER_ERROR"

    def __init__(self, provider: str, supported_providers: Optional[list] = None, context: Optional[dict] = None):
        ctx = {"provider": provider}
        if supported_providers:
            ctx["supported_providers"] = supported_providers
        if context:
            ctx.update(context)
        msg = f"Unsupported provider: {provider}"
        if supported_providers:
            msg += f". Supported: {', '.join(supported_providers)}"
        super().__init__(msg, ctx)
        self.provider = provider


class UnsupportedTransportError(IrrecoverableError, ValueError):
    """Raised when an unsupported transport is requested."""
    error_code = "UNSUPPORTED_TRANSPORT_ERROR"
    category = ErrorCategory.CONFIG

    def __init__(self, transport: str, supported_transports: Optional[list] = None, context: Optional[dict] = None):
        ctx = {"transport": transport}
        if supported_transports:
            ctx["supported_transports"] = supported_transports
        if context:
            ctx.update(context)
        msg = f"Unsupported transport: {transport}"
        if supported_transports:
            msg += f". Supported: {', '.join(supported_transports)}"
        super().__init__(msg, ctx)
        self.transport = transport


class DependencyMissingError(IrrecoverableError):
    """Raised when a required dependency is missing."""
    error_code = "DEPENDENCY_MISSING_ERROR"
    category = ErrorCategory.SYSTEM

    def __init__(self, dependency: str, message: str = "", context: Optional[dict] = None):
        ctx = {"dependency": dependency}
        if context:
            ctx.update(context)
        msg = f"Missing dependency: {dependency}"
        if message:
            msg += f". {message}"
        super().__init__(msg, ctx)
        self.dependency = dependency


# =============================================================================
# Backup & Import/Export Errors
# =============================================================================


class BackupError(StorageError):
    """Base exception for backup-related errors."""
    error_code = "BACKUP_ERROR"

    def __init__(self, operation: str, message: str = "", context: Optional[dict] = None):
        ctx = {"operation": operation}
        if context:
            ctx.update(context)
        msg = f"Backup {operation} failed"
        if message:
            msg += f": {message}"
        super().__init__(msg, ctx)
        self.operation = operation


class SnapshotError(BackupError):
    """Raised when snapshot creation or restoration fails."""
    error_code = "SNAPSHOT_ERROR"

    def __init__(self, snapshot_id: str, reason: str, context: Optional[dict] = None):
        ctx = {"snapshot_id": snapshot_id}
        if context:
            ctx.update(context)
        super().__init__("snapshot", f"{reason} for snapshot '{snapshot_id}'", ctx)
        self.snapshot_id = snapshot_id


class SnapshotCorruptionError(IrrecoverableError, StorageError):
    """Raised when a snapshot is corrupted or invalid."""
    error_code = "SNAPSHOT_CORRUPTION_ERROR"
    category = ErrorCategory.STORAGE

    def __init__(self, snapshot_id: str, reason: str = "Snapshot verification failed", context: Optional[dict] = None):
        ctx = {"snapshot_id": snapshot_id}
        if context:
            ctx.update(context)
        super().__init__(f"{reason} for snapshot '{snapshot_id}'", ctx)
        self.snapshot_id = snapshot_id


class ImportError(IrrecoverableError):
    """Raised when memory import fails."""
    error_code = "IMPORT_ERROR"
    category = ErrorCategory.STORAGE

    def __init__(self, source: str, reason: str, records_processed: int = 0, context: Optional[dict] = None):
        ctx = {"source": source, "records_processed": records_processed}
        if context:
            ctx.update(context)
        super().__init__(f"Import from '{source}' failed: {reason}", ctx)
        self.source = source
        self.records_processed = records_processed


class ExportError(StorageError):
    """Raised when memory export fails."""
    error_code = "EXPORT_ERROR"

    def __init__(self, collection: str, reason: str, context: Optional[dict] = None):
        ctx = {"collection": collection}
        if context:
            ctx.update(context)
        super().__init__(f"Export from '{collection}' failed: {reason}", ctx)
        self.collection = collection


class DeduplicationError(ImportError):
    """Raised when deduplication fails during import."""
    error_code = "DEDUPLICATION_ERROR"

    def __init__(self, source: str, duplicate_id: str, context: Optional[dict] = None):
        ctx = {"duplicate_id": duplicate_id}
        if context:
            ctx.update(context)
        super().__init__(source, f"Duplicate ID detected: '{duplicate_id}'", 0, ctx)
        self.duplicate_id = duplicate_id


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


def is_debug_mode() -> bool:
    """Check if debug mode is enabled via environment variable."""
    return os.environ.get("MNEMO_DEBUG", "").lower() in ("true", "1", "yes")


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    # Base
    "MnemoCoreError",
    "RecoverableError",
    "IrrecoverableError",
    "ErrorCategory",
    # Storage
    "StorageError",
    "StorageConnectionError",
    "StorageTimeoutError",
    "DataCorruptionError",
    # Vector
    "VectorError",
    "DimensionMismatchError",
    "VectorOperationError",
    # Config
    "ConfigurationError",
    # Circuit Breaker
    "CircuitOpenError",
    # Memory
    "MemoryOperationError",
    # Validation
    "ValidationError",
    "MetadataValidationError",
    "AttributeValidationError",
    # Not Found
    "NotFoundError",
    "AgentNotFoundError",
    "MemoryNotFoundError",
    # Provider
    "ProviderError",
    "UnsupportedProviderError",
    "UnsupportedTransportError",
    "DependencyMissingError",
    # Backup & Import/Export
    "BackupError",
    "SnapshotError",
    "SnapshotCorruptionError",
    "ImportError",
    "ExportError",
    "DeduplicationError",
    # Utilities
    "wrap_storage_exception",
    "is_debug_mode",
]
