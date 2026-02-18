"""
Tests for MnemoCore Error Handling
===================================
Tests the exception hierarchy, error codes, and FastAPI integration.
"""

import pytest
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemocore.core.exceptions import (
    # Base
    MnemoCoreError,
    RecoverableError,
    IrrecoverableError,
    ErrorCategory,
    # Storage
    StorageError,
    StorageConnectionError,
    StorageTimeoutError,
    DataCorruptionError,
    # Vector
    VectorError,
    DimensionMismatchError,
    VectorOperationError,
    # Config
    ConfigurationError,
    # Circuit Breaker
    CircuitOpenError,
    # Memory
    MemoryOperationError,
    # Validation
    ValidationError,
    MetadataValidationError,
    AttributeValidationError,
    # Not Found
    NotFoundError,
    AgentNotFoundError,
    MemoryNotFoundError,
    # Provider
    ProviderError,
    UnsupportedProviderError,
    UnsupportedTransportError,
    DependencyMissingError,
    # Utilities
    wrap_storage_exception,
    is_debug_mode,
)


class TestExceptionHierarchy:
    """Test the exception inheritance hierarchy."""

    def test_base_exception(self):
        """Test MnemoCoreError base class."""
        exc = MnemoCoreError("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.context == {}
        assert exc.recoverable is True
        assert exc.error_code == "MNEMO_CORE_ERROR"

    def test_exception_with_context(self):
        """Test exception with context."""
        exc = MnemoCoreError("Test error", context={"key": "value"})
        assert exc.context == {"key": "value"}
        assert "context=" in str(exc)

    def test_exception_to_dict(self):
        """Test to_dict conversion."""
        exc = ValidationError(
            field="test_field",
            reason="Invalid value",
            value="bad_data"
        )
        d = exc.to_dict()
        assert d["error"] == "Validation error for 'test_field': Invalid value"
        assert d["code"] == "VALIDATION_ERROR"
        assert d["recoverable"] is False
        assert "traceback" not in d

    def test_exception_to_dict_with_traceback(self):
        """Test to_dict with traceback in debug mode."""
        exc = ValidationError(field="test", reason="test")
        d = exc.to_dict(include_traceback=True)
        assert "traceback" in d


class TestRecoverableErrors:
    """Test recoverable error classes."""

    def test_storage_connection_error_is_recoverable(self):
        """Storage connection errors should be recoverable."""
        exc = StorageConnectionError("redis", "Connection refused")
        assert exc.recoverable is True
        assert exc.error_code == "STORAGE_CONNECTION_ERROR"
        assert exc.backend == "redis"

    def test_storage_timeout_error_is_recoverable(self):
        """Storage timeout errors should be recoverable."""
        exc = StorageTimeoutError("qdrant", "search", timeout_ms=5000)
        assert exc.recoverable is True
        assert exc.error_code == "STORAGE_TIMEOUT_ERROR"
        assert exc.backend == "qdrant"
        assert exc.operation == "search"
        assert exc.context["timeout_ms"] == 5000

    def test_circuit_open_error_is_recoverable(self):
        """Circuit breaker open errors should be recoverable."""
        exc = CircuitOpenError("storage", failures=5)
        assert exc.recoverable is True
        assert exc.error_code == "CIRCUIT_OPEN_ERROR"
        assert exc.breaker_name == "storage"
        assert exc.failures == 5


class TestIrrecoverableErrors:
    """Test irrecoverable error classes."""

    def test_validation_error_is_irrecoverable(self):
        """Validation errors should be irrecoverable."""
        exc = ValidationError(field="content", reason="Cannot be empty")
        assert exc.recoverable is False
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.field == "content"

    def test_configuration_error_is_irrecoverable(self):
        """Configuration errors should be irrecoverable."""
        exc = ConfigurationError("api_key", "Missing required key")
        assert exc.recoverable is False
        assert exc.error_code == "CONFIGURATION_ERROR"
        assert exc.config_key == "api_key"

    def test_data_corruption_error_is_irrecoverable(self):
        """Data corruption errors should be irrecoverable."""
        exc = DataCorruptionError("mem_123", "Invalid checksum")
        assert exc.recoverable is False
        assert exc.error_code == "DATA_CORRUPTION_ERROR"
        assert exc.resource_id == "mem_123"

    def test_not_found_errors_are_irrecoverable(self):
        """Not found errors should be irrecoverable."""
        exc = MemoryNotFoundError("mem_123")
        assert exc.recoverable is False
        assert exc.error_code == "MEMORY_NOT_FOUND_ERROR"

        exc2 = AgentNotFoundError("agent_456")
        assert exc2.recoverable is False
        assert exc2.error_code == "AGENT_NOT_FOUND_ERROR"

    def test_unsupported_provider_error_is_irrecoverable(self):
        """Unsupported provider errors should be irrecoverable."""
        exc = UnsupportedProviderError("unknown", supported_providers=["openai", "anthropic"])
        assert exc.recoverable is False
        assert exc.error_code == "UNSUPPORTED_PROVIDER_ERROR"
        assert exc.provider == "unknown"
        assert "openai" in str(exc)


class TestVectorErrors:
    """Test vector-related errors."""

    def test_dimension_mismatch_error(self):
        """Test dimension mismatch error."""
        exc = DimensionMismatchError(expected=16384, actual=10000, operation="encode")
        assert exc.recoverable is False
        assert exc.error_code == "DIMENSION_MISMATCH_ERROR"
        assert exc.expected == 16384
        assert exc.actual == 10000
        assert "16384" in str(exc)
        assert "10000" in str(exc)

    def test_vector_operation_error(self):
        """Test vector operation error."""
        exc = VectorOperationError("bundle", "NaN detected")
        assert exc.recoverable is False
        assert exc.error_code == "VECTOR_OPERATION_ERROR"
        assert exc.operation == "bundle"


class TestStorageErrorWrapper:
    """Test wrap_storage_exception utility."""

    def test_wrap_timeout_exception(self):
        """Timeout exceptions should be wrapped as StorageTimeoutError."""
        exc = Exception("Connection timeout after 5000ms")
        wrapped = wrap_storage_exception("redis", "get", exc)
        assert isinstance(wrapped, StorageTimeoutError)
        assert wrapped.backend == "redis"
        assert wrapped.operation == "get"

    def test_wrap_connection_exception(self):
        """Connection exceptions should be wrapped as StorageConnectionError."""
        # Create a mock exception with 'Connection' in the class name
        class ConnectionRefusedError(Exception):
            pass
        exc = ConnectionRefusedError("Connection refused")
        wrapped = wrap_storage_exception("qdrant", "search", exc)
        assert isinstance(wrapped, StorageConnectionError)
        assert wrapped.backend == "qdrant"

    def test_wrap_generic_exception(self):
        """Generic exceptions should be wrapped as StorageError."""
        exc = Exception("Unknown error")
        wrapped = wrap_storage_exception("redis", "set", exc)
        assert isinstance(wrapped, StorageError)
        assert "redis" in str(wrapped)
        assert "set" in str(wrapped)


class TestDebugMode:
    """Test debug mode detection."""

    def test_debug_mode_off_by_default(self):
        """Debug mode should be off by default."""
        # Save and clear env
        old_val = os.environ.get("MNEMO_DEBUG")
        if "MNEMO_DEBUG" in os.environ:
            del os.environ["MNEMO_DEBUG"]

        try:
            assert is_debug_mode() is False
        finally:
            if old_val:
                os.environ["MNEMO_DEBUG"] = old_val

    def test_debug_mode_on_with_true(self):
        """Debug mode should be on when set to 'true'."""
        old_val = os.environ.get("MNEMO_DEBUG")
        os.environ["MNEMO_DEBUG"] = "true"

        try:
            assert is_debug_mode() is True
        finally:
            if old_val:
                os.environ["MNEMO_DEBUG"] = old_val
            else:
                del os.environ["MNEMO_DEBUG"]

    def test_debug_mode_on_with_1(self):
        """Debug mode should be on when set to '1'."""
        old_val = os.environ.get("MNEMO_DEBUG")
        os.environ["MNEMO_DEBUG"] = "1"

        try:
            assert is_debug_mode() is True
        finally:
            if old_val:
                os.environ["MNEMO_DEBUG"] = old_val
            else:
                del os.environ["MNEMO_DEBUG"]


class TestErrorCategories:
    """Test error category classification."""

    def test_storage_error_category(self):
        """Storage errors should have STORAGE category."""
        exc = StorageError("test")
        assert exc.category == ErrorCategory.STORAGE

    def test_vector_error_category(self):
        """Vector errors should have VECTOR category."""
        exc = VectorError("test")
        assert exc.category == ErrorCategory.VECTOR

    def test_config_error_category(self):
        """Config errors should have CONFIG category."""
        exc = ConfigurationError("key", "reason")
        assert exc.category == ErrorCategory.CONFIG

    def test_validation_error_category(self):
        """Validation errors should have VALIDATION category."""
        exc = ValidationError("field", "reason")
        assert exc.category == ErrorCategory.VALIDATION

    def test_memory_error_category(self):
        """Memory errors should have MEMORY category."""
        exc = MemoryOperationError("store", "mem_1", "failed")
        assert exc.category == ErrorCategory.MEMORY

    def test_agent_error_category(self):
        """Agent errors should have AGENT category."""
        exc = AgentNotFoundError("agent_1")
        assert exc.category == ErrorCategory.AGENT

    def test_provider_error_category(self):
        """Provider errors should have PROVIDER category."""
        exc = UnsupportedProviderError("unknown")
        assert exc.category == ErrorCategory.PROVIDER


class TestMetadataValidationErrors:
    """Test specialized validation errors."""

    def test_metadata_validation_error(self):
        """Test metadata validation error."""
        exc = MetadataValidationError("metadata", "Too many keys")
        assert exc.error_code == "METADATA_VALIDATION_ERROR"
        assert exc.recoverable is False

    def test_attribute_validation_error(self):
        """Test attribute validation error."""
        exc = AttributeValidationError("attributes", "Key too long")
        assert exc.error_code == "ATTRIBUTE_VALIDATION_ERROR"
        assert exc.recoverable is False


class TestUnsupportedTransportError:
    """Test unsupported transport error."""

    def test_unsupported_transport_error(self):
        """Test unsupported transport error."""
        exc = UnsupportedTransportError(
            transport="websocket",
            supported_transports=["stdio", "sse"]
        )
        assert exc.recoverable is False
        assert exc.error_code == "UNSUPPORTED_TRANSPORT_ERROR"
        assert exc.transport == "websocket"
        assert "stdio" in str(exc)
        assert "sse" in str(exc)


class TestDependencyMissingError:
    """Test dependency missing error."""

    def test_dependency_missing_error(self):
        """Test dependency missing error."""
        exc = DependencyMissingError(
            dependency="mcp",
            message="Install with: pip install mcp"
        )
        assert exc.recoverable is False
        assert exc.error_code == "DEPENDENCY_MISSING_ERROR"
        assert exc.dependency == "mcp"
        assert "pip install mcp" in str(exc)


class TestErrorContext:
    """Test error context handling."""

    def test_context_preserved_in_subclass(self):
        """Context should be preserved in subclasses."""
        exc = StorageConnectionError(
            backend="redis",
            message="Connection failed",
            context={"retry_count": 3, "last_error": "ECONNREFUSED"}
        )
        assert exc.context["retry_count"] == 3
        assert exc.context["last_error"] == "ECONNREFUSED"
        assert exc.context["backend"] == "redis"

    def test_value_truncation_in_validation_error(self):
        """Large values should be truncated in validation error context."""
        large_value = "x" * 200
        exc = ValidationError("field", "too long", value=large_value)
        assert len(exc.context["value"]) == 103  # 100 + "..."


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
