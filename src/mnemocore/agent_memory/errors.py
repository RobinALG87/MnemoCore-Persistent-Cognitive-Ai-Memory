"""Errors raised by MnemoCore's agent-memory API."""


class AgentMemoryError(Exception):
    """Base error for the vNext agent-memory API."""


class ValidationError(AgentMemoryError):
    """Raised when an agent-memory value is invalid."""


class ScopeError(ValidationError):
    """Raised when a memory scope is invalid."""


class StorageError(AgentMemoryError):
    """Raised when a memory storage operation fails."""


class MemoryNotFoundError(AgentMemoryError):
    """Raised when a requested memory does not exist."""


class MemoryConflictError(AgentMemoryError):
    """Raised when a memory operation conflicts with existing truth."""


class ClosedStoreError(StorageError):
    """Raised when an operation targets a closed memory store."""
