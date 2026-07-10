"""Public API for MnemoCore's vNext agent-memory foundation."""

from .client import AgentMemory, MemorySession, SyncAgentMemory, SyncMemorySession
from .errors import (
    AgentMemoryError,
    ClosedStoreError,
    MemoryNotFoundError,
    ScopeError,
    StorageError,
    ValidationError,
)
from .models import (
    MemoryEvent,
    MemoryEventType,
    MemoryHistoryEntry,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    RecallResult,
    utc_now,
)

__all__ = [
    "AgentMemory",
    "AgentMemoryError",
    "ClosedStoreError",
    "MemoryEvent",
    "MemoryEventType",
    "MemoryHistoryEntry",
    "MemoryKind",
    "MemoryNotFoundError",
    "MemoryRecord",
    "MemoryScope",
    "MemoryStatus",
    "RecallResult",
    "MemorySession",
    "ScopeError",
    "StorageError",
    "SyncAgentMemory",
    "SyncMemorySession",
    "ValidationError",
    "utc_now",
]
