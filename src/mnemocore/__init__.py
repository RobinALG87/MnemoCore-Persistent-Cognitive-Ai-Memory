"""
MnemoCore - Infrastructure for Persistent Cognitive Memory
===========================================================

Lightweight agentic memory for AI. ``Memory`` remains a lazy compatibility
export only so it can raise the v3 migration error without importing the
legacy engine stack. Use AgentMemory or HybridMemoryRuntime for all new code.
"""

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from ._memory import Memory

__all__ = ["Memory", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "Memory":
        from ._memory import Memory

        globals()[name] = Memory
        return Memory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
