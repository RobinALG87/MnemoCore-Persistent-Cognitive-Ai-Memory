"""
MnemoCore - Infrastructure for Persistent Cognitive Memory
===========================================================

Lightweight agentic memory for AI. The legacy ``Memory`` facade remains
available as a lazy export so focused subpackages can be imported without
initializing the legacy engine stack.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._memory import Memory

__version__ = "2.0.0"
__all__ = ["Memory", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "Memory":
        from ._memory import Memory

        globals()[name] = Memory
        return Memory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
