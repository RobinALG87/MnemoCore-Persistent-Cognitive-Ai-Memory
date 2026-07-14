"""Optional legacy v2 Click CLI; it is not the scoped v3 runtime."""

from typing import Any

__all__ = ["cli"]


def __getattr__(name: str) -> Any:
    if name == "cli":
        from .main import cli

        globals()[name] = cli
        return cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
