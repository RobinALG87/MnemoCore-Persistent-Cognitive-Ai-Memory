"""Optional benchmark console entrypoint."""

from __future__ import annotations

import asyncio


def main() -> int:
    """Run the benchmark CLI when the benchmark extra is installed."""
    try:
        from .run_benchmarks import main as async_main
    except ImportError as error:  # pragma: no cover - clean-install path
        raise SystemExit(
            "The benchmark CLI requires optional dependencies. "
            "Install them with: pip install 'mnemocore[benchmark]'"
        ) from error
    return asyncio.run(async_main())


__all__ = ["main"]

