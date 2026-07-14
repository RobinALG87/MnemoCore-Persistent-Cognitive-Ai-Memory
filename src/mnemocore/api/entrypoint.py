"""Optional legacy v2 REST entrypoint; v3 deployments compose create_v3_app."""

from __future__ import annotations

import os


def main() -> None:
    """Start the legacy HAIM REST server when the server extra is installed."""
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - clean-install path
        raise SystemExit(
            "The REST server requires optional dependencies. "
            "Install them with: pip install 'mnemocore[server]'"
        ) from error

    uvicorn.run(
        "mnemocore.api.main:app",
        host=os.getenv("MNEMOCORE_HOST", "127.0.0.1"),
        port=int(os.getenv("MNEMOCORE_PORT", "8100")),
    )


__all__ = ["main"]
