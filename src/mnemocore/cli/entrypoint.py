"""Optional legacy CLI console entrypoint."""

from __future__ import annotations


def main() -> None:
    """Dispatch to the Click CLI when the CLI extra is installed."""
    try:
        from .main import cli
    except ImportError as error:  # pragma: no cover - clean-install path
        raise SystemExit(
            "The CLI requires optional dependencies. "
            "Install them with: pip install 'mnemocore[cli]'"
        ) from error

    cli()


__all__ = ["main"]
