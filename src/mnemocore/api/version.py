"""Runtime version lookup without importing optional integrations."""

from importlib.metadata import PackageNotFoundError, version as metadata_version

from mnemocore._version import __version__ as _SOURCE_VERSION

_cached_version: str | None = None


def get_version() -> str:
    """Return the installed distribution version or source fallback."""
    global _cached_version
    if _cached_version is not None:
        return _cached_version

    try:
        _cached_version = metadata_version("mnemocore")
    except PackageNotFoundError:
        _cached_version = _SOURCE_VERSION
    return _cached_version


__version__ = get_version()

__all__ = ["get_version", "__version__"]
