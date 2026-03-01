"""
Version Management
==================
Centralized version handling for MnemoCore API.
"""

from typing import Optional

# Fallback version if package metadata is not available
_FALLBACK_VERSION = "2.0.0"

_cached_version: Optional[str] = None


def get_version() -> str:
    """
    Get the current MnemoCore version.

    Tries to read from package metadata first, falls back to hardcoded version.

    Returns:
        The version string (e.g., "5.0.0")
    """
    global _cached_version

    if _cached_version is not None:
        return _cached_version

    # Try to get version from package metadata
    try:
        from importlib.metadata import version
        _cached_version = version("mnemocore")
        return _cached_version
    except Exception:
        # Package not installed or metadata not available
        pass

    # Try to read from pyproject.toml
    try:
        from pathlib import Path
        pyproject_path = Path(__file__).parent.parent.parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            import tomllib
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                if "project" in data and "version" in data["project"]:
                    _cached_version = data["project"]["version"]
                    return _cached_version
    except Exception:
        pass

    # Fallback to hardcoded version
    _cached_version = _FALLBACK_VERSION
    return _cached_version


# Module-level version string for direct access
__version__ = get_version()

__all__ = ["get_version", "__version__"]
