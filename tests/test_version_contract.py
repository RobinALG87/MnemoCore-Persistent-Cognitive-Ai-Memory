"""Version authority and optional entrypoint contracts."""

from importlib.metadata import PackageNotFoundError, version as metadata_version

import pytest

import mnemocore
from mnemocore.api.entrypoint import main as server_main
from mnemocore.api.version import get_version
from mnemocore.cli.entrypoint import main as cli_main


def test_version_is_single_source_and_public() -> None:
    assert mnemocore.__version__ == get_version()
    assert mnemocore.__version__


def test_installed_metadata_matches_public_version_when_available() -> None:
    try:
        installed = metadata_version("mnemocore")
    except PackageNotFoundError:
        pytest.skip("package metadata is unavailable in source checkout")
    assert installed == mnemocore.__version__


def test_optional_console_entrypoints_are_real_callables() -> None:
    assert callable(server_main)
    assert callable(cli_main)

