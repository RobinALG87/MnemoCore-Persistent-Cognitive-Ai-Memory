"""Smoke checks for the service-independent AgentMemory import surface."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from os import environ
from pathlib import Path

import pytest


_WHEEL_BUILD_AVAILABLE = (
    importlib.util.find_spec("hatchling") is not None
    and importlib.util.find_spec("build") is not None
)

_BENCHMARK_RUNTIME_FILES = {
    "benchmarks/__init__.py",
    "benchmarks/base.py",
    "benchmarks/comparison.py",
    "benchmarks/entrypoint.py",
    "benchmarks/latency.py",
    "benchmarks/memory_footprint.py",
    "benchmarks/regression.py",
    "benchmarks/run_benchmarks.py",
    "benchmarks/runner.py",
    "benchmarks/throughput.py",
}

_BENCHMARK_WHEEL_EXCLUDES = {
    "benchmarks/agent_memory_baseline.py",
    "benchmarks/bench_*.py",
    "benchmarks/pytest_benchmarks.py",
    "benchmarks/test_*.py",
}


def test_agent_memory_import_does_not_load_optional_integrations() -> None:
    code = (
        "import sys; import mnemocore.agent_memory; "
        "assert not any(name in sys.modules for name in "
        "('fastapi', 'uvicorn', 'redis', 'qdrant_client', 'numpy'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={
            **environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        },
    )
    assert result.returncode == 0, result.stderr


def test_build_config_includes_benchmark_console_target() -> None:
    project_root = Path(__file__).parents[1]
    with (project_root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    scripts = config["project"]["scripts"]
    wheel_config = config["tool"]["hatch"]["build"]["targets"]["wheel"]
    sdist_include = config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert scripts["mnemocore-benchmark"] == "benchmarks.entrypoint:main"
    assert "benchmarks" in wheel_config["packages"]
    assert set(wheel_config["exclude"]) == _BENCHMARK_WHEEL_EXCLUDES
    assert "benchmarks/" in sdist_include


def test_benchmark_runtime_manifest_matches_source_tree() -> None:
    project_root = Path(__file__).parents[1]
    benchmark_files = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "benchmarks").glob("*.py")
        if not path.name.startswith("test_")
        and not path.name.startswith("bench_")
        and path.name not in {"agent_memory_baseline.py", "pytest_benchmarks.py"}
    }
    assert benchmark_files == _BENCHMARK_RUNTIME_FILES


def test_benchmark_entrypoint_help_smoke() -> None:
    code = (
        "import sys; sys.argv=['mnemocore-benchmark', '--help']; "
        "from benchmarks.entrypoint import main; main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={**environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert result.returncode == 0, result.stderr
    assert "MnemoCore Benchmark Suite" in result.stdout


@pytest.mark.skipif(
    not _WHEEL_BUILD_AVAILABLE,
    reason="wheel smoke requires the optional build and hatchling packages",
)
def test_built_wheel_installs_core_surface_in_clean_venv() -> None:
    project_root = Path(__file__).parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        wheel_dir = root / "dist"
        build_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheel_dir),
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert build_result.returncode == 0, build_result.stderr
        wheel = next(wheel_dir.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            packaged_benchmarks = {
                name for name in archive.namelist() if name.startswith("benchmarks/")
            }
        assert packaged_benchmarks == _BENCHMARK_RUNTIME_FILES

        venv_dir = root / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        venv_python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        install_result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--no-deps", str(wheel)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert install_result.returncode == 0, install_result.stderr
        smoke = subprocess.run(
            [
                str(venv_python),
                "-c",
                "import mnemocore.agent_memory; import mnemocore; print(mnemocore.__version__)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert smoke.returncode == 0, smoke.stderr
