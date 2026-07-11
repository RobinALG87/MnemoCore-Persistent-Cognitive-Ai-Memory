from pathlib import Path
import tomllib

import tests.conftest as harness


def test_default_pytest_path_excludes_benchmarks():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        config = tomllib.load(project_file)

    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


class _Config:
    def __init__(self, *, run_integration=False, run_slow=False):
        self.options = {
            "--run-integration": run_integration,
            "--run-slow": run_slow,
        }

    def getoption(self, name, default=False):
        return self.options.get(name, default)


class _Item:
    def __init__(self, *keywords):
        self.keywords = set(keywords)
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


def test_default_collection_does_not_probe_external_services(monkeypatch):
    def unexpected_probe():
        raise AssertionError("offline collection must not probe services")

    monkeypatch.setattr(harness, "_check_redis_available", unexpected_probe)
    monkeypatch.setattr(harness, "_check_qdrant_available", unexpected_probe)

    harness.pytest_collection_modifyitems(_Config(), [_Item("integration")])


def test_service_probes_run_once_when_integration_is_requested(monkeypatch):
    calls = {"redis": 0, "qdrant": 0}

    def redis_probe():
        calls["redis"] += 1
        return True

    def qdrant_probe():
        calls["qdrant"] += 1
        return True

    monkeypatch.setattr(harness, "_check_redis_available", redis_probe)
    monkeypatch.setattr(harness, "_check_qdrant_available", qdrant_probe)
    items = [
        _Item("integration", "requires_redis"),
        _Item("integration", "requires_redis", "requires_qdrant"),
    ]

    harness.pytest_collection_modifyitems(
        _Config(run_integration=True), items
    )

    assert calls == {"redis": 1, "qdrant": 1}


def test_clean_lane_does_not_require_legacy_hardware_mocks():
    root = Path(__file__).resolve().parents[1]

    assert not harness._requires_legacy_hardware_mocks(
        [root / "tests" / "agent_memory" / "test_client.py"]
    )
    assert not harness._requires_legacy_hardware_mocks(
        [root / "tests" / "integrations" / "test_agent_memory_integrations.py"]
    )
    assert harness._requires_legacy_hardware_mocks(
        [root / "tests" / "test_engine.py"]
    )
