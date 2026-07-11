from pathlib import Path
import tomllib

import pytest

import tests.conftest as harness


def test_default_pytest_path_excludes_benchmarks():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        config = tomllib.load(project_file)

    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_required_offline_lane_is_green_adapters_only():
    root = Path(__file__).resolve().parents[1]
    lanes = (root / "docs" / "TEST_LANES.md").read_text(encoding="utf-8")

    assert "python -m pytest tests/integrations -q" in lanes
    assert (
        "python -m pytest tests/integrations "
        "tests/test_integration_store_query_cycle.py" not in lanes
    )


def test_known_red_lifecycle_is_quarantined_and_service_lane_is_inactive():
    root = Path(__file__).resolve().parents[1]
    lanes = (root / "docs" / "TEST_LANES.md").read_text(encoding="utf-8")

    assert "Quarantined known-red" in lanes
    assert "test_integration_store_query_cycle.py" in lanes
    assert "obsolete `AsyncQdrantClient` patch target" in lanes
    assert "stale lifecycle behavior assumptions" in lanes
    assert "Service lane: inactive" in lanes
    assert "nonzero collection" in lanes
    inactive_section = lanes.split("Service lane: inactive", 1)[1]
    assert "python -m pytest tests --run-integration -m" not in inactive_section


def test_agent_memory_contract_tests_are_not_orphaned_in_benchmarks():
    root = Path(__file__).resolve().parents[1]
    contract_names = ("test_fingerprint.py", "test_hdv_isolation.py")

    for name in contract_names:
        assert (root / "tests" / "agent_memory" / name).is_file()
        assert not (root / "benchmarks" / name).exists()


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


def test_single_service_marker_only_probes_that_service(monkeypatch):
    calls = {"redis": 0}

    def redis_probe():
        calls["redis"] += 1
        return True

    def unexpected_qdrant_probe():
        raise AssertionError("Redis-only tests must not probe Qdrant")

    monkeypatch.setattr(harness, "_check_redis_available", redis_probe)
    monkeypatch.setattr(
        harness, "_check_qdrant_available", unexpected_qdrant_probe
    )

    harness.pytest_collection_modifyitems(
        _Config(run_integration=True),
        [_Item("integration", "requires_redis")],
    )

    assert calls == {"redis": 1}


def test_service_requirement_without_integration_marker_is_rejected():
    with pytest.raises(pytest.UsageError, match="requires_redis.*integration"):
        harness.pytest_collection_modifyitems(
            _Config(run_integration=True), [_Item("requires_redis")]
        )


def test_offline_integration_file_has_no_service_requirements():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tests" / "test_integration_store_query_cycle.py").read_text(
        encoding="utf-8"
    )

    module_markers = source.split("import numpy", 1)[0]
    assert "pytest.mark.integration" in module_markers
    assert "pytest.mark.requires_redis" not in module_markers
    assert "pytest.mark.requires_qdrant" not in module_markers


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


def test_clean_lane_does_not_require_legacy_config_reset():
    root = Path(__file__).resolve().parents[1]

    assert not harness._requires_legacy_config_reset(
        root / "tests" / "agent_memory" / "test_client.py"
    )
    assert not harness._requires_legacy_config_reset(
        root / "tests" / "integrations" / "test_agent_memory_integrations.py"
    )
    assert harness._requires_legacy_config_reset(
        root / "tests" / "test_engine.py"
    )
