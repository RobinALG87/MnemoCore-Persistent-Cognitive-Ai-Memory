"""Migration checks for removed v2 lite entry points."""

import pytest

from mnemocore import Memory
from mnemocore.core.config import ConfigurationError, load_config


def test_lite_memory_profile_fails_with_a_v3_migration_error():
    with pytest.raises(RuntimeError, match="v3 migration.*AgentMemory"):
        Memory(profile="lite")


def test_default_memory_is_treated_as_the_removed_lite_profile():
    with pytest.raises(RuntimeError, match="v3 migration.*AgentMemory"):
        Memory()


def test_default_memory_raises_before_loading_the_legacy_engine(monkeypatch):
    import mnemocore._memory as memory_module

    def legacy_engine_must_not_load():
        pytest.fail("bare Memory() must not initialize the legacy HAIM engine")

    monkeypatch.setattr(memory_module, "_get_haim_engine", legacy_engine_must_not_load)
    with pytest.raises(RuntimeError, match="v3 migration.*AgentMemory"):
        Memory()


def test_lite_environment_profile_fails_with_a_v3_migration_error(monkeypatch):
    monkeypatch.setenv("HAIM_PROFILE", "lite")
    with pytest.raises(ConfigurationError, match="v3 migration.*AgentMemory"):
        load_config()
