"""
HAIM Test Suite — Configuration Tests
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.config import (
    HAIMConfig,
    load_config,
    get_config,
    reset_config,
    TierConfig,
    LTPConfig,
)


@pytest.fixture(autouse=True)
def clean_config():
    """Reset global config singleton between tests."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def sample_config_path(tmp_path):
    """Create a temporary config.yaml."""
    config_data = {
        "haim": {
            "version": "3.0-test",
            "dimensionality": 1024,  # Small for tests
            "encoding": {"mode": "binary", "token_method": "bundle"},
            "tiers": {
                "hot": {"max_memories": 100, "ltp_threshold_min": 0.7},
                "warm": {
                    "max_memories": 1000,
                    "ltp_threshold_min": 0.3,
                    "consolidation_interval_hours": 1,
                    "storage_backend": "mmap",
                },
                "cold": {
                    "max_memories": 0,
                    "ltp_threshold_min": 0.0,
                    "storage_backend": "filesystem",
                },
            },
            "ltp": {
                "initial_importance": 0.5,
                "decay_lambda": 0.01,
                "permanence_threshold": 0.95,
                "half_life_days": 30.0,
            },
            "hysteresis": {"promote_delta": 0.15, "demote_delta": 0.10},
            "redis": {"url": "redis://localhost:6379/0"},
            "qdrant": {"url": "http://localhost:6333"},
            "gpu": {"enabled": False},
            "observability": {"log_level": "DEBUG"},
            "paths": {"data_dir": str(tmp_path / "data")},
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestLoadConfig:
    def test_load_from_yaml(self, sample_config_path):
        config = load_config(sample_config_path)
        assert config.version == "3.0-test"
        assert config.dimensionality == 1024

    def test_default_values_when_no_file(self, tmp_path):
        missing_path = tmp_path / "nonexistent.yaml"
        config = load_config(missing_path)
        assert config.dimensionality == 16384
        assert config.version == "3.0"

    def test_dimensionality_must_be_multiple_of_64(self, tmp_path):
        bad_config = {"haim": {"dimensionality": 100}}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(ValueError, match="multiple of 64"):
            load_config(path)

    def test_encoding_mode(self, sample_config_path):
        config = load_config(sample_config_path)
        assert config.encoding.mode == "binary"
        assert config.encoding.token_method == "bundle"

    def test_tier_config(self, sample_config_path):
        config = load_config(sample_config_path)
        assert config.tiers_hot.max_memories == 100
        assert config.tiers_hot.ltp_threshold_min == 0.7
        assert config.tiers_warm.storage_backend == "mmap"
        assert config.tiers_warm.consolidation_interval_hours == 1

    def test_ltp_config(self, sample_config_path):
        config = load_config(sample_config_path)
        assert config.ltp.decay_lambda == 0.01
        assert config.ltp.permanence_threshold == 0.95

    def test_hysteresis_config(self, sample_config_path):
        config = load_config(sample_config_path)
        assert config.hysteresis.promote_delta == 0.15
        assert config.hysteresis.demote_delta == 0.10


class TestEnvironmentOverrides:
    def test_dimensionality_override(self, sample_config_path):
        os.environ["HAIM_DIMENSIONALITY"] = "2048"
        try:
            config = load_config(sample_config_path)
            assert config.dimensionality == 2048
        finally:
            del os.environ["HAIM_DIMENSIONALITY"]

    def test_redis_url_override(self, sample_config_path):
        os.environ["HAIM_REDIS_URL"] = "redis://custom:6380/1"
        try:
            config = load_config(sample_config_path)
            assert config.redis.url == "redis://custom:6380/1"
        finally:
            del os.environ["HAIM_REDIS_URL"]

    def test_gpu_enabled_override(self, sample_config_path):
        os.environ["HAIM_GPU_ENABLED"] = "true"
        try:
            config = load_config(sample_config_path)
            assert config.gpu.enabled is True
        finally:
            del os.environ["HAIM_GPU_ENABLED"]

    def test_log_level_override(self, sample_config_path):
        os.environ["HAIM_LOG_LEVEL"] = "WARNING"
        try:
            config = load_config(sample_config_path)
            assert config.observability.log_level == "WARNING"
        finally:
            del os.environ["HAIM_LOG_LEVEL"]


class TestConfigSingleton:
    def test_get_config_returns_same_instance(self):
        config_a = get_config()
        config_b = get_config()
        assert config_a is config_b

    def test_reset_clears_singleton(self):
        config_a = get_config()
        reset_config()
        config_b = get_config()
        # New instance after reset (they're equal but not same object)
        assert config_a is not config_b

    def test_config_is_frozen(self):
        config = get_config()
        with pytest.raises(AttributeError):
            config.dimensionality = 9999


class TestConfigValidation:
    def test_valid_dimensionalities(self, tmp_path):
        for dim in [64, 128, 1024, 16384]:
            data = {"haim": {"dimensionality": dim}}
            path = tmp_path / f"config_{dim}.yaml"
            with open(path, "w") as f:
                yaml.dump(data, f)
            config = load_config(path)
            assert config.dimensionality == dim

    def test_invalid_dimensionalities(self, tmp_path):
        for dim in [100, 1000, 10000, 15000]:
            data = {"haim": {"dimensionality": dim}}
            path = tmp_path / f"config_{dim}.yaml"
            with open(path, "w") as f:
                yaml.dump(data, f)
            with pytest.raises(ValueError):
                load_config(path)
