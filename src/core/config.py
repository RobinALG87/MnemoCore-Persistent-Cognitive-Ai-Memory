"""
HAIM Configuration System
========================
Centralized, validated configuration with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class TierConfig:
    max_memories: int
    ltp_threshold_min: float
    eviction_policy: str = "lru"
    consolidation_interval_hours: Optional[int] = None
    storage_backend: str = "memory"
    compression: str = "gzip"
    archive_threshold_days: int = 30


@dataclass(frozen=True)
class LTPConfig:
    initial_importance: float = 0.5
    decay_lambda: float = 0.01
    permanence_threshold: float = 0.95
    half_life_days: float = 30.0


@dataclass(frozen=True)
class HysteresisConfig:
    promote_delta: float = 0.15
    demote_delta: float = 0.10


@dataclass(frozen=True)
class RedisConfig:
    url: str = "redis://localhost:6379/0"
    stream_key: str = "haim:subconscious"
    max_connections: int = 10
    socket_timeout: int = 5


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection_hot: str = "haim_hot"
    collection_warm: str = "haim_warm"
    binary_quantization: bool = True
    always_ram: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100


@dataclass(frozen=True)
class GPUConfig:
    enabled: bool = False
    device: str = "cuda:0"
    batch_size: int = 1000
    fallback_to_cpu: bool = True


@dataclass(frozen=True)
class SecurityConfig:
    api_key: str = "mnemocore-beta-key"


@dataclass(frozen=True)
class ObservabilityConfig:
    metrics_port: int = 9090
    log_level: str = "INFO"
    structured_logging: bool = True


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "./data"
    memory_file: str = "./data/memory.jsonl"
    codebook_file: str = "./data/codebook.json"
    concepts_file: str = "./data/concepts.json"
    synapses_file: str = "./data/synapses.json"
    warm_mmap_dir: str = "./data/warm_tier"
    cold_archive_dir: str = "./data/cold_archive"


@dataclass(frozen=True)
class EncodingConfig:
    mode: str = "binary"  # "binary" or "float"
    token_method: str = "bundle"


@dataclass(frozen=True)
class HAIMConfig:
    """Root configuration for the HAIM system."""

    version: str = "3.0"
    dimensionality: int = 16384
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    tiers_hot: TierConfig = field(
        default_factory=lambda: TierConfig(max_memories=2000, ltp_threshold_min=0.7)
    )
    tiers_warm: TierConfig = field(
        default_factory=lambda: TierConfig(
            max_memories=100000,
            ltp_threshold_min=0.3,
            consolidation_interval_hours=1,
            storage_backend="mmap",
        )
    )
    tiers_cold: TierConfig = field(
        default_factory=lambda: TierConfig(
            max_memories=0,  # unlimited
            ltp_threshold_min=0.0,
            storage_backend="filesystem",
        )
    )
    ltp: LTPConfig = field(default_factory=LTPConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _env_override(key: str, default):
    """Check for HAIM_<KEY> environment variable override."""
    env_key = f"HAIM_{key.upper()}"
    val = os.environ.get(env_key)
    if val is None:
        return default
    # Type coercion based on the default's type
    if isinstance(default, bool):
        return val.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


def _build_tier(raw: dict) -> TierConfig:
    return TierConfig(
        max_memories=raw.get("max_memories", 0),
        ltp_threshold_min=raw.get("ltp_threshold_min", 0.0),
        eviction_policy=raw.get("eviction_policy", "lru"),
        consolidation_interval_hours=raw.get("consolidation_interval_hours"),
        storage_backend=raw.get("storage_backend", "memory"),
        compression=raw.get("compression", "gzip"),
        archive_threshold_days=raw.get("archive_threshold_days", 30),
    )


def load_config(path: Optional[Path] = None) -> HAIMConfig:
    """
    Load configuration from YAML file with environment variable overrides.

    Priority: ENV > YAML > defaults.

    Args:
        path: Path to config.yaml. If None, searches ./config.yaml and ../config.yaml.

    Returns:
        Validated HAIMConfig instance.

    Raises:
        ValueError: If dimensionality is not a multiple of 64.
        FileNotFoundError: If no config file is found and path is explicitly set.
    """
    if path is None:
        # Search common locations
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    raw = {}
    if path is not None and path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f).get("haim", {})

    # Apply env overrides to top-level scalars
    dimensionality = _env_override(
        "DIMENSIONALITY", raw.get("dimensionality", 16384)
    )

    # Validate
    if dimensionality % 64 != 0:
        raise ValueError(
            f"Dimensionality must be a multiple of 64 for efficient bit packing, got {dimensionality}"
        )

    # Build tier configs
    tiers_raw = raw.get("tiers", {})
    hot_raw = tiers_raw.get("hot", {"max_memories": 2000, "ltp_threshold_min": 0.7})
    warm_raw = tiers_raw.get(
        "warm",
        {
            "max_memories": 100000,
            "ltp_threshold_min": 0.3,
            "consolidation_interval_hours": 1,
            "storage_backend": "mmap",
        },
    )
    cold_raw = tiers_raw.get(
        "cold",
        {
            "max_memories": 0,
            "ltp_threshold_min": 0.0,
            "storage_backend": "filesystem",
        },
    )

    # Build encoding config
    enc_raw = raw.get("encoding", {})
    encoding = EncodingConfig(
        mode=_env_override("ENCODING_MODE", enc_raw.get("mode", "binary")),
        token_method=enc_raw.get("token_method", "bundle"),
    )

    # Build LTP config
    ltp_raw = raw.get("ltp", {})
    ltp = LTPConfig(
        initial_importance=ltp_raw.get("initial_importance", 0.5),
        decay_lambda=ltp_raw.get("decay_lambda", 0.01),
        permanence_threshold=ltp_raw.get("permanence_threshold", 0.95),
        half_life_days=ltp_raw.get("half_life_days", 30.0),
    )

    # Build paths config
    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        data_dir=paths_raw.get("data_dir", "./data"),
        memory_file=paths_raw.get("memory_file", "./data/memory.jsonl"),
        codebook_file=paths_raw.get("codebook_file", "./data/codebook.json"),
        concepts_file=paths_raw.get("concepts_file", "./data/concepts.json"),
        synapses_file=paths_raw.get("synapses_file", "./data/synapses.json"),
        warm_mmap_dir=paths_raw.get("warm_mmap_dir", "./data/warm_tier"),
        cold_archive_dir=paths_raw.get("cold_archive_dir", "./data/cold_archive"),
    )

    # Build redis config
    redis_raw = raw.get("redis", {})
    redis = RedisConfig(
        url=_env_override("REDIS_URL", redis_raw.get("url", "redis://localhost:6379/0")),
        stream_key=redis_raw.get("stream_key", "haim:subconscious"),
        max_connections=redis_raw.get("max_connections", 10),
        socket_timeout=redis_raw.get("socket_timeout", 5),
    )

    # Build qdrant config
    qdrant_raw = raw.get("qdrant", {})
    qdrant = QdrantConfig(
        url=_env_override(
            "QDRANT_URL", qdrant_raw.get("url", "http://localhost:6333")
        ),
        collection_hot=qdrant_raw.get("collection_hot", "haim_hot"),
        collection_warm=qdrant_raw.get("collection_warm", "haim_warm"),
        binary_quantization=qdrant_raw.get("binary_quantization", True),
        always_ram=qdrant_raw.get("always_ram", True),
        hnsw_m=qdrant_raw.get("hnsw_m", 16),
        hnsw_ef_construct=qdrant_raw.get("hnsw_ef_construct", 100),
    )

    # Build GPU config
    gpu_raw = raw.get("gpu", {})
    gpu = GPUConfig(
        enabled=_env_override("GPU_ENABLED", gpu_raw.get("enabled", False)),
        device=gpu_raw.get("device", "cuda:0"),
        batch_size=gpu_raw.get("batch_size", 1000),
        fallback_to_cpu=gpu_raw.get("fallback_to_cpu", True),
    )

    # Build observability config
    obs_raw = raw.get("observability", {})
    observability = ObservabilityConfig(
        metrics_port=obs_raw.get("metrics_port", 9090),
        log_level=_env_override("LOG_LEVEL", obs_raw.get("log_level", "INFO")),
        structured_logging=obs_raw.get("structured_logging", True),
    )

    # Build security config
    sec_raw = raw.get("security", {})
    security = SecurityConfig(
        api_key=_env_override("API_KEY", sec_raw.get("api_key", "mnemocore-beta-key")),
    )

    # Build hysteresis config
    hyst_raw = raw.get("hysteresis", {})
    hysteresis = HysteresisConfig(
        promote_delta=hyst_raw.get("promote_delta", 0.15),
        demote_delta=hyst_raw.get("demote_delta", 0.10),
    )

    return HAIMConfig(
        version=raw.get("version", "3.0"),
        dimensionality=dimensionality,
        encoding=encoding,
        tiers_hot=_build_tier(hot_raw),
        tiers_warm=_build_tier(warm_raw),
        tiers_cold=_build_tier(cold_raw),
        ltp=ltp,
        hysteresis=hysteresis,
        redis=redis,
        qdrant=qdrant,
        gpu=gpu,
        security=security,
        observability=observability,
        paths=paths,
    )


# Module-level singleton (lazy-loaded)
_CONFIG: Optional[HAIMConfig] = None


def get_config() -> HAIMConfig:
    """Get or initialize the global config singleton."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def reset_config():
    """Reset the global config singleton (useful for testing)."""
    global _CONFIG
    _CONFIG = None
