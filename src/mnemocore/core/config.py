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

from mnemocore.core.exceptions import ConfigurationError


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
    password: Optional[str] = None


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection_hot: str = "haim_hot"
    collection_warm: str = "haim_warm"
    binary_quantization: bool = True
    always_ram: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    api_key: Optional[str] = None


@dataclass(frozen=True)
class GPUConfig:
    enabled: bool = False
    device: str = "cuda:0"
    batch_size: int = 1000
    fallback_to_cpu: bool = True


@dataclass(frozen=True)
class SecurityConfig:
    api_key: Optional[str] = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass(frozen=True)
class ObservabilityConfig:
    metrics_port: int = 9090
    log_level: str = "INFO"
    structured_logging: bool = True


@dataclass(frozen=True)
class MCPConfig:
    enabled: bool = False
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8110
    api_base_url: str = "http://localhost:8100"
    api_key: Optional[str] = None
    timeout_seconds: int = 15
    allow_tools: list[str] = field(
        default_factory=lambda: [
            "memory_store",
            "memory_query",
            "memory_get",
            "memory_delete",
            "memory_stats",
            "memory_health",
        ]
    )


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
class AttentionMaskingConfig:
    """Configuration for XOR-based project isolation (Phase 4.1)."""
    enabled: bool = True


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for semantic consolidation (Phase 4.0+)."""
    enabled: bool = True
    interval_seconds: int = 3600  # 1 hour
    similarity_threshold: float = 0.85
    min_cluster_size: int = 2
    hot_tier_enabled: bool = True
    warm_tier_enabled: bool = True


@dataclass(frozen=True)
class EncodingConfig:
    mode: str = "binary"  # "binary" or "float"
    token_method: str = "bundle"


@dataclass(frozen=True)
class SynapseConfig:
    """Configuration for Phase 12.1: Aggressive Synapse Formation"""
    similarity_threshold: float = 0.5
    auto_bind_on_store: bool = True
    multi_hop_depth: int = 2


@dataclass(frozen=True)
class ContextConfig:
    """Configuration for Phase 12.2: Contextual Awareness"""
    enabled: bool = True
    shift_threshold: float = 0.3
    rolling_window_size: int = 5


@dataclass(frozen=True)
class PreferenceConfig:
    """Configuration for Phase 12.3: Preference Learning"""
    enabled: bool = True
    learning_rate: float = 0.1
    history_limit: int = 100


@dataclass(frozen=True)
class AnticipatoryConfig:
    """Configuration for Phase 13.2: Anticipatory Memory"""
    enabled: bool = True
    predictive_depth: int = 1


@dataclass(frozen=True)
class DreamLoopConfig:
    """Configuration for the dream loop (subconscious background processing)."""
    enabled: bool = True
    frequency_seconds: int = 60
    batch_size: int = 10
    max_iterations: int = 0  # 0 = unlimited
    subconscious_queue_maxlen: Optional[int] = None
    ollama_url: str = "http://localhost:11434/api/generate"
    model: str = "gemma3:1b"


@dataclass(frozen=True)
class SubconsciousAIConfig:
    """
    Configuration for the Subconscious AI worker (Phase 4.4).

    A small LLM (Phi 3.5, Llama 7B) that pulses in the background,
    performing memory sorting, enhanced dreaming, and micro self-improvement.

    This is a BETA feature that must be explicitly enabled.
    """
    # Opt-in BETA feature flag (MUST be explicitly enabled)
    enabled: bool = False
    beta_mode: bool = True  # Extra safety checks when True

    # Model configuration
    model_provider: str = "ollama"  # "ollama" | "lm_studio" | "openai_api" | "anthropic_api"
    model_name: str = "phi3.5:3.8b"  # Default: Phi 3.5 (small, fast)
    model_url: str = "http://localhost:11434"
    api_key: Optional[str] = None  # For API providers
    api_base_url: Optional[str] = None  # Override base URL for API providers

    # Pulse configuration
    pulse_interval_seconds: int = 120  # Default: 2 minutes between pulses
    pulse_backoff_enabled: bool = True  # Increase interval on errors
    pulse_backoff_max_seconds: int = 600  # Max backoff: 10 minutes

    # Resource management
    max_cpu_percent: float = 30.0  # Skip pulse if CPU > this
    cycle_timeout_seconds: int = 30  # Max time per LLM call
    rate_limit_per_hour: int = 50  # Max LLM calls per hour

    # Operations (all can be toggled independently)
    memory_sorting_enabled: bool = True  # Categorize and tag memories
    enhanced_dreaming_enabled: bool = True  # LLM-assisted consolidation
    micro_self_improvement_enabled: bool = False  # Pattern analysis (disabled by default)

    # Safety settings
    dry_run: bool = True  # When True, only log suggestions without applying
    log_all_decisions: bool = True  # Full audit trail
    audit_trail_path: Optional[str] = "./data/subconscious_audit.jsonl"
    max_memories_per_cycle: int = 10  # Process at most N memories per pulse


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
    mcp: MCPConfig = field(default_factory=MCPConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    attention_masking: AttentionMaskingConfig = field(default_factory=AttentionMaskingConfig)
    synapse: SynapseConfig = field(default_factory=SynapseConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    anticipatory: AnticipatoryConfig = field(default_factory=AnticipatoryConfig)
    dream_loop: DreamLoopConfig = field(default_factory=DreamLoopConfig)
    subconscious_ai: SubconsciousAIConfig = field(default_factory=SubconsciousAIConfig)


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


def _build_tier(name: str, raw: dict) -> TierConfig:
    prefix = f"TIERS_{name.upper()}"
    return TierConfig(
        max_memories=_env_override(f"{prefix}_MAX_MEMORIES", raw.get("max_memories", 0)),
        ltp_threshold_min=_env_override(f"{prefix}_LTP_THRESHOLD_MIN", raw.get("ltp_threshold_min", 0.0)),
        eviction_policy=raw.get("eviction_policy", "lru"),
        consolidation_interval_hours=raw.get("consolidation_interval_hours"),
        storage_backend=raw.get("storage_backend", "memory"),
        compression=raw.get("compression", "gzip"),
        archive_threshold_days=raw.get("archive_threshold_days", 30),
    )


def _parse_optional_positive_int(value: Optional[object]) -> Optional[int]:
    """Parse positive int values. Non-positive/invalid values become None."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def load_config(path: Optional[Path] = None) -> HAIMConfig:
    """
    Load configuration from YAML file with environment variable overrides.

    Priority: ENV > YAML > defaults.

    Args:
        path: Path to config.yaml. If None, searches ./config.yaml and ../config.yaml.

    Returns:
        Validated HAIMConfig instance.

    Raises:
        ConfigurationError: If dimensionality is not a multiple of 64.
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
            loaded = yaml.safe_load(f) or {}
            raw = loaded.get("haim") or {}

    # Apply env overrides to top-level scalars
    dimensionality = _env_override(
        "DIMENSIONALITY", raw.get("dimensionality", 16384)
    )

    # Validate
    if dimensionality % 64 != 0:
        raise ConfigurationError(
            config_key="dimensionality",
            reason=f"Dimensionality must be a multiple of 64 for efficient bit packing, got {dimensionality}"
        )

    # Build tier configs
    tiers_raw = raw.get("tiers") or {}
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
    enc_raw = raw.get("encoding") or {}
    encoding = EncodingConfig(
        mode=_env_override("ENCODING_MODE", enc_raw.get("mode", "binary")),
        token_method=enc_raw.get("token_method", "bundle"),
    )

    # Build LTP config
    ltp_raw = raw.get("ltp") or {}
    ltp = LTPConfig(
        initial_importance=ltp_raw.get("initial_importance", 0.5),
        decay_lambda=ltp_raw.get("decay_lambda", 0.01),
        permanence_threshold=ltp_raw.get("permanence_threshold", 0.95),
        half_life_days=ltp_raw.get("half_life_days", 30.0),
    )

    # Build paths config
    paths_raw = raw.get("paths") or {}
    paths = PathsConfig(
        data_dir=_env_override("DATA_DIR", paths_raw.get("data_dir", "./data")),
        memory_file=_env_override("MEMORY_FILE", paths_raw.get("memory_file", "./data/memory.jsonl")),
        codebook_file=_env_override("CODEBOOK_FILE", paths_raw.get("codebook_file", "./data/codebook.json")),
        concepts_file=_env_override("CONCEPTS_FILE", paths_raw.get("concepts_file", "./data/concepts.json")),
        synapses_file=_env_override("SYNAPSES_FILE", paths_raw.get("synapses_file", "./data/synapses.json")),
        warm_mmap_dir=_env_override("WARM_MMAP_DIR", paths_raw.get("warm_mmap_dir", "./data/warm_tier")),
        cold_archive_dir=_env_override("COLD_ARCHIVE_DIR", paths_raw.get("cold_archive_dir", "./data/cold_archive")),
    )

    # Build redis config
    redis_raw = raw.get("redis") or {}
    redis = RedisConfig(
        url=_env_override("REDIS_URL", redis_raw.get("url", "redis://localhost:6379/0")),
        stream_key=redis_raw.get("stream_key", "haim:subconscious"),
        max_connections=redis_raw.get("max_connections", 10),
        socket_timeout=redis_raw.get("socket_timeout", 5),
        password=_env_override("REDIS_PASSWORD", redis_raw.get("password")),
    )

    # Build qdrant config
    qdrant_raw = raw.get("qdrant") or {}
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
        api_key=_env_override("QDRANT_API_KEY", qdrant_raw.get("api_key")),
    )

    # Build GPU config
    gpu_raw = raw.get("gpu") or {}
    gpu = GPUConfig(
        enabled=_env_override("GPU_ENABLED", gpu_raw.get("enabled", False)),
        device=gpu_raw.get("device", "cuda:0"),
        batch_size=gpu_raw.get("batch_size", 1000),
        fallback_to_cpu=gpu_raw.get("fallback_to_cpu", True),
    )

    # Build observability config
    obs_raw = raw.get("observability") or {}
    observability = ObservabilityConfig(
        metrics_port=obs_raw.get("metrics_port", 9090),
        log_level=_env_override("LOG_LEVEL", obs_raw.get("log_level", "INFO")),
        structured_logging=obs_raw.get("structured_logging", True),
    )

    # Build security config
    sec_raw = raw.get("security") or {}

    # Parse CORS origins from env (comma-separated) or config
    cors_env = os.environ.get("HAIM_CORS_ORIGINS")
    if cors_env:
        cors_origins = [o.strip() for o in cors_env.split(",")]
    else:
        cors_origins = sec_raw.get("cors_origins", ["*"])

    security = SecurityConfig(
        api_key=_env_override("API_KEY", sec_raw.get("api_key")),
        cors_origins=cors_origins,
        rate_limit_enabled=_env_override("RATE_LIMIT_ENABLED", sec_raw.get("rate_limit_enabled", True)),
        rate_limit_requests=_env_override("RATE_LIMIT_REQUESTS", sec_raw.get("rate_limit_requests", 100)),
        rate_limit_window=_env_override("RATE_LIMIT_WINDOW", sec_raw.get("rate_limit_window", 60)),
    )

    # Build MCP config
    mcp_raw = raw.get("mcp") or {}
    allow_tools_default = [
        "memory_store",
        "memory_query",
        "memory_get",
        "memory_delete",
        "memory_stats",
        "memory_health",
    ]
    mcp = MCPConfig(
        enabled=_env_override("MCP_ENABLED", mcp_raw.get("enabled", False)),
        transport=_env_override("MCP_TRANSPORT", mcp_raw.get("transport", "stdio")),
        host=_env_override("MCP_HOST", mcp_raw.get("host", "127.0.0.1")),
        port=_env_override("MCP_PORT", mcp_raw.get("port", 8110)),
        api_base_url=_env_override("MCP_API_BASE_URL", mcp_raw.get("api_base_url", "http://localhost:8100")),
        api_key=_env_override("MCP_API_KEY", mcp_raw.get("api_key", sec_raw.get("api_key"))),
        timeout_seconds=_env_override("MCP_TIMEOUT_SECONDS", mcp_raw.get("timeout_seconds", 15)),
        allow_tools=mcp_raw.get("allow_tools", allow_tools_default),
    )

    # Build hysteresis config
    hyst_raw = raw.get("hysteresis") or {}
    hysteresis = HysteresisConfig(
        promote_delta=_env_override("HYSTERESIS_PROMOTE_DELTA", hyst_raw.get("promote_delta", 0.15)),
        demote_delta=_env_override("HYSTERESIS_DEMOTE_DELTA", hyst_raw.get("demote_delta", 0.10)),
    )

    # Build LTP config
    ltp_raw = raw.get("ltp") or {}
    ltp = LTPConfig(
        initial_importance=_env_override("LTP_INITIAL_IMPORTANCE", ltp_raw.get("initial_importance", 0.5)),
        decay_lambda=_env_override("LTP_DECAY_LAMBDA", ltp_raw.get("decay_lambda", 0.01)),
        permanence_threshold=_env_override("LTP_PERMANENCE_THRESHOLD", ltp_raw.get("permanence_threshold", 0.95)),
        half_life_days=_env_override("LTP_HALF_LIFE_DAYS", ltp_raw.get("half_life_days", 30.0)),
    )

    # Build attention masking config (Phase 4.1)
    attn_raw = raw.get("attention_masking") or {}
    attention_masking = AttentionMaskingConfig(
        enabled=_env_override("ATTENTION_MASKING_ENABLED", attn_raw.get("enabled", True)),
    )

    # Build consolidation config (Phase 4.0+)
    cons_raw = raw.get("consolidation") or {}
    consolidation = ConsolidationConfig(
        enabled=_env_override("CONSOLIDATION_ENABLED", cons_raw.get("enabled", True)),
        interval_seconds=_env_override("CONSOLIDATION_INTERVAL_SECONDS", cons_raw.get("interval_seconds", 3600)),
        similarity_threshold=_env_override("CONSOLIDATION_SIMILARITY_THRESHOLD", cons_raw.get("similarity_threshold", 0.85)),
        min_cluster_size=_env_override("CONSOLIDATION_MIN_CLUSTER_SIZE", cons_raw.get("min_cluster_size", 2)),
        hot_tier_enabled=_env_override("CONSOLIDATION_HOT_TIER_ENABLED", cons_raw.get("hot_tier_enabled", True)),
        warm_tier_enabled=_env_override("CONSOLIDATION_WARM_TIER_ENABLED", cons_raw.get("warm_tier_enabled", True)),
    )

    # Build dream loop config
    dream_raw = raw.get("dream_loop") or {}
    raw_queue_maxlen = dream_raw.get("subconscious_queue_maxlen")
    env_queue_maxlen = os.environ.get("HAIM_DREAM_LOOP_SUBCONSCIOUS_QUEUE_MAXLEN")
    queue_maxlen = _parse_optional_positive_int(
        env_queue_maxlen if env_queue_maxlen is not None else raw_queue_maxlen
    )
    dream_loop = DreamLoopConfig(
        enabled=_env_override("DREAM_LOOP_ENABLED", dream_raw.get("enabled", True)),
        frequency_seconds=_env_override("DREAM_LOOP_FREQUENCY_SECONDS", dream_raw.get("frequency_seconds", 60)),
        batch_size=_env_override("DREAM_LOOP_BATCH_SIZE", dream_raw.get("batch_size", 10)),
        max_iterations=_env_override("DREAM_LOOP_MAX_ITERATIONS", dream_raw.get("max_iterations", 0)),
        subconscious_queue_maxlen=queue_maxlen,
        ollama_url=_env_override("DREAM_LOOP_OLLAMA_URL", dream_raw.get("ollama_url", "http://localhost:11434/api/generate")),
        model=_env_override("DREAM_LOOP_MODEL", dream_raw.get("model", "gemma3:1b")),
    )

    # Build synapse config (Phase 12.1)
    syn_raw = raw.get("synapse") or {}
    synapse = SynapseConfig(
        similarity_threshold=_env_override("SYNAPSE_SIMILARITY_THRESHOLD", syn_raw.get("similarity_threshold", 0.5)),
        auto_bind_on_store=_env_override("SYNAPSE_AUTO_BIND_ON_STORE", syn_raw.get("auto_bind_on_store", True)),
        multi_hop_depth=_env_override("SYNAPSE_MULTI_HOP_DEPTH", syn_raw.get("multi_hop_depth", 2)),
    )

    # Build context config (Phase 12.2)
    ctx_raw = raw.get("context") or {}
    context = ContextConfig(
        enabled=_env_override("CONTEXT_ENABLED", ctx_raw.get("enabled", True)),
        shift_threshold=_env_override("CONTEXT_SHIFT_THRESHOLD", ctx_raw.get("shift_threshold", 0.3)),
        rolling_window_size=_env_override("CONTEXT_ROLLING_WINDOW_SIZE", ctx_raw.get("rolling_window_size", 5)),
    )

    # Build preference config (Phase 12.3)
    pref_raw = raw.get("preference") or {}
    preference = PreferenceConfig(
        enabled=_env_override("PREFERENCE_ENABLED", pref_raw.get("enabled", True)),
        learning_rate=_env_override("PREFERENCE_LEARNING_RATE", pref_raw.get("learning_rate", 0.1)),
        history_limit=_env_override("PREFERENCE_HISTORY_LIMIT", pref_raw.get("history_limit", 100)),
    )

    # Build anticipatory config (Phase 13.2)
    ant_raw = raw.get("anticipatory") or {}
    anticipatory = AnticipatoryConfig(
        enabled=_env_override("ANTICIPATORY_ENABLED", ant_raw.get("enabled", True)),
        predictive_depth=_env_override("ANTICIPATORY_PREDICTIVE_DEPTH", ant_raw.get("predictive_depth", 1)),
    )

    # Build subconscious AI config (Phase 4.4 BETA)
    sub_raw = raw.get("subconscious_ai") or {}
    subconscious_ai = SubconsciousAIConfig(
        enabled=_env_override("SUBCONSCIOUS_AI_ENABLED", sub_raw.get("enabled", False)),
        beta_mode=_env_override("SUBCONSCIOUS_AI_BETA_MODE", sub_raw.get("beta_mode", True)),
        model_provider=_env_override("SUBCONSCIOUS_AI_MODEL_PROVIDER", sub_raw.get("model_provider", "ollama")),
        model_name=_env_override("SUBCONSCIOUS_AI_MODEL_NAME", sub_raw.get("model_name", "phi3.5:3.8b")),
        model_url=_env_override("SUBCONSCIOUS_AI_MODEL_URL", sub_raw.get("model_url", "http://localhost:11434")),
        api_key=_env_override("SUBCONSCIOUS_AI_API_KEY", sub_raw.get("api_key")),
        api_base_url=_env_override("SUBCONSCIOUS_AI_API_BASE_URL", sub_raw.get("api_base_url")),
        pulse_interval_seconds=_env_override("SUBCONSCIOUS_AI_PULSE_INTERVAL_SECONDS", sub_raw.get("pulse_interval_seconds", 120)),
        pulse_backoff_enabled=_env_override("SUBCONSCIOUS_AI_PULSE_BACKOFF_ENABLED", sub_raw.get("pulse_backoff_enabled", True)),
        pulse_backoff_max_seconds=_env_override("SUBCONSCIOUS_AI_PULSE_BACKOFF_MAX_SECONDS", sub_raw.get("pulse_backoff_max_seconds", 600)),
        max_cpu_percent=_env_override("SUBCONSCIOUS_AI_MAX_CPU_PERCENT", sub_raw.get("max_cpu_percent", 30.0)),
        cycle_timeout_seconds=_env_override("SUBCONSCIOUS_AI_CYCLE_TIMEOUT_SECONDS", sub_raw.get("cycle_timeout_seconds", 30)),
        rate_limit_per_hour=_env_override("SUBCONSCIOUS_AI_RATE_LIMIT_PER_HOUR", sub_raw.get("rate_limit_per_hour", 50)),
        memory_sorting_enabled=_env_override("SUBCONSCIOUS_AI_MEMORY_SORTING_ENABLED", sub_raw.get("memory_sorting_enabled", True)),
        enhanced_dreaming_enabled=_env_override("SUBCONSCIOUS_AI_ENHANCED_DREAMING_ENABLED", sub_raw.get("enhanced_dreaming_enabled", True)),
        micro_self_improvement_enabled=_env_override("SUBCONSCIOUS_AI_MICRO_SELF_IMPROVEMENT_ENABLED", sub_raw.get("micro_self_improvement_enabled", False)),
        dry_run=_env_override("SUBCONSCIOUS_AI_DRY_RUN", sub_raw.get("dry_run", True)),
        log_all_decisions=_env_override("SUBCONSCIOUS_AI_LOG_ALL_DECISIONS", sub_raw.get("log_all_decisions", True)),
        audit_trail_path=_env_override("SUBCONSCIOUS_AI_AUDIT_TRAIL_PATH", sub_raw.get("audit_trail_path", "./data/subconscious_audit.jsonl")),
        max_memories_per_cycle=_env_override("SUBCONSCIOUS_AI_MAX_MEMORIES_PER_CYCLE", sub_raw.get("max_memories_per_cycle", 10)),
    )

    return HAIMConfig(
        version=raw.get("version", "3.0"),
        dimensionality=dimensionality,
        encoding=encoding,
        tiers_hot=_build_tier("hot", hot_raw),
        tiers_warm=_build_tier("warm", warm_raw),
        tiers_cold=_build_tier("cold", cold_raw),
        ltp=ltp,
        hysteresis=hysteresis,
        redis=redis,
        qdrant=qdrant,
        gpu=gpu,
        security=security,
        observability=observability,
        mcp=mcp,
        paths=paths,
        consolidation=consolidation,
        attention_masking=attention_masking,
        synapse=synapse,
        context=context,
        preference=preference,
        anticipatory=anticipatory,
        dream_loop=dream_loop,
        subconscious_ai=subconscious_ai,
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
