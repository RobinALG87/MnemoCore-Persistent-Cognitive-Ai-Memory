"""
HAIM Configuration System
========================
Centralized, validated configuration with environment variable overrides.

Organization:
  Section 1 — Infrastructure  (TierConfig, LTPConfig, HysteresisConfig, RedisConfig,
                         QdrantConfig, GPUConfig, PathsConfig)
  Section 2 — API & Security  (SecurityConfig, ObservabilityConfig, MCPConfig, SearchConfig)
  Section 3 — Encoding & Core (EncodingConfig, AttentionMaskingConfig, ConsolidationConfig,
                         SynapseConfig, ContextConfig, PreferenceConfig, AnticipatoryConfig)
  Section 4 — Subconscious    (DreamLoopConfig, SubconsciousAIConfig, DreamingConfig)
  Section 5 — Performance     (PerformanceConfig, VectorCompressionConfig, BackupConfig)
  Section 6 — Cognitive (Ph5) (WorkingMemoryConfig, EpisodicConfig, SemanticConfig,
                         ProceduralConfig, MetaMemoryConfig, SelfImprovementConfig,
                         PulseConfig)
  Section 7 — Extensions      (EmbeddingRegistryConfig, EFTConfig, WebhookConfig, EventsConfig)
  Section 8 — Root            (HAIMConfig — composes all sections)
  Section 9 — Loader          (load_config, get_config, reset_config)
"""

import os
import threading
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import yaml

from mnemocore.core.exceptions import ConfigurationError


# ═══════════════════════════════════════════════════════════════════════
# §1  Infrastructure
# ═══════════════════════════════════════════════════════════════════════

# Task 4.9: Valid eviction policies for TierConfig
_VALID_EVICTION_POLICIES = frozenset({"lru", "lfu", "fifo", "random"})
_VALID_STORAGE_BACKENDS = frozenset({"memory", "mmap", "filesystem", "qdrant", "redis"})


@dataclass(frozen=True)
class TierConfig:
    max_memories: int
    ltp_threshold_min: float
    eviction_policy: str = "lru"
    consolidation_interval_hours: Optional[int] = None
    storage_backend: str = "memory"
    compression: str = "gzip"
    archive_threshold_days: int = 30

    def __post_init__(self):
        # Task 4.9: Validate TierConfig
        if self.max_memories < 0:
            raise ConfigurationError(
                config_key="max_memories",
                reason=f"max_memories must be non-negative, got {self.max_memories}"
            )
        if not 0.0 <= self.ltp_threshold_min <= 1.0:
            raise ConfigurationError(
                config_key="ltp_threshold_min",
                reason=f"ltp_threshold_min must be between 0 and 1, got {self.ltp_threshold_min}"
            )
        if self.eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ConfigurationError(
                config_key="eviction_policy",
                reason=f"eviction_policy must be one of {sorted(_VALID_EVICTION_POLICIES)}, got '{self.eviction_policy}'"
            )
        if self.storage_backend not in _VALID_STORAGE_BACKENDS:
            raise ConfigurationError(
                config_key="storage_backend",
                reason=f"storage_backend must be one of {sorted(_VALID_STORAGE_BACKENDS)}, got '{self.storage_backend}'"
            )
        if self.archive_threshold_days < 0:
            raise ConfigurationError(
                config_key="archive_threshold_days",
                reason=f"archive_threshold_days must be non-negative, got {self.archive_threshold_days}"
            )


@dataclass(frozen=True)
class LTPConfig:
    initial_importance: float = 0.5
    decay_lambda: float = 0.01
    permanence_threshold: float = 0.95
    half_life_days: float = 30.0

    def __post_init__(self):
        # Task 4.9: Validate LTPConfig
        if not 0.0 <= self.initial_importance <= 1.0:
            raise ConfigurationError(
                config_key="initial_importance",
                reason=f"initial_importance must be between 0 and 1, got {self.initial_importance}"
            )
        if self.decay_lambda < 0:
            raise ConfigurationError(
                config_key="decay_lambda",
                reason=f"decay_lambda must be non-negative, got {self.decay_lambda}"
            )
        if not 0.0 <= self.permanence_threshold <= 1.0:
            raise ConfigurationError(
                config_key="permanence_threshold",
                reason=f"permanence_threshold must be between 0 and 1, got {self.permanence_threshold}"
            )
        if self.half_life_days <= 0:
            raise ConfigurationError(
                config_key="half_life_days",
                reason=f"half_life_days must be positive, got {self.half_life_days}"
            )


@dataclass(frozen=True)
class HysteresisConfig:
    promote_delta: float = 0.15
    demote_delta: float = 0.10

    def __post_init__(self):
        # Task 4.9: Validate HysteresisConfig
        if self.promote_delta < 0:
            raise ConfigurationError(
                config_key="promote_delta",
                reason=f"promote_delta must be non-negative, got {self.promote_delta}"
            )
        if self.demote_delta < 0:
            raise ConfigurationError(
                config_key="demote_delta",
                reason=f"demote_delta must be non-negative, got {self.demote_delta}"
            )


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
class SearchConfig:
    """Configuration for hybrid search (Phase 4.6)."""
    mode: str = "hybrid"  # "dense" | "sparse" | "hybrid"
    hybrid_alpha: float = 0.7  # Weight for dense search in hybrid mode
    rrf_k: int = 60  # RRF constant for rank fusion
    sparse_model: str = "bm25"  # "bm25" or SPLADE model path
    enable_query_expansion: bool = True
    min_dense_score: float = 0.0
    min_sparse_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# §2  API & Security
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SecurityConfig:
    # SECURITY: API key must be set via environment variable HAIM_API_KEY
    # Setting via YAML config is DEPRECATED and will be ignored with a warning
    api_key: Optional[str] = None
    # SECURITY: Default CORS origins are localhost-only for development.
    # Production deployments MUST set explicit origins via HAIM_CORS_ORIGINS env var
    # or cors_origins in config.yaml.
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8100"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    # Trusted proxies for X-Forwarded-For header (for rate limiting)
    # Only trust X-Forwarded-For from these IPs
    trusted_proxies: list[str] = field(default_factory=list)
    # HSTS (HTTP Strict Transport Security)
    # Enable in production with SSL. Disable in development to avoid browser caching issues.
    hsts_enabled: bool = True


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


# ═══════════════════════════════════════════════════════════════════════
# §3  Encoding & Core
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# §4  Subconscious
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# §5  Performance
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PerformanceConfig:
    """Configuration for CPU/resource optimization."""
    background_rebuild_enabled: bool = True
    process_priority_low: bool = True
    vector_cache_enabled: bool = True
    vector_cache_path: Optional[str] = "./data/vector_cache.sqlite"


# ═══════════════════════════════════════════════════════════════════════
# §6  Cognitive (WM, Episodic, Semantic, Procedural, Meta, SI, Pulse)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WorkingMemoryConfig:
    """Configuration for Phase 5: Working Memory Service."""
    max_items_per_agent: int = 20
    default_ttl_seconds: int = 3600
    importance_boost_on_access: float = 0.1
    ttl_refresh_on_promote: int = 1800
    prune_interval_seconds: int = 60


@dataclass(frozen=True)
class EpisodicConfig:
    """Configuration for Phase 5: Episodic Store Service."""
    max_active_episodes_per_agent: int = 5
    max_history_per_agent: int = 500
    auto_chain_episodes: bool = True
    enable_hdv_embedding: bool = True
    tier_persistence_enabled: bool = True


@dataclass(frozen=True)
class SemanticConfig:
    """Configuration for Phase 5: Semantic Store Service."""
    min_similarity_threshold: float = 0.5
    max_local_cache_size: int = 10000
    enable_qdrant_persistence: bool = True
    auto_consolidate_from_episodes: bool = True
    concept_reliability_decay: float = 0.01
    consolidation_min_support: int = 3


@dataclass(frozen=True)
class ProceduralConfig:
    """Configuration for Phase 5: Procedural Store Service."""
    max_procedures: int = 5000
    reliability_boost_on_success: float = 0.05
    reliability_penalty_on_failure: float = 0.10
    min_reliability_threshold: float = 0.1
    enable_semantic_matching: bool = True
    persistence_path: Optional[str] = None


@dataclass(frozen=True)
class MetaMemoryConfig:
    """Configuration for Phase 5: Meta Memory Service."""
    max_metrics_history: int = 10000
    anomaly_failure_rate_threshold: float = 0.10
    anomaly_hit_rate_threshold: float = 0.50
    anomaly_latency_threshold_ms: float = 1000.0
    reflection_interval_ticks: int = 10
    enable_llm_proposals: bool = True


@dataclass(frozen=True)
class SelfImprovementConfig:
    """Configuration for Phase 5.4: Self-Improvement Worker (per SELF_IMPROVEMENT_DEEP_DIVE.md)."""
    enabled: bool = False
    dry_run: bool = True
    safety_mode: str = "strict"  # strict | balanced
    interval_seconds: int = 300
    batch_size: int = 8
    max_cycle_seconds: int = 20
    max_candidates_per_topic: int = 2
    cooldown_minutes: int = 120
    min_improvement_score: float = 0.15
    min_semantic_similarity: float = 0.82
    allow_llm_rewrite: bool = False


@dataclass(frozen=True)
class PulseConfig:
    """Configuration for Phase 5 AGI Pulse Loop orchestrator."""
    enabled: bool = True
    interval_seconds: int = 30
    max_agents_per_tick: int = 50
    max_episodes_per_tick: int = 200


@dataclass(frozen=True)
class StrategyBankConfig:
    """
    Configuration for Closed-Loop Strategy Memory.

    Implements ReasoningBank (arXiv 2502.12110) + A-MEM cycle:
    Retrieve → Execute → Judge → Distill → Store.
    """
    enabled: bool = True
    max_strategies: int = 10000
    max_outcomes_per_strategy: int = 100       # Max outcomes tracked per strategy
    target_negative_ratio: float = 0.4        # Target negative exemplar ratio
    min_confidence_threshold: float = 0.3     # Min confidence to keep strategy
    decay_rate: float = 0.005                 # Per-tick confidence decay
    judge_relevance_weight: float = 0.4
    judge_completeness_weight: float = 0.25
    judge_freshness_weight: float = 0.15
    judge_actionability_weight: float = 0.2
    persistence_path: Optional[str] = None
    auto_persist: bool = True


@dataclass(frozen=True)
class KnowledgeGraphConfig:
    """
    Configuration for Bidirectional Knowledge Graph.

    Implements Mnemosyne (Georgia Tech 2025) / Zettelkasten-style
    self-organizing knowledge graph with dynamic edge weights.
    """
    enabled: bool = True
    max_nodes: int = 50000
    max_edges_per_node: int = 100              # Max outgoing edges per node
    reciprocal_weight_factor: float = 0.7     # B→A weight = A→B × this
    edge_decay_half_life_days: float = 30.0
    activation_decay: float = 0.5             # Activation spreading decay factor
    redundancy_threshold: float = 0.92        # Jaccard threshold for merging
    min_edge_weight: float = 0.05             # Prune edges below this
    persistence_path: Optional[str] = None
    auto_persist: bool = True


@dataclass(frozen=True)
class MemorySchedulerConfig:
    """
    Configuration for MemoryOS-style scheduler (EMNLP 2025).

    Priority-based job scheduling with interrupts, load shedding,
    and Neuroca health scoring (recency + frequency + stability).
    """
    enabled: bool = True
    max_queue_size: int = 10000
    max_batch_per_tick: int = 50               # Jobs processed per tick
    load_shedding_threshold: int = 500
    interrupt_threshold: float = 0.9           # Importance ≥ this triggers interrupt
    enable_interrupts: bool = True
    health_check_interval_ticks: int = 5
    max_retries: int = 3


@dataclass(frozen=True)
class MemoryExchangeConfig:
    """
    Configuration for Multi-Agent Memory Exchange (SAMEP, arXiv 2507).

    Fine-grained access control, semantic discovery across agents,
    and cryptographic provenance for shared memories.
    """
    enabled: bool = False                     # Opt-in: multi-agent is optional
    max_shared_memories: int = 50000
    max_annotations_per_memory: int = 50
    default_access_level: int = 1             # 0=NONE, 1=READ, 2=ANNOTATE, 3=FORK, 4=FULL
    persistence_path: Optional[str] = None
    auto_persist: bool = True


# ═══════════════════════════════════════════════════════════════════════
# §7  Extensions (Embedding Registry, Dreaming, Backup, Vectors, EFT,
#     Webhooks, Events)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EmbeddingRegistryConfig:
    """Configuration for Phase 6.0: Embedding Version Registry."""
    enabled: bool = True
    registry_db_path: Optional[str] = None  # Defaults to data/embedding_registry.sqlite
    auto_migrate: bool = False  # Auto-migrate on model switch
    migration_batch_size: int = 100
    migration_throttle_ms: int = 10
    max_parallel_workers: int = 2
    backup_old_vectors: bool = True  # Enable rollback support
    max_retries: int = 3
    worker_enabled: bool = True  # Background re-embedding worker


@dataclass(frozen=True)
class DreamingConfig:
    """Configuration for Phase 6.0: Dream Scheduler and Pipeline."""
    # Master switch
    enabled: bool = True

    # Idle detection
    idle_threshold_seconds: float = 300.0  # 5 minutes
    min_idle_duration: float = 60.0  # 1 minute
    max_cpu_percent: float = 25.0

    # Scheduling (cron-like)
    schedules: list = field(default_factory=lambda: [
        {"name": "nightly", "cron_expression": "0 2 * * *", "enabled": True}
    ])

    # Session configuration
    session_max_duration_seconds: float = 600.0  # 10 minutes
    session_max_memories: int = 1000

    # Pipeline stages
    enable_episodic_clustering: bool = True
    enable_pattern_extraction: bool = True
    enable_recursive_synthesis: bool = True
    enable_contradiction_resolution: bool = True
    enable_semantic_promotion: bool = True
    enable_dream_report: bool = True

    # Stage-specific
    cluster_time_window_hours: float = 24.0
    pattern_min_frequency: int = 2
    synthesis_max_depth: int = 3
    auto_resolve_contradictions: bool = False
    promotion_ltp_threshold: float = 0.7

    # Reporting
    persist_reports: bool = True
    report_path: str = "./data/dream_reports"
    report_include_memory_details: bool = False


@dataclass(frozen=True)
class BackupConfig:
    """
    Configuration for automated backup and snapshotting.

    Controls Qdrant snapshots, WAL (Write-Ahead Log), and retention policies.
    """
    # Snapshot settings
    enabled: bool = True
    auto_snapshot_enabled: bool = True
    snapshot_interval_hours: int = 24
    max_snapshots: int = 7
    compression_enabled: bool = True

    # WAL settings
    wal_enabled: bool = True
    wal_flush_interval_seconds: int = 300  # 5 minutes
    wal_max_size_mb: int = 100

    # Storage settings
    backup_dir: str = "./backups"
    verify_checksums: bool = True

    # Retention policy
    retention_days: int = 30
    keep_daily: int = 7
    keep_weekly: int = 4
    keep_monthly: int = 12

    # Recovery settings
    restore_timeout_seconds: int = 300
    verify_after_restore: bool = True


@dataclass(frozen=True)
class VectorCompressionConfig:
    """
    Configuration for Phase 6: Vector Compression Layer.

    Controls Product Quantization (PQ) and Scalar Quantization (INT8)
    for memory optimization.
    """
    enabled: bool = True
    pq_n_subvectors: int = 32  # Number of subvectors for PQ
    pq_n_bits: int = 8  # Bits per PQ code (256 centroids)
    int8_threshold_confidence: float = 0.4  # Use INT8 for low-confidence memories
    age_threshold_hours: float = 24.0  # Compress memories older than this
    compression_interval_seconds: int = 3600  # Background compression scan interval
    max_batch_size: int = 1000  # Max vectors to compress per batch
    storage_path: str = "./data/vector_compression.db"
    hot_tier_compression: bool = False  # Keep hot tier uncompressed for speed
    warm_tier_compression: bool = True  # Compress warm tier
    cold_tier_compression: bool = True  # Aggressively compress cold tier


@dataclass(frozen=True)
class EFTConfig:
    """
    Configuration for Phase 7.0: Episodic Future Thinking.

    Controls scenario generation, decay, and attention integration.
    """
    enabled: bool = True

    # Scenario generation
    max_scenarios_per_simulation: int = 5
    min_similarity_threshold: float = 0.55
    temporal_horizon_hours: float = 24.0
    branching_factor: int = 3

    # Decay parameters
    scenario_decay_lambda: float = 0.05
    scenario_half_life_hours: float = 12.0
    min_scenario_confidence: float = 0.1

    # Attention integration
    attention_boost_factor: float = 0.2
    scenario_attention_weight: float = 0.15

    # Storage
    max_stored_scenarios: int = 100
    persist_scenarios: bool = True


@dataclass(frozen=True)
class WebhookRetryConfig:
    """Retry configuration for webhook deliveries."""
    max_attempts: int = 5
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0  # 5 minutes
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class WebhookConfig:
    """
    Configuration for webhook event delivery.

    Controls how MnemoCore delivers events to external HTTP endpoints.
    """
    enabled: bool = True
    persistence_path: str = "./data/webhooks.json"
    max_history_size: int = 10000

    # Default retry configuration
    retry: WebhookRetryConfig = field(default_factory=WebhookRetryConfig)

    # Event type subscriptions for convenience
    # Pre-configured webhooks for common events
    on_consolidation_url: Optional[str] = None
    on_contradiction_url: Optional[str] = None
    on_dream_complete_url: Optional[str] = None
    on_memory_created_url: Optional[str] = None

    # Default headers for all webhooks
    default_headers: dict = field(default_factory=lambda: {
        "User-Agent": "MnemoCore-Webhook/1.0"
    })


@dataclass(frozen=True)
class AssociationsConfig:
    """
    Configuration for Phase 6.0 Association Network.

    Task 4.7: Added to HAIMConfig to replace getattr fallbacks.
    """
    auto_save: bool = True
    decay_enabled: bool = True


@dataclass(frozen=True)
class EventsConfig:
    """
    Configuration for the internal event system.

    Controls EventBus and internal event handling.
    """
    enabled: bool = True
    max_queue_size: int = 10000
    delivery_timeout: float = 30.0
    history_size: int = 1000

    # Which events to enable (empty = all enabled)
    disabled_events: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# §8  Root Composite
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HAIMConfig:
    """Root configuration for the HAIM system."""

    version: str = "4.5"
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
    pulse: PulseConfig = field(default_factory=PulseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    embedding_registry: EmbeddingRegistryConfig = field(default_factory=EmbeddingRegistryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    dreaming: DreamingConfig = field(default_factory=DreamingConfig)
    vector_compression: VectorCompressionConfig = field(default_factory=VectorCompressionConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    eft: EFTConfig = field(default_factory=EFTConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    # Task 4.7: Associations config (was using getattr fallback)
    associations: AssociationsConfig = field(default_factory=AssociationsConfig)
    # Phase 5 Cognitive Services
    working_memory: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    procedural: ProceduralConfig = field(default_factory=ProceduralConfig)
    meta_memory: MetaMemoryConfig = field(default_factory=MetaMemoryConfig)
    self_improvement: SelfImprovementConfig = field(default_factory=SelfImprovementConfig)
    # Phase 6+ Research Services
    strategy_bank: StrategyBankConfig = field(default_factory=StrategyBankConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    memory_scheduler: MemorySchedulerConfig = field(default_factory=MemorySchedulerConfig)
    memory_exchange: MemoryExchangeConfig = field(default_factory=MemoryExchangeConfig)

    def __post_init__(self):
        # Task 4.9: Validate HAIMConfig
        if self.dimensionality <= 0:
            raise ConfigurationError(
                config_key="dimensionality",
                reason=f"dimensionality must be positive, got {self.dimensionality}"
            )
        if self.dimensionality % 64 != 0:
            raise ConfigurationError(
                config_key="dimensionality",
                reason=f"dimensionality must be a multiple of 64 for efficient bit packing, got {self.dimensionality}"
            )


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


# ═══════════════════════════════════════════════════════════════════════
# §9  Loader (YAML + env-override)
# ═══════════════════════════════════════════════════════════════════════

def _load_section(
    raw: dict,
    section_key: str,
    config_cls: type,
    env_prefix: Optional[str] = None,
) -> Any:
    """
    Generic section loader for config dataclasses (Task 4.8).

    This helper reduces repetitive code in load_config() by providing a
    standardized way to load config sections with environment variable overrides.

    Args:
        raw: The raw YAML dictionary.
        section_key: The key to look up in the raw dict (e.g., "encoding").
        config_cls: The dataclass to instantiate (e.g., EncodingConfig).
        env_prefix: Optional env prefix for overrides (e.g., "ENCODING").
                   If None, defaults to section_key.upper().

    Returns:
        An instance of config_cls with values from YAML/ENV/defaults.

    Example:
        encoding = _load_section(raw, "encoding", EncodingConfig, "ENCODING")
    """
    import dataclasses
    from typing import get_type_hints

    section_raw = raw.get(section_key) or {}
    prefix = (env_prefix or section_key).upper()

    # Get field info from the dataclass
    fields = dataclasses.fields(config_cls)
    kwargs = {}

    for f in fields:
        field_name = f.name
        env_key = f"{prefix}_{field_name.upper()}"
        yaml_val = section_raw.get(field_name)

        # Check for env override first
        env_val = os.environ.get(f"HAIM_{env_key}")
        if env_val is not None:
            # Type coercion based on field type
            field_type = f.type
            if hasattr(field_type, '__origin__'):
                field_type = field_type.__origin__
            if field_type == bool or (isinstance(f.default, bool)):
                kwargs[field_name] = env_val.lower() in ("true", "1", "yes")
            elif field_type == int or (isinstance(f.default, int)):
                kwargs[field_name] = int(env_val)
            elif field_type == float or (isinstance(f.default, float)):
                kwargs[field_name] = float(env_val)
            else:
                kwargs[field_name] = env_val
        elif yaml_val is not None:
            kwargs[field_name] = yaml_val
        # If neither env nor yaml, use the default (don't add to kwargs)

    return config_cls(**kwargs)


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
        with open(path, encoding="utf-8") as f:
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

    # SECURITY: API key must be set via environment variable
    # YAML-based API keys are DEPRECATED and will be ignored with a warning
    yaml_api_key = sec_raw.get("api_key")
    env_api_key = os.environ.get("HAIM_API_KEY")

    if yaml_api_key and not env_api_key:
        import warnings
        warnings.warn(
            DeprecationWarning(
                "Setting api_key in config.yaml is DEPRECATED and will be ignored. "
                "Use the HAIM_API_KEY environment variable instead. "
                "This prevents accidental exposure of secrets in version control."
            ),
            stacklevel=2
        )
        from loguru import logger
        logger.warning(
            "SECURITY WARNING: api_key in config.yaml is ignored. "
            "Set the HAIM_API_KEY environment variable instead."
        )

    # Only use environment variable for API key
    effective_api_key = env_api_key

    # Parse CORS origins from env (comma-separated) or config
    cors_env = os.environ.get("HAIM_CORS_ORIGINS")
    if cors_env:
        cors_origins = [o.strip() for o in cors_env.split(",")]
    else:
        # Default to localhost-only origins (not ["*"])
        cors_origins = sec_raw.get("cors_origins", ["http://localhost:3000", "http://localhost:8100"])

    # Parse trusted proxies from env (comma-separated) or config
    trusted_proxies_env = os.environ.get("HAIM_TRUSTED_PROXIES")
    if trusted_proxies_env:
        trusted_proxies = [p.strip() for p in trusted_proxies_env.split(",")]
    else:
        trusted_proxies = sec_raw.get("trusted_proxies", [])

    security = SecurityConfig(
        api_key=effective_api_key,
        cors_origins=cors_origins,
        rate_limit_enabled=_env_override("RATE_LIMIT_ENABLED", sec_raw.get("rate_limit_enabled", True)),
        rate_limit_requests=_env_override("RATE_LIMIT_REQUESTS", sec_raw.get("rate_limit_requests", 100)),
        rate_limit_window=_env_override("RATE_LIMIT_WINDOW", sec_raw.get("rate_limit_window", 60)),
        trusted_proxies=trusted_proxies,
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

    # Build pulse config (Phase 5.0)
    pulse_raw = raw.get("pulse") or {}
    pulse = PulseConfig(
        enabled=_env_override("PULSE_ENABLED", pulse_raw.get("enabled", True)),
        interval_seconds=_env_override("PULSE_INTERVAL_SECONDS", pulse_raw.get("interval_seconds", 30)),
        max_agents_per_tick=_env_override("PULSE_MAX_AGENTS_PER_TICK", pulse_raw.get("max_agents_per_tick", 50)),
        max_episodes_per_tick=_env_override("PULSE_MAX_EPISODES_PER_TICK", pulse_raw.get("max_episodes_per_tick", 200)),
    )

    # Build performance config
    perf_raw = raw.get("performance") or {}
    performance = PerformanceConfig(
        background_rebuild_enabled=_env_override("PERFORMANCE_BACKGROUND_REBUILD_ENABLED", perf_raw.get("background_rebuild_enabled", True)),
        process_priority_low=_env_override("PERFORMANCE_PROCESS_PRIORITY_LOW", perf_raw.get("process_priority_low", True)),
        vector_cache_enabled=_env_override("PERFORMANCE_VECTOR_CACHE_ENABLED", perf_raw.get("vector_cache_enabled", True)),
        vector_cache_path=_env_override("PERFORMANCE_VECTOR_CACHE_PATH", perf_raw.get("vector_cache_path", "./data/vector_cache.sqlite")),
    )

    # Build embedding registry config (Phase 6.0)
    embed_raw = raw.get("embedding_registry") or {}
    embedding_registry = EmbeddingRegistryConfig(
        enabled=_env_override("EMBEDDING_REGISTRY_ENABLED", embed_raw.get("enabled", True)),
        registry_db_path=_env_override("EMBEDDING_REGISTRY_DB_PATH", embed_raw.get("registry_db_path")),
        auto_migrate=_env_override("EMBEDDING_REGISTRY_AUTO_MIGRATE", embed_raw.get("auto_migrate", False)),
        migration_batch_size=_env_override("EMBEDDING_REGISTRY_MIGRATION_BATCH_SIZE", embed_raw.get("migration_batch_size", 100)),
        migration_throttle_ms=_env_override("EMBEDDING_REGISTRY_MIGRATION_THROTTLE_MS", embed_raw.get("migration_throttle_ms", 10)),
        max_parallel_workers=_env_override("EMBEDDING_REGISTRY_MAX_PARALLEL_WORKERS", embed_raw.get("max_parallel_workers", 2)),
        backup_old_vectors=_env_override("EMBEDDING_REGISTRY_BACKUP_OLD_VECTORS", embed_raw.get("backup_old_vectors", True)),
        max_retries=_env_override("EMBEDDING_REGISTRY_MAX_RETRIES", embed_raw.get("max_retries", 3)),
        worker_enabled=_env_override("EMBEDDING_REGISTRY_WORKER_ENABLED", embed_raw.get("worker_enabled", True)),
    )

    # Build search config (Phase 4.6)
    search_raw = raw.get("search") or {}
    search = SearchConfig(
        mode=_env_override("SEARCH_MODE", search_raw.get("mode", "hybrid")),
        hybrid_alpha=_env_override("SEARCH_HYBRID_ALPHA", search_raw.get("hybrid_alpha", 0.7)),
        rrf_k=_env_override("SEARCH_RRF_K", search_raw.get("rrf_k", 60)),
        sparse_model=_env_override("SEARCH_SPARSE_MODEL", search_raw.get("sparse_model", "bm25")),
        enable_query_expansion=_env_override("SEARCH_ENABLE_QUERY_EXPANSION", search_raw.get("enable_query_expansion", True)),
        min_dense_score=_env_override("SEARCH_MIN_DENSE_SCORE", search_raw.get("min_dense_score", 0.0)),
        min_sparse_score=_env_override("SEARCH_MIN_SPARSE_SCORE", search_raw.get("min_sparse_score", 0.0)),
    )

    # Build dreaming config (Phase 6.0)
    dream_raw = raw.get("dreaming") or {}
    session_raw = dream_raw.get("session", {})
    dreaming = DreamingConfig(
        enabled=_env_override("DREAMING_ENABLED", dream_raw.get("enabled", True)),
        idle_threshold_seconds=_env_override("DREAMING_IDLE_THRESHOLD_SECONDS", dream_raw.get("idle_threshold_seconds", 300.0)),
        min_idle_duration=_env_override("DREAMING_MIN_IDLE_DURATION", dream_raw.get("min_idle_duration", 60.0)),
        max_cpu_percent=_env_override("DREAMING_MAX_CPU_PERCENT", dream_raw.get("max_cpu_percent", 25.0)),
        schedules=dream_raw.get("schedules", [{"name": "nightly", "cron_expression": "0 2 * * *", "enabled": True}]),
        session_max_duration_seconds=_env_override("DREAMING_SESSION_MAX_DURATION_SECONDS", session_raw.get("max_duration_seconds", 600.0)),
        session_max_memories=_env_override("DREAMING_SESSION_MAX_MEMORIES", session_raw.get("max_memories_to_process", 1000)),
        enable_episodic_clustering=_env_override("DREAMING_ENABLE_EPISODIC_CLUSTERING", session_raw.get("enable_episodic_clustering", True)),
        enable_pattern_extraction=_env_override("DREAMING_ENABLE_PATTERN_EXTRACTION", session_raw.get("enable_pattern_extraction", True)),
        enable_recursive_synthesis=_env_override("DREAMING_ENABLE_RECURSIVE_SYNTHESIS", session_raw.get("enable_recursive_synthesis", True)),
        enable_contradiction_resolution=_env_override("DREAMING_ENABLE_CONTRADICTION_RESOLUTION", session_raw.get("enable_contradiction_resolution", True)),
        enable_semantic_promotion=_env_override("DREAMING_ENABLE_SEMANTIC_PROMOTION", session_raw.get("enable_semantic_promotion", True)),
        enable_dream_report=_env_override("DREAMING_ENABLE_DREAM_REPORT", session_raw.get("enable_dream_report", True)),
        cluster_time_window_hours=_env_override("DREAMING_CLUSTER_TIME_WINDOW_HOURS", session_raw.get("cluster_time_window_hours", 24.0)),
        pattern_min_frequency=_env_override("DREAMING_PATTERN_MIN_FREQUENCY", session_raw.get("pattern_min_frequency", 2)),
        synthesis_max_depth=_env_override("DREAMING_SYNTHESIS_MAX_DEPTH", session_raw.get("synthesis_max_depth", 3)),
        auto_resolve_contradictions=_env_override("DREAMING_AUTO_RESOLVE_CONTRADICTIONS", session_raw.get("auto_resolve_contradictions", False)),
        promotion_ltp_threshold=_env_override("DREAMING_PROMOTION_LTP_THRESHOLD", session_raw.get("promotion_ltp_threshold", 0.7)),
        persist_reports=_env_override("DREAMING_PERSIST_REPORTS", dream_raw.get("persist_reports", True)),
        report_path=_env_override("DREAMING_REPORT_PATH", dream_raw.get("report_path", "./data/dream_reports")),
        report_include_memory_details=_env_override("DREAMING_REPORT_INCLUDE_MEMORY_DETAILS", dream_raw.get("report_include_memory_details", False)),
    )

    # Build vector compression config (Phase 6)
    vc_raw = raw.get("vector_compression") or {}
    vector_compression = VectorCompressionConfig(
        enabled=_env_override("VECTOR_COMPRESSION_ENABLED", vc_raw.get("enabled", True)),
        pq_n_subvectors=_env_override("VECTOR_COMPRESSION_PQ_N_SUBVECTORS", vc_raw.get("pq_n_subvectors", 32)),
        pq_n_bits=_env_override("VECTOR_COMPRESSION_PQ_N_BITS", vc_raw.get("pq_n_bits", 8)),
        int8_threshold_confidence=_env_override("VECTOR_COMPRESSION_INT8_THRESHOLD_CONFIDENCE", vc_raw.get("int8_threshold_confidence", 0.4)),
        age_threshold_hours=_env_override("VECTOR_COMPRESSION_AGE_THRESHOLD_HOURS", vc_raw.get("age_threshold_hours", 24.0)),
        compression_interval_seconds=_env_override("VECTOR_COMPRESSION_INTERVAL_SECONDS", vc_raw.get("compression_interval_seconds", 3600)),
        max_batch_size=_env_override("VECTOR_COMPRESSION_MAX_BATCH_SIZE", vc_raw.get("max_batch_size", 1000)),
        storage_path=_env_override("VECTOR_COMPRESSION_STORAGE_PATH", vc_raw.get("storage_path", "./data/vector_compression.db")),
        hot_tier_compression=_env_override("VECTOR_COMPRESSION_HOT_TIER", vc_raw.get("hot_tier_compression", False)),
        warm_tier_compression=_env_override("VECTOR_COMPRESSION_WARM_TIER", vc_raw.get("warm_tier_compression", True)),
        cold_tier_compression=_env_override("VECTOR_COMPRESSION_COLD_TIER", vc_raw.get("cold_tier_compression", True)),
    )

    # Build backup config
    backup_raw = raw.get("backup") or {}
    backup = BackupConfig(
        enabled=_env_override("BACKUP_ENABLED", backup_raw.get("enabled", True)),
        auto_snapshot_enabled=_env_override("BACKUP_AUTO_SNAPSHOT_ENABLED", backup_raw.get("auto_snapshot_enabled", True)),
        snapshot_interval_hours=_env_override("BACKUP_SNAPSHOT_INTERVAL_HOURS", backup_raw.get("snapshot_interval_hours", 24)),
        max_snapshots=_env_override("BACKUP_MAX_SNAPSHOTS", backup_raw.get("max_snapshots", 7)),
        compression_enabled=_env_override("BACKUP_COMPRESSION_ENABLED", backup_raw.get("compression_enabled", True)),
        wal_enabled=_env_override("BACKUP_WAL_ENABLED", backup_raw.get("wal_enabled", True)),
        wal_flush_interval_seconds=_env_override("BACKUP_WAL_FLUSH_INTERVAL_SECONDS", backup_raw.get("wal_flush_interval_seconds", 300)),
        wal_max_size_mb=_env_override("BACKUP_WAL_MAX_SIZE_MB", backup_raw.get("wal_max_size_mb", 100)),
        backup_dir=_env_override("BACKUP_DIR", backup_raw.get("backup_dir", "./backups")),
        verify_checksums=_env_override("BACKUP_VERIFY_CHECKSUMS", backup_raw.get("verify_checksums", True)),
        retention_days=_env_override("BACKUP_RETENTION_DAYS", backup_raw.get("retention_days", 30)),
        keep_daily=_env_override("BACKUP_KEEP_DAILY", backup_raw.get("keep_daily", 7)),
        keep_weekly=_env_override("BACKUP_KEEP_WEEKLY", backup_raw.get("keep_weekly", 4)),
        keep_monthly=_env_override("BACKUP_KEEP_MONTHLY", backup_raw.get("keep_monthly", 12)),
        restore_timeout_seconds=_env_override("BACKUP_RESTORE_TIMEOUT_SECONDS", backup_raw.get("restore_timeout_seconds", 300)),
        verify_after_restore=_env_override("BACKUP_VERIFY_AFTER_RESTORE", backup_raw.get("verify_after_restore", True)),
    )

    # Build EFT config (Phase 7.0: Episodic Future Thinking)
    eft_raw = raw.get("eft") or {}
    eft = EFTConfig(
        enabled=_env_override("EFT_ENABLED", eft_raw.get("enabled", True)),
        max_scenarios_per_simulation=_env_override("EFT_MAX_SCENARIOS", eft_raw.get("max_scenarios_per_simulation", 5)),
        min_similarity_threshold=_env_override("EFT_MIN_SIMILARITY", eft_raw.get("min_similarity_threshold", 0.55)),
        temporal_horizon_hours=_env_override("EFT_TEMPORAL_HORIZON", eft_raw.get("temporal_horizon_hours", 24.0)),
        branching_factor=_env_override("EFT_BRANCHING_FACTOR", eft_raw.get("branching_factor", 3)),
        scenario_decay_lambda=_env_override("EFT_DECAY_LAMBDA", eft_raw.get("scenario_decay_lambda", 0.05)),
        scenario_half_life_hours=_env_override("EFT_HALF_LIFE", eft_raw.get("scenario_half_life_hours", 12.0)),
        min_scenario_confidence=_env_override("EFT_MIN_CONFIDENCE", eft_raw.get("min_scenario_confidence", 0.1)),
        attention_boost_factor=_env_override("EFT_ATTENTION_BOOST", eft_raw.get("attention_boost_factor", 0.2)),
        scenario_attention_weight=_env_override("EFT_ATTENTION_WEIGHT", eft_raw.get("scenario_attention_weight", 0.15)),
        max_stored_scenarios=_env_override("EFT_MAX_STORED", eft_raw.get("max_stored_scenarios", 100)),
        persist_scenarios=_env_override("EFT_PERSIST", eft_raw.get("persist_scenarios", True)),
    )

    # Build Phase 5 cognitive service configs
    wm_raw = raw.get("working_memory") or {}
    working_memory_cfg = WorkingMemoryConfig(
        max_items_per_agent=_env_override("WM_MAX_ITEMS", wm_raw.get("max_items_per_agent", 20)),
        default_ttl_seconds=_env_override("WM_DEFAULT_TTL", wm_raw.get("default_ttl_seconds", 3600)),
        importance_boost_on_access=wm_raw.get("importance_boost_on_access", 0.1),
        ttl_refresh_on_promote=wm_raw.get("ttl_refresh_on_promote", 1800),
        prune_interval_seconds=wm_raw.get("prune_interval_seconds", 60),
    )

    ep_raw = raw.get("episodic") or {}
    episodic_cfg = EpisodicConfig(
        max_active_episodes_per_agent=ep_raw.get("max_active_episodes_per_agent", 5),
        max_history_per_agent=ep_raw.get("max_history_per_agent", 500),
        auto_chain_episodes=ep_raw.get("auto_chain_episodes", True),
        enable_hdv_embedding=ep_raw.get("enable_hdv_embedding", True),
        tier_persistence_enabled=_env_override("EPISODIC_TIER_PERSISTENCE", ep_raw.get("tier_persistence_enabled", True)),
    )

    sem_raw = raw.get("semantic") or {}
    semantic_cfg = SemanticConfig(
        min_similarity_threshold=sem_raw.get("min_similarity_threshold", 0.5),
        max_local_cache_size=sem_raw.get("max_local_cache_size", 10000),
        enable_qdrant_persistence=_env_override("SEMANTIC_QDRANT_ENABLED", sem_raw.get("enable_qdrant_persistence", True)),
        auto_consolidate_from_episodes=sem_raw.get("auto_consolidate_from_episodes", True),
        concept_reliability_decay=sem_raw.get("concept_reliability_decay", 0.01),
        consolidation_min_support=sem_raw.get("consolidation_min_support", 3),
    )

    proc_raw = raw.get("procedural") or {}
    procedural_cfg = ProceduralConfig(
        max_procedures=proc_raw.get("max_procedures", 5000),
        reliability_boost_on_success=proc_raw.get("reliability_boost_on_success", 0.05),
        reliability_penalty_on_failure=proc_raw.get("reliability_penalty_on_failure", 0.10),
        min_reliability_threshold=proc_raw.get("min_reliability_threshold", 0.1),
        enable_semantic_matching=proc_raw.get("enable_semantic_matching", True),
        persistence_path=proc_raw.get("persistence_path"),
    )

    mm_raw = raw.get("meta_memory") or {}
    meta_memory_cfg = MetaMemoryConfig(
        max_metrics_history=mm_raw.get("max_metrics_history", 10000),
        anomaly_failure_rate_threshold=mm_raw.get("anomaly_failure_rate_threshold", 0.10),
        anomaly_hit_rate_threshold=mm_raw.get("anomaly_hit_rate_threshold", 0.50),
        anomaly_latency_threshold_ms=mm_raw.get("anomaly_latency_threshold_ms", 1000.0),
        reflection_interval_ticks=mm_raw.get("reflection_interval_ticks", 10),
        enable_llm_proposals=_env_override("META_MEMORY_LLM_PROPOSALS", mm_raw.get("enable_llm_proposals", True)),
    )

    si_raw = raw.get("self_improvement") or {}
    self_improvement_cfg = SelfImprovementConfig(
        enabled=_env_override("SELF_IMPROVEMENT_ENABLED", si_raw.get("enabled", False)),
        dry_run=_env_override("SELF_IMPROVEMENT_DRY_RUN", si_raw.get("dry_run", True)),
        safety_mode=si_raw.get("safety_mode", "strict"),
        interval_seconds=si_raw.get("interval_seconds", 300),
        batch_size=si_raw.get("batch_size", 8),
        max_cycle_seconds=si_raw.get("max_cycle_seconds", 20),
        max_candidates_per_topic=si_raw.get("max_candidates_per_topic", 2),
        cooldown_minutes=si_raw.get("cooldown_minutes", 120),
        min_improvement_score=si_raw.get("min_improvement_score", 0.15),
        min_semantic_similarity=si_raw.get("min_semantic_similarity", 0.82),
        allow_llm_rewrite=_env_override("SELF_IMPROVEMENT_LLM_REWRITE", si_raw.get("allow_llm_rewrite", False)),
    )

    # Build strategy bank config (Closed-Loop Strategy Memory)
    sb_raw = raw.get("strategy_bank") or {}
    strategy_bank_cfg = StrategyBankConfig(
        enabled=_env_override("STRATEGY_BANK_ENABLED", sb_raw.get("enabled", True)),
        max_strategies=sb_raw.get("max_strategies", 10000),
        max_outcomes_per_strategy=sb_raw.get("max_outcomes_per_strategy", 100),
        target_negative_ratio=sb_raw.get("target_negative_ratio", 0.4),
        min_confidence_threshold=sb_raw.get("min_confidence_threshold", 0.3),
        decay_rate=sb_raw.get("decay_rate", 0.005),
        judge_relevance_weight=sb_raw.get("judge_relevance_weight", 0.4),
        judge_completeness_weight=sb_raw.get("judge_completeness_weight", 0.25),
        judge_freshness_weight=sb_raw.get("judge_freshness_weight", 0.15),
        judge_actionability_weight=sb_raw.get("judge_actionability_weight", 0.2),
        persistence_path=sb_raw.get("persistence_path"),
        auto_persist=sb_raw.get("auto_persist", True),
    )

    # Build knowledge graph config (Bidirectional Knowledge Graph)
    kg_raw = raw.get("knowledge_graph") or {}
    knowledge_graph_cfg = KnowledgeGraphConfig(
        enabled=_env_override("KNOWLEDGE_GRAPH_ENABLED", kg_raw.get("enabled", True)),
        max_nodes=kg_raw.get("max_nodes", 50000),
        max_edges_per_node=kg_raw.get("max_edges_per_node", 100),
        reciprocal_weight_factor=kg_raw.get("reciprocal_weight_factor", 0.7),
        edge_decay_half_life_days=kg_raw.get("edge_decay_half_life_days", 30.0),
        activation_decay=kg_raw.get("activation_decay", 0.5),
        redundancy_threshold=kg_raw.get("redundancy_threshold", 0.92),
        min_edge_weight=kg_raw.get("min_edge_weight", 0.05),
        persistence_path=kg_raw.get("persistence_path"),
        auto_persist=kg_raw.get("auto_persist", True),
    )

    # Build memory scheduler config (MemoryOS Scheduler)
    ms_raw = raw.get("memory_scheduler") or {}
    memory_scheduler_cfg = MemorySchedulerConfig(
        enabled=_env_override("MEMORY_SCHEDULER_ENABLED", ms_raw.get("enabled", True)),
        max_queue_size=ms_raw.get("max_queue_size", 10000),
        max_batch_per_tick=ms_raw.get("max_batch_per_tick", 50),
        load_shedding_threshold=ms_raw.get("load_shedding_threshold", 500),
        interrupt_threshold=ms_raw.get("interrupt_threshold", 0.9),
        enable_interrupts=ms_raw.get("enable_interrupts", True),
        health_check_interval_ticks=ms_raw.get("health_check_interval_ticks", 5),
        max_retries=ms_raw.get("max_retries", 3),
    )

    # Build memory exchange config (SAMEP multi-agent protocol)
    me_raw = raw.get("memory_exchange") or {}
    memory_exchange_cfg = MemoryExchangeConfig(
        enabled=_env_override("MEMORY_EXCHANGE_ENABLED", me_raw.get("enabled", False)),
        max_shared_memories=me_raw.get("max_shared_memories", 50000),
        max_annotations_per_memory=me_raw.get("max_annotations_per_memory", 50),
        default_access_level=me_raw.get("default_access_level", 1),
        persistence_path=me_raw.get("persistence_path"),
        auto_persist=me_raw.get("auto_persist", True),
    )

    # Task 4.7: Build associations config (replaces getattr fallbacks)
    assoc_raw = raw.get("associations") or {}
    associations_cfg = AssociationsConfig(
        auto_save=_env_override("ASSOCIATIONS_AUTO_SAVE", assoc_raw.get("auto_save", True)),
        decay_enabled=_env_override("ASSOCIATIONS_DECAY_ENABLED", assoc_raw.get("decay_enabled", True)),
    )

    return HAIMConfig(
        version=raw.get("version", "4.5"),
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
        pulse=pulse,
        performance=performance,
        embedding_registry=embedding_registry,
        search=search,
        dreaming=dreaming,
        vector_compression=vector_compression,
        backup=backup,
        eft=eft,
        # Task 4.7: associations config (replaces getattr fallbacks)
        associations=associations_cfg,
        working_memory=working_memory_cfg,
        episodic=episodic_cfg,
        semantic=semantic_cfg,
        procedural=procedural_cfg,
        meta_memory=meta_memory_cfg,
        self_improvement=self_improvement_cfg,
        strategy_bank=strategy_bank_cfg,
        knowledge_graph=knowledge_graph_cfg,
        memory_scheduler=memory_scheduler_cfg,
        memory_exchange=memory_exchange_cfg,
    )


# Module-level singleton (lazy-loaded) with thread-safe access
_CONFIG: Optional[HAIMConfig] = None
_CONFIG_LOCK = threading.Lock()


def get_config() -> HAIMConfig:
    """
    Get or initialize the global config singleton.

    Thread-safe: Uses threading.Lock to prevent race conditions
    during concurrent first access.
    """
    global _CONFIG
    with _CONFIG_LOCK:
        if _CONFIG is None:
            _CONFIG = load_config()
        return _CONFIG


def reset_config():
    """
    Reset the global config singleton (useful for testing).

    Thread-safe: Uses threading.Lock to prevent race conditions.
    """
    global _CONFIG
    with _CONFIG_LOCK:
        _CONFIG = None
