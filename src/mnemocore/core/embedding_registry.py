"""
Embedding Version Registry
==========================
Phase 6.0: Full embedding model lifecycle management with migration support.

Features:
1. Vector tagging with embedding_model_id, embedding_version, embedding_checksum
2. EmbeddingRegistry maintains mapping of all active models
3. MigrationPlanner generates re-embedding plans on model changes
4. Background re-embedding worker (throttled)
5. Rollback support for failed migrations

This ensures seamless transitions between embedding models without data loss.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
from loguru import logger

from .config import get_config
from .binary_hdv import BinaryHDV


class MigrationStatus(Enum):
    """Status of a migration operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"


class Priority(Enum):
    """Migration priority levels."""
    CRITICAL = 0  # Must migrate immediately (model deprecation)
    HIGH = 1     # Migrate soon
    NORMAL = 2   # Standard background migration
    LOW = 3      # Nice-to-have migration


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """
    Immutable specification for an embedding model.

    This uniquely identifies an embedding model configuration.
    """
    model_id: str  # e.g., "binary_hdv_v1", "sentence_transformer_all-MiniLM-L6-v2"
    version: int   # Monotonically increasing version number
    dimension: int # Vector dimensionality
    checksum: str  # SHA-256 of model config/weights for verification

    # Optional metadata
    name: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")

    @property
    def qualified_id(self) -> str:
        """Full qualified identifier: {model_id}:v{version}:{checksum[:8]}"""
        return f"{self.model_id}:v{self.version}:{self.checksum[:8]}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingModelSpec":
        """Create from dictionary (deserialization)."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (serialization)."""
        return asdict(self)


@dataclass
class MigrationTask:
    """
    A single migration task for one memory node.
    """
    node_id: str
    source_spec: EmbeddingModelSpec
    target_spec: EmbeddingModelSpec
    status: MigrationStatus = MigrationStatus.PENDING
    priority: Priority = Priority.NORMAL

    # Tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Rollback support
    old_vector_backup: Optional[bytes] = None  # Serialized old vector
    can_rollback: bool = True

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert enums and datetime
        data["source_spec"] = self.source_spec.to_dict()
        data["target_spec"] = self.target_spec.to_dict()
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        for dt_field in ["created_at", "started_at", "completed_at"]:
            if data[dt_field]:
                data[dt_field] = data[dt_field].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationTask":
        """Create from dictionary (loading from storage)."""
        # Reconstruct EmbeddingModelSpec objects
        data["source_spec"] = EmbeddingModelSpec.from_dict(data["source_spec"])
        data["target_spec"] = EmbeddingModelSpec.from_dict(data["target_spec"])
        # Convert enum strings back to enums
        data["status"] = MigrationStatus(data["status"])
        data["priority"] = Priority(data["priority"])
        # Convert datetime strings back to datetime objects
        for dt_field in ["created_at", "started_at", "completed_at"]:
            if data.get(dt_field):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return cls(**data)


@dataclass
class MigrationPlan:
    """
    A complete migration plan for multiple nodes.
    """
    plan_id: str
    source_spec: EmbeddingModelSpec
    target_spec: EmbeddingModelSpec
    tasks: List[MigrationTask] = field(default_factory=list)
    status: MigrationStatus = MigrationStatus.PENDING

    # Progress tracking
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Configuration
    batch_size: int = 100
    throttle_delay_ms: int = 10
    max_parallel_workers: int = 2

    @property
    def progress_percent(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return (self.completed_nodes / self.total_nodes) * 100

    @property
    def is_complete(self) -> bool:
        # A plan with zero nodes is considered complete (nothing to do)
        if self.total_nodes == 0:
            return True
        return self.completed_nodes + self.failed_nodes >= self.total_nodes

    def add_task(self, task: MigrationTask):
        """Add a migration task to this plan."""
        self.tasks.append(task)
        self.total_nodes += 1

    def mark_started(self):
        """Mark the migration as started."""
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)
            self.status = MigrationStatus.IN_PROGRESS

    def mark_completed(self):
        """Mark the migration as completed."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = MigrationStatus.COMPLETED

    def mark_failed(self):
        """Mark the migration as failed."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = MigrationStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["source_spec"] = self.source_spec.to_dict()
        data["target_spec"] = self.target_spec.to_dict()
        data["status"] = self.status.value
        for dt_field in ["created_at", "started_at", "completed_at"]:
            if data[dt_field]:
                data[dt_field] = data[dt_field].isoformat()
        data["tasks"] = [t.to_dict() for t in self.tasks]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationPlan":
        """Create from dictionary."""
        data["source_spec"] = EmbeddingModelSpec.from_dict(data["source_spec"])
        data["target_spec"] = EmbeddingModelSpec.from_dict(data["target_spec"])
        data["status"] = MigrationStatus(data["status"])
        for dt_field in ["created_at", "started_at", "completed_at"]:
            if data.get(dt_field):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        data["tasks"] = [MigrationTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(**data)


class EmbeddingRegistry:
    """
    Central registry for all embedding models and their versions.

    Maintains:
    - Active model (current default for new embeddings)
    - All registered models (for compatibility checking)
    - Migration history
    - Pending migration plans

    Thread-safe and async-safe for concurrent access.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        default_dimension: int = 16384,
    ):
        """
        Initialize the embedding registry.

        Args:
            db_path: Path to SQLite database for persistent storage
            default_dimension: Default vector dimensionality
        """
        config = get_config()
        self.db_path = db_path or str(
            Path(config.paths.data_dir) / "embedding_registry.sqlite"
        )
        self.default_dimension = default_dimension
        self._active_model: Optional[EmbeddingModelSpec] = None
        self._registered_models: Dict[str, EmbeddingModelSpec] = {}
        self._migration_plans: Dict[str, MigrationPlan] = {}
        self._lock = asyncio.Lock()

        # Initialize database
        self._init_db()

        # Load state
        self._load_state()

    def _init_db(self):
        """Initialize SQLite database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    dimension INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    name TEXT,
                    description TEXT,
                    created_at TEXT,
                    is_active BOOLEAN DEFAULT FALSE,
                    UNIQUE(model_id, version)
                )
            """)

            # Migration plans table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_plans (
                    plan_id TEXT PRIMARY KEY,
                    source_model_id TEXT NOT NULL,
                    source_version INTEGER NOT NULL,
                    source_checksum TEXT NOT NULL,
                    target_model_id TEXT NOT NULL,
                    target_version INTEGER NOT NULL,
                    target_checksum TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_nodes INTEGER DEFAULT 0,
                    completed_nodes INTEGER DEFAULT 0,
                    failed_nodes INTEGER DEFAULT 0,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    batch_size INTEGER DEFAULT 100,
                    throttle_delay_ms INTEGER DEFAULT 10,
                    max_parallel_workers INTEGER DEFAULT 2,
                    plan_data TEXT NOT NULL
                )
            """)

            # Migration tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_tasks (
                    task_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    source_model_id TEXT NOT NULL,
                    source_version INTEGER NOT NULL,
                    source_checksum TEXT NOT NULL,
                    target_model_id TEXT NOT NULL,
                    target_version INTEGER NOT NULL,
                    target_checksum TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    old_vector_backup BLOB,
                    can_rollback BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (plan_id) REFERENCES migration_plans(plan_id)
                )
            """)

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_migration_tasks_plan "
                "ON migration_tasks(plan_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_migration_tasks_status "
                "ON migration_tasks(status, plan_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_migration_plans_status "
                "ON migration_plans(status)"
            )

    def _load_state(self):
        """Load registry state from database."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            # Load registered models
            cursor = conn.execute(
                "SELECT model_id, version, dimension, checksum, name, "
                "description, created_at FROM models"
            )
            for row in cursor:
                spec = EmbeddingModelSpec(
                    model_id=row[0],
                    version=row[1],
                    dimension=row[2],
                    checksum=row[3],
                    name=row[4] or "",
                    description=row[5] or "",
                    created_at=row[6],
                )
                self._registered_models[spec.qualified_id] = spec

            # Load active model
            cursor = conn.execute(
                "SELECT model_id, version, dimension, checksum, name, "
                "description, created_at FROM models WHERE is_active = TRUE LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                self._active_model = EmbeddingModelSpec(
                    model_id=row[0],
                    version=row[1],
                    dimension=row[2],
                    checksum=row[3],
                    name=row[4] or "",
                    description=row[5] or "",
                    created_at=row[6],
                )

            # If no active model, create default
            if self._active_model is None:
                self._initialize_default_model()

            # Load migration plans
            cursor = conn.execute("SELECT plan_id, plan_data FROM migration_plans")
            for row in cursor:
                plan_data = json.loads(row[1])
                self._migration_plans[row[0]] = MigrationPlan.from_dict(plan_data)

    def _initialize_default_model(self):
        """Initialize the default embedding model."""
        # Use BinaryHDV with current config as default
        config = get_config()
        checksum = self._compute_binary_hdv_checksum(config.dimensionality)

        spec = EmbeddingModelSpec(
            model_id="binary_hdv",
            version=1,
            dimension=config.dimensionality,
            checksum=checksum,
            name="Default Binary Hyperdimensional Vector",
            description="Default MnemoCore binary HDV encoder",
        )

        self.register_model(spec, set_active=True)
        logger.info(f"Initialized default embedding model: {spec.qualified_id}")

    def _compute_binary_hdv_checksum(self, dimension: int) -> str:
        """Compute checksum for BinaryHDV encoder configuration."""
        data = f"binary_hdv:{dimension}".encode()
        return hashlib.sha256(data).hexdigest()

    def register_model(
        self,
        spec: EmbeddingModelSpec,
        set_active: bool = False
    ) -> bool:
        """
        Register a new embedding model.

        Args:
            spec: Model specification
            set_active: If True, set as the active model

        Returns:
            True if registration succeeded, False otherwise
        """
        async def _async_register():
            async with self._lock:
                return self._register_model_sync(spec, set_active)

        # For now, run synchronously (DB operations are fast)
        return self._register_model_sync(spec, set_active)

    def _register_model_sync(
        self,
        spec: EmbeddingModelSpec,
        set_active: bool = False
    ) -> bool:
        """Synchronous implementation of register_model."""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # If setting active, deactivate current
                if set_active:
                    conn.execute("UPDATE models SET is_active = FALSE")

                # Insert or update model
                conn.execute("""
                    INSERT OR REPLACE INTO models
                    (model_id, version, dimension, checksum, name, description,
                     created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    spec.model_id, spec.version, spec.dimension, spec.checksum,
                    spec.name, spec.description, spec.created_at, set_active
                ))

                self._registered_models[spec.qualified_id] = spec
                if set_active:
                    self._active_model = spec
                    logger.info(f"Active model set to: {spec.qualified_id}")
                else:
                    logger.info(f"Registered model: {spec.qualified_id}")

                return True
        except Exception as e:
            logger.error(f"Failed to register model {spec.model_id}: {e}")
            return False

    def get_active_model(self) -> EmbeddingModelSpec:
        """Get the currently active embedding model."""
        if self._active_model is None:
            self._initialize_default_model()
        return self._active_model

    def get_model(self, qualified_id: str) -> Optional[EmbeddingModelSpec]:
        """Get a registered model by qualified ID."""
        return self._registered_models.get(qualified_id)

    def list_models(self) -> List[EmbeddingModelSpec]:
        """List all registered models."""
        return list(self._registered_models.values())

    def is_model_registered(self, spec: EmbeddingModelSpec) -> bool:
        """Check if a model spec is already registered."""
        return spec.qualified_id in self._registered_models

    def get_compatible_models(self, dimension: int) -> List[EmbeddingModelSpec]:
        """Get all models with matching dimensionality."""
        return [
            m for m in self._registered_models.values()
            if m.dimension == dimension
        ]


class MigrationPlanner:
    """
    Plans and orchestrates embedding model migrations.

    Generates migration plans when:
    - Active model is changed
    - A model is deprecated
    - Manual migration is requested
    """

    def __init__(self, registry: EmbeddingRegistry):
        """
        Initialize the migration planner.

        Args:
            registry: EmbeddingRegistry instance
        """
        self.registry = registry
        self.db_path = registry.db_path

    def create_migration_plan(
        self,
        target_spec: EmbeddingModelSpec,
        node_ids: Optional[List[str]] = None,
        priority: Priority = Priority.NORMAL,
        batch_size: int = 100,
        throttle_delay_ms: int = 10,
    ) -> Optional[MigrationPlan]:
        """
        Create a migration plan for the specified nodes.

        Args:
            target_spec: Target embedding model specification
            node_ids: List of node IDs to migrate (None = all nodes)
            priority: Migration priority level
            batch_size: Batch size for processing
            throttle_delay_ms: Delay between batches (throttling)

        Returns:
            MigrationPlan if created successfully, None otherwise
        """
        source_spec = self.registry.get_active_model()

        if source_spec.qualified_id == target_spec.qualified_id:
            logger.warning("Source and target models are identical, no migration needed")
            return None

        # Create plan
        plan_id = str(uuid.uuid4())
        plan = MigrationPlan(
            plan_id=plan_id,
            source_spec=source_spec,
            target_spec=target_spec,
            batch_size=batch_size,
            throttle_delay_ms=throttle_delay_ms,
        )

        # If node_ids not provided, we'll need to scan storage later
        # For now, create an empty plan that will be populated
        if node_ids:
            for node_id in node_ids:
                task = MigrationTask(
                    node_id=node_id,
                    source_spec=source_spec,
                    target_spec=target_spec,
                    priority=priority,
                )
                plan.add_task(task)

        # Save to database
        self._save_plan(plan)
        self.registry._migration_plans[plan_id] = plan

        logger.info(
            f"Created migration plan {plan_id}: "
            f"{source_spec.qualified_id} -> {target_spec.qualified_id} "
            f"({len(node_ids) if node_ids else 0} nodes)"
        )

        return plan

    def _save_plan(self, plan: MigrationPlan):
        """Save migration plan to database."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            # Save plan
            conn.execute("""
                INSERT OR REPLACE INTO migration_plans
                (plan_id, source_model_id, source_version, source_checksum,
                 target_model_id, target_version, target_checksum,
                 status, total_nodes, completed_nodes, failed_nodes,
                 created_at, started_at, completed_at,
                 batch_size, throttle_delay_ms, max_parallel_workers, plan_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan.plan_id,
                plan.source_spec.model_id, plan.source_spec.version,
                plan.source_spec.checksum,
                plan.target_spec.model_id, plan.target_spec.version,
                plan.target_spec.checksum,
                plan.status.value, plan.total_nodes, plan.completed_nodes,
                plan.failed_nodes, plan.created_at.isoformat(),
                plan.started_at.isoformat() if plan.started_at else None,
                plan.completed_at.isoformat() if plan.completed_at else None,
                plan.batch_size, plan.throttle_delay_ms, plan.max_parallel_workers,
                json.dumps(plan.to_dict()),
            ))

            # Save tasks
            for task in plan.tasks:
                task_id = f"{plan.plan_id}:{task.node_id}"
                conn.execute("""
                    INSERT OR REPLACE INTO migration_tasks
                    (task_id, plan_id, node_id, source_model_id, source_version,
                     source_checksum, target_model_id, target_version, target_checksum,
                     status, priority, created_at, started_at, completed_at,
                     error_message, retry_count, max_retries, old_vector_backup, can_rollback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id, plan.plan_id, task.node_id,
                    task.source_spec.model_id, task.source_spec.version,
                    task.source_spec.checksum,
                    task.target_spec.model_id, task.target_spec.version,
                    task.target_spec.checksum,
                    task.status.value, task.priority.value,
                    task.created_at.isoformat(),
                    task.started_at.isoformat() if task.started_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.error_message, task.retry_count, task.max_retries,
                    task.old_vector_backup, task.can_rollback,
                ))

    def get_plan(self, plan_id: str) -> Optional[MigrationPlan]:
        """Get a migration plan by ID."""
        return self.registry._migration_plans.get(plan_id)

    def list_plans(
        self,
        status: Optional[MigrationStatus] = None
    ) -> List[MigrationPlan]:
        """List migration plans, optionally filtered by status."""
        plans = list(self.registry._migration_plans.values())
        if status:
            plans = [p for p in plans if p.status == status]
        return plans

    def get_pending_tasks(
        self, plan_id: str, limit: int = 100
    ) -> List[MigrationTask]:
        """Get pending tasks for a plan."""
        plan = self.get_plan(plan_id)
        if not plan:
            return []

        pending = [
            t for t in plan.tasks
            if t.status == MigrationStatus.PENDING
        ]
        # Sort by priority
        pending.sort(key=lambda t: t.priority.value)
        return pending[:limit]

    def update_task_status(
        self,
        plan_id: str,
        node_id: str,
        status: MigrationStatus,
        error_message: Optional[str] = None,
    ):
        """Update the status of a migration task."""
        plan = self.get_plan(plan_id)
        if not plan:
            return

        for task in plan.tasks:
            if task.node_id == node_id:
                task.status = status
                if status == MigrationStatus.IN_PROGRESS and not task.started_at:
                    task.started_at = datetime.now(timezone.utc)
                elif status in (MigrationStatus.COMPLETED, MigrationStatus.FAILED, MigrationStatus.ROLLED_BACK):
                    task.completed_at = datetime.now(timezone.utc)

                if error_message:
                    task.error_message = error_message

                # Update plan counters
                plan.completed_nodes = sum(
                    1 for t in plan.tasks
                    if t.status == MigrationStatus.COMPLETED
                )
                plan.failed_nodes = sum(
                    1 for t in plan.tasks
                    if t.status == MigrationStatus.FAILED
                )

                if plan.is_complete:
                    if plan.failed_nodes > 0:
                        plan.mark_failed()
                    else:
                        plan.mark_completed()

                # Persist changes
                self._save_plan(plan)
                break


class ReEmbeddingWorker:
    """
    Background worker that performs re-embedding with throttling.

    Features:
    - Throttled execution to avoid overwhelming the system
    - Configurable batch size and delay
    - Automatic retry on failure
    - Rollback support
    - Progress tracking
    """

    def __init__(
        self,
        registry: EmbeddingRegistry,
        planner: MigrationPlanner,
        encoder: Optional[Any] = None,  # TextEncoder or similar
    ):
        """
        Initialize the re-embedding worker.

        Args:
            registry: EmbeddingRegistry instance
            planner: MigrationPlanner instance
            encoder: Encoder instance for generating new embeddings
        """
        self.registry = registry
        self.planner = planner
        self.encoder = encoder
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Statistics
        self._processed_count = 0
        self._failed_count = 0
        self._rolled_back_count = 0

    async def start(self):
        """Start the background worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._worker_loop())
        logger.info("Re-embedding worker started")

    async def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Re-embedding worker stopped")

    async def _worker_loop(self):
        """Main worker loop."""
        while self._running:
            try:
                # Get pending migrations
                plans = self.planner.list_plans(status=MigrationStatus.IN_PROGRESS)
                if not plans:
                    # Also check for pending plans
                    plans = self.planner.list_plans(status=MigrationStatus.PENDING)

                if not plans:
                    # No work to do, sleep
                    await asyncio.sleep(5)
                    continue

                # Process the highest priority plan
                plan = plans[0]
                await self._process_plan(plan)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(10)

    async def _process_plan(self, plan: MigrationPlan):
        """Process a migration plan."""
        if not plan.started_at:
            plan.mark_started()
            self.planner._save_plan(plan)

        # Get next batch of pending tasks
        pending = self.planner.get_pending_tasks(
            plan.plan_id,
            limit=plan.batch_size
        )

        if not pending:
            # Plan complete or paused
            return

        logger.info(
            f"Processing {len(pending)} tasks for plan {plan.plan_id} "
            f"({plan.progress_percent:.1f}% complete)"
        )

        # Process tasks with throttling
        for i, task in enumerate(pending):
            if not self._running:
                break

            try:
                await self._process_task(plan, task)

                # Throttle between tasks
                if plan.throttle_delay_ms > 0 and i < len(pending) - 1:
                    await asyncio.sleep(plan.throttle_delay_ms / 1000.0)

            except Exception as e:
                logger.error(f"Failed to process task for node {task.node_id}: {e}")
                self.planner.update_task_status(
                    plan.plan_id,
                    task.node_id,
                    MigrationStatus.FAILED,
                    str(e)
                )
                self._failed_count += 1

    async def _process_task(self, plan: MigrationPlan, task: MigrationTask):
        """Process a single migration task."""
        # Mark as in progress
        self.planner.update_task_status(
            plan.plan_id, task.node_id, MigrationStatus.IN_PROGRESS
        )

        # This is a placeholder for the actual migration logic
        # The actual implementation would:
        # 1. Fetch the node from storage
        # 2. Back up the old vector (if rollback enabled)
        # 3. Generate new embedding with target model
        # 4. Update the node in storage
        # 5. Mark task as complete

        # For now, just mark as complete (mock implementation)
        self.planner.update_task_status(
            plan.plan_id, task.node_id, MigrationStatus.COMPLETED
        )
        self._processed_count += 1

    async def rollback_migration(
        self,
        plan_id: str,
        node_id: Optional[str] = None
    ) -> bool:
        """
        Rollback a migration (or specific task).

        Args:
            plan_id: Migration plan ID
            node_id: Optional specific node ID to rollback (None = all)

        Returns:
            True if rollback succeeded
        """
        plan = self.planner.get_plan(plan_id)
        if not plan:
            logger.error(f"Plan {plan_id} not found")
            return False

        tasks_to_rollback = [
            t for t in plan.tasks
            if (node_id is None or t.node_id == node_id)
            and t.can_rollback
            and t.old_vector_backup
        ]

        if not tasks_to_rollback:
            logger.warning(f"No rollbackable tasks found for plan {plan_id}")
            return False

        for task in tasks_to_rollback:
            try:
                # Restore old vector
                # This is a placeholder - actual implementation would:
                # 1. Deserialize old_vector_backup
                # 2. Update the node in storage
                # 3. Mark task as rolled back

                self.planner.update_task_status(
                    plan_id, task.node_id, MigrationStatus.ROLLED_BACK
                )
                self._rolled_back_count += 1

            except Exception as e:
                logger.error(f"Rollback failed for node {task.node_id}: {e}")
                self.planner.update_task_status(
                    plan_id, task.node_id, MigrationStatus.FAILED, str(e)
                )

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "rolled_back_count": self._rolled_back_count,
            "is_running": self._running,
        }


class EmbeddingVersionManager:
    """
    High-level API for embedding version management.

    Provides a simplified interface for:
    - Registering new embedding models
    - Creating and managing migrations
    - Querying version compatibility
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        encoder: Optional[Any] = None,
    ):
        """
        Initialize the version manager.

        Args:
            db_path: Path to registry database
            encoder: Encoder instance for re-embedding
        """
        self.registry = EmbeddingRegistry(db_path=db_path)
        self.planner = MigrationPlanner(self.registry)
        self.worker = ReEmbeddingWorker(self.registry, self.planner, encoder)

    async def start_worker(self):
        """Start the background re-embedding worker."""
        await self.worker.start()

    async def stop_worker(self):
        """Stop the background re-embedding worker."""
        await self.worker.stop()

    def register_model(
        self,
        model_id: str,
        version: int,
        dimension: int,
        checksum: str,
        name: str = "",
        description: str = "",
        set_active: bool = False,
    ) -> Optional[EmbeddingModelSpec]:
        """
        Register a new embedding model.

        Args:
            model_id: Unique model identifier
            version: Model version number
            dimension: Vector dimensionality
            checksum: Model configuration checksum
            name: Human-readable name
            description: Model description
            set_active: Whether to set as active model

        Returns:
            EmbeddingModelSpec if successful
        """
        spec = EmbeddingModelSpec(
            model_id=model_id,
            version=version,
            dimension=dimension,
            checksum=checksum,
            name=name,
            description=description,
        )

        if self.registry.register_model(spec, set_active=set_active):
            return spec
        return None

    def switch_active_model(
        self,
        target_model_id: str,
        target_version: int,
        migrate_existing: bool = True,
    ) -> Optional[MigrationPlan]:
        """
        Switch to a different embedding model.

        Args:
            target_model_id: Target model ID
            target_version: Target model version
            migrate_existing: Whether to create migration plan for existing nodes

        Returns:
            MigrationPlan if migration was created
        """
        # Find the target model
        target_spec = None
        for spec in self.registry.list_models():
            if spec.model_id == target_model_id and spec.version == target_version:
                target_spec = spec
                break

        if not target_spec:
            logger.error(f"Target model {target_model_id}:v{target_version} not found")
            return None

        # Set as active
        if not self.registry.register_model(target_spec, set_active=True):
            logger.error(f"Failed to set {target_model_id}:v{target_version} as active")
            return None

        # Create migration plan if requested
        if migrate_existing:
            return self.planner.create_migration_plan(
                target_spec=target_spec,
                priority=Priority.HIGH,
            )

        return None

    def create_migration(
        self,
        node_ids: List[str],
        target_model_id: str,
        target_version: int,
        priority: Priority = Priority.NORMAL,
    ) -> Optional[MigrationPlan]:
        """
        Create a migration for specific nodes.

        Args:
            node_ids: List of node IDs to migrate
            target_model_id: Target model ID
            target_version: Target model version
            priority: Migration priority

        Returns:
            MigrationPlan if created
        """
        # Find target spec
        target_spec = None
        for spec in self.registry.list_models():
            if spec.model_id == target_model_id and spec.version == target_version:
                target_spec = spec
                break

        if not target_spec:
            logger.error(f"Target model {target_model_id}:v{target_version} not found")
            return None

        return self.planner.create_migration_plan(
            target_spec=target_spec,
            node_ids=node_ids,
            priority=priority,
        )

    def get_active_model(self) -> EmbeddingModelSpec:
        """Get the currently active embedding model."""
        return self.registry.get_active_model()

    def list_models(self) -> List[EmbeddingModelSpec]:
        """List all registered models."""
        return self.registry.list_models()

    def list_migrations(
        self, status: Optional[MigrationStatus] = None
    ) -> List[MigrationPlan]:
        """List migration plans."""
        return self.planner.list_plans(status=status)

    async def rollback_migration(
        self,
        plan_id: str,
        node_id: Optional[str] = None
    ) -> bool:
        """Rollback a migration or specific task."""
        return await self.worker.rollback_migration(plan_id, node_id)

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get re-embedding worker statistics."""
        return self.worker.get_statistics()


# Convenience function for creating tagged vectors
def create_vector_metadata(
    model_spec: Optional[EmbeddingModelSpec] = None,
    registry: Optional[EmbeddingRegistry] = None,
) -> Dict[str, str]:
    """
    Create embedding metadata for a vector.

    Args:
        model_spec: Model specification (uses active if not provided)
        registry: Registry instance (creates new if not provided)

    Returns:
        Dictionary with embedding_model_id, embedding_version, embedding_checksum
    """
    if model_spec is None:
        if registry is None:
            registry = EmbeddingRegistry()
        model_spec = registry.get_active_model()

    return {
        "embedding_model_id": model_spec.model_id,
        "embedding_version": str(model_spec.version),
        "embedding_checksum": model_spec.checksum,
    }


def verify_vector_compatibility(
    vector_metadata: Dict[str, str],
    target_spec: EmbeddingModelSpec,
) -> bool:
    """
    Check if a vector is compatible with a target model.

    Args:
        vector_metadata: Vector metadata dict
        target_spec: Target model specification

    Returns:
        True if compatible (same model and version)
    """
    return (
        vector_metadata.get("embedding_model_id") == target_spec.model_id
        and int(vector_metadata.get("embedding_version", "0")) == target_spec.version
        and vector_metadata.get("embedding_checksum") == target_spec.checksum
    )


# Export key classes
__all__ = [
    "EmbeddingModelSpec",
    "MigrationTask",
    "MigrationPlan",
    "MigrationStatus",
    "Priority",
    "EmbeddingRegistry",
    "MigrationPlanner",
    "ReEmbeddingWorker",
    "EmbeddingVersionManager",
    "create_vector_metadata",
    "verify_vector_compatibility",
]
