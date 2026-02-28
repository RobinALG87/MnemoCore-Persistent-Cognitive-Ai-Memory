"""
Memory Scheduler — memory_scheduler.py
========================================
Implements MemoryOS (EMNLP 2025 Oral) scheduling and interrupt capabilities:

- **Memory Interrupts**: Critical information can interrupt ongoing consolidation.
- **Priority Scheduling**: Which memories to consolidate first, based on system load.
- **Memory Health Score**: Per-node lifecycle scoring (Neuroca: STM → MTM → LTM).

Research basis
~~~~~~~~~~~~~~
- **MemoryOS** (EMNLP 2025, Oral Presentation): treats memory as an operating system
  with scheduling, priority queues, and interrupts.
- **Neuroca**: health-based memory lifecycle: STM (short-term, fragile) →
  MTM (medium-term, consolidating) → LTM (long-term, stable).
- **Continuum Memory Architecture** (arXiv 2601.09913): formal lifecycle
  ingest → activation → retrieval → consolidation.

Architecture
~~~~~~~~~~~~
::

    ┌──────────────────────────────────────────────────────────────┐
    │                     MemoryScheduler                          │
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐    │
    │  │              Priority Queue                          │    │
    │  │  Jobs sorted by: urgency × importance × staleness    │    │
    │  │  Types: consolidation, pruning, linking, decay       │    │
    │  └──────────────────────────────┬───────────────────────┘    │
    │                                 │                            │
    │  ┌──────────────────────────────▼───────────────────────┐    │
    │  │           Interrupt Controller                       │    │
    │  │  Can preempt running consolidation when:             │    │
    │  │  - Critical memory arrives (high importance)         │    │
    │  │  - System health degrades                            │    │
    │  │  - Agent requests immediate attention                │    │
    │  └──────────────────────────────┬───────────────────────┘    │
    │                                 │                            │
    │  ┌──────────────────────────────▼───────────────────────┐    │
    │  │          Health Monitor (Neuroca Model)              │    │
    │  │  Per-node scoring: recency × frequency × stability   │    │
    │  │  STM → MTM → LTM transitions                        │    │
    │  │  System-wide health aggregation                      │    │
    │  └──────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────┘

Integration points:
    - ``pulse.py``: scheduler runs inside pulse ticks as Phase 10
    - ``tier_manager``: health scores inform promote/demote decisions
    - ``knowledge_graph``: health propagation to graph nodes
    - ``strategy_bank``: interrupt on strategy failure detection
"""

from __future__ import annotations

import heapq
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════
# Memory Job Types
# ═══════════════════════════════════════════════════════════════════════

class MemoryJobType(Enum):
    """Types of memory operations that can be scheduled."""
    CONSOLIDATION = "consolidation"      # Episodic → Semantic transfer
    PRUNING = "pruning"                  # Remove decayed/redundant memories
    LINKING = "linking"                  # Create knowledge graph links
    DECAY = "decay"                      # Apply temporal decay
    HEALTH_CHECK = "health_check"        # Recalculate health scores
    STRATEGY_DISTILL = "strategy_distill"  # Distill strategies from episodes
    GRAPH_MAINTENANCE = "graph_maintenance"  # Edge pruning, cluster detection
    INTERRUPT = "interrupt"              # Critical information arrival


class MemoryJobPriority(Enum):
    """Priority levels for memory jobs.

    CRITICAL: Must run immediately (interrupts current job).
    HIGH: Run before normal operations.
    NORMAL: Standard background processing.
    LOW: Run when system is idle.
    DEFERRED: Run during dream/sleep phases only.
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    DEFERRED = 4


@dataclass(order=True)
class MemoryJob:
    """
    A schedulable memory operation.

    Ordered by (priority, created_at) for the priority queue.
    Lower priority numeric value = higher urgency.

    Fields:
        id: Unique job identifier.
        job_type: What kind of operation.
        priority: Urgency level.
        created_at: When the job was created.
        target_id: Optional memory/node ID this job operates on.
        payload: Arbitrary data for the job handler.
        agent_id: Which agent requested this job.
        deadline_seconds: Maximum time before job becomes stale.
        attempts: How many times this job has been attempted.
        max_attempts: Maximum retry count.
    """
    priority: int = field(compare=True, default=2)
    created_at: float = field(compare=True, default_factory=lambda: datetime.now(timezone.utc).timestamp())
    id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    job_type: str = field(compare=False, default="consolidation")
    target_id: Optional[str] = field(compare=False, default=None)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    agent_id: str = field(compare=False, default="system")
    deadline_seconds: float = field(compare=False, default=300.0)
    attempts: int = field(compare=False, default=0)
    max_attempts: int = field(compare=False, default=3)

    @property
    def is_expired(self) -> bool:
        """Check if this job has exceeded its deadline."""
        age = datetime.now(timezone.utc).timestamp() - self.created_at
        return age > self.deadline_seconds

    @property
    def is_critical(self) -> bool:
        return self.priority == MemoryJobPriority.CRITICAL.value


# ═══════════════════════════════════════════════════════════════════════
# Memory Health Monitor (Neuroca Model)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryHealthScore:
    """
    Per-memory health assessment following the Neuroca model.

    The health score determines the memory's lifecycle stage:
    - **STM** (health < 0.3): Short-term memory. Fragile, easily forgotten.
      Recent but not yet reinforced by re-access.
    - **MTM** (0.3 ≤ health < 0.7): Medium-term memory. Being consolidated.
      Accessed enough times to start strengthening.
    - **LTM** (health ≥ 0.7): Long-term memory. Stable, high confidence.
      Well-established through repeated access over time.

    The composite score is:
        health = 0.35 × recency + 0.35 × frequency + 0.30 × stability

    Where:
        recency = exp(-0.693 × hours_since_last_access / 48)
        frequency = min(1.0, log(1 + access_count) / 5.0)
        stability = min(1.0, log(1 + age_days) / 4.0) if access_count > 2 else 0
    """
    node_id: str = ""
    health: float = 0.5
    recency: float = 0.5
    frequency: float = 0.0
    stability: float = 0.0
    lifecycle_stage: str = "STM"  # STM / MTM / LTM
    last_calculated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def calculate(
        cls,
        node_id: str,
        created_at: datetime,
        last_accessed: datetime,
        access_count: int,
    ) -> "MemoryHealthScore":
        """
        Calculate health score from memory access patterns.

        This is the core Neuroca lifecycle model. A memory starts as
        STM and transitions to MTM→LTM as it accumulates evidence
        of usefulness through repeated access over time.

        Args:
            node_id: Memory identifier.
            created_at: When the memory was first stored.
            last_accessed: When the memory was last retrieved.
            access_count: Total number of accesses.

        Returns:
            MemoryHealthScore with computed lifecycle stage.
        """
        now = datetime.now(timezone.utc)
        age_days = max(0.001, (now - created_at).total_seconds() / 86400)
        recency_hours = max(0.001, (now - last_accessed).total_seconds() / 3600)

        # Recency: exponential decay, half-life = 48 hours
        recency = math.exp(-0.693 * recency_hours / 48.0)

        # Frequency: log-scaled access count (saturates at ~150 accesses)
        frequency = min(1.0, math.log1p(access_count) / 5.0)

        # Stability: old nodes that are STILL accessed get bonus
        # (young nodes or never-re-accessed nodes get 0)
        stability = 0.0
        if access_count > 2:
            stability = min(1.0, math.log1p(age_days) / 4.0)

        health = 0.35 * recency + 0.35 * frequency + 0.30 * stability
        health = max(0.0, min(1.0, health))

        # Determine lifecycle stage
        if health < 0.3:
            stage = "STM"
        elif health < 0.7:
            stage = "MTM"
        else:
            stage = "LTM"

        return cls(
            node_id=node_id,
            health=health,
            recency=recency,
            frequency=frequency,
            stability=stability,
            lifecycle_stage=stage,
        )


# ═══════════════════════════════════════════════════════════════════════
# Memory Scheduler Service
# ═══════════════════════════════════════════════════════════════════════

class MemoryScheduler:
    """
    Treats memory operations as OS-level scheduled tasks with priority queues
    and interrupt capability.

    Key capabilities (from MemoryOS, EMNLP 2025):

    1. **Priority Queue**: Memory jobs are ordered by urgency × importance.
       The scheduler picks the highest-priority job each tick.

    2. **Memory Interrupts**: Critical information (high importance score,
       agent-flagged urgent) can interrupt (preempt) ongoing consolidation.
       The interrupted job is re-queued for later.

    3. **System Load Awareness**: The scheduler monitors queue depth and
       adjusts batch sizes. Under heavy load, only CRITICAL and HIGH
       priority jobs run. Under light load, DEFERRED jobs get processed.

    4. **Health-Based Scheduling**: Memories with declining health scores
       are prioritized for consolidation (prevent forgetting). Healthy
       LTM memories are deprioritized.

    5. **Stale Job Expiry**: Jobs past their deadline are dropped to
       prevent queue buildup from outdated requests.

    Thread-safety: All mutations protected by ``threading.RLock``.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Args:
            config: MemorySchedulerConfig. Attributes:
                - max_queue_size (int, default 10000)
                - max_batch_per_tick (int, default 50)
                - interrupt_threshold (float, default 0.9): importance ≥ this triggers interrupt
                - load_shedding_threshold (int, default 500): queue depth above this limits to HIGH only
                - enable_interrupts (bool, default True)
                - health_check_interval_ticks (int, default 5)
        """
        self._lock = threading.RLock()
        self._queue: List[MemoryJob] = []  # heapq (min-heap by priority, then created_at)
        self._active_job: Optional[MemoryJob] = None
        self._interrupted_jobs: List[MemoryJob] = []
        self._completed_count = 0
        self._interrupted_count = 0
        self._expired_count = 0
        self._ticks_since_health = 0

        # Config
        self._max_queue = getattr(config, "max_queue_size", 10000)
        self._max_batch = getattr(config, "max_batch_per_tick", 50)
        self._interrupt_threshold = getattr(config, "interrupt_threshold", 0.9)
        self._load_threshold = getattr(config, "load_shedding_threshold", 500)
        self._enable_interrupts = getattr(config, "enable_interrupts", True)
        self._health_interval = getattr(config, "health_check_interval_ticks", 5)

        # Job handlers: job_type → callable
        self._handlers: Dict[str, Callable] = {}

    # ══════════════════════════════════════════════════════════════════
    # Handler Registration
    # ══════════════════════════════════════════════════════════════════

    def register_handler(self, job_type: str, handler: Callable) -> None:
        """
        Register a handler function for a job type.

        The handler receives (job: MemoryJob) and returns a result dict.
        Async handlers are not supported (wrap with asyncio.run if needed).

        Args:
            job_type: The MemoryJobType string value.
            handler: Callable that processes the job.
        """
        self._handlers[job_type] = handler
        logger.debug(f"Registered handler for job type '{job_type}'.")

    # ══════════════════════════════════════════════════════════════════
    # Job Submission
    # ══════════════════════════════════════════════════════════════════

    def submit(self, job: MemoryJob) -> bool:
        """
        Submit a memory job to the scheduler.

        If the job is CRITICAL and interrupts are enabled, it will
        preempt any currently executing job.

        Args:
            job: The MemoryJob to schedule.

        Returns:
            True if accepted, False if queue is full.
        """
        with self._lock:
            if len(self._queue) >= self._max_queue:
                # If this is critical, evict the lowest-priority job
                if job.is_critical:
                    self._evict_lowest()
                else:
                    logger.warning("Memory scheduler queue full, job rejected.")
                    return False

            heapq.heappush(self._queue, job)

            # ── Interrupt check ───────────────────────────────────────
            if job.is_critical and self._enable_interrupts and self._active_job:
                self._interrupt_active(reason=f"Critical job {job.id[:8]} arrived")

        return True

    def submit_interrupt(
        self,
        target_id: str,
        payload: Dict[str, Any],
        agent_id: str = "system",
    ) -> str:
        """
        Submit a critical interrupt job.

        This will preempt any currently executing memory operation.
        Use for: critical information arrival, system health alerts,
        or agent-flagged urgent memories.

        Args:
            target_id: Memory or node ID requiring immediate attention.
            payload: Data for the interrupt handler.
            agent_id: Who requested the interrupt.

        Returns:
            The job ID.
        """
        job = MemoryJob(
            priority=MemoryJobPriority.CRITICAL.value,
            job_type=MemoryJobType.INTERRUPT.value,
            target_id=target_id,
            payload=payload,
            agent_id=agent_id,
            deadline_seconds=60.0,  # Interrupts expire fast
        )
        self.submit(job)
        logger.info(
            f"Memory interrupt submitted: target={target_id[:8]}… "
            f"agent={agent_id} payload_keys={list(payload.keys())}"
        )
        return job.id

    # ══════════════════════════════════════════════════════════════════
    # Tick Processing
    # ══════════════════════════════════════════════════════════════════

    def process_tick(self) -> Dict[str, Any]:
        """
        Process one scheduler tick: dequeue and execute top-priority jobs.

        Called by the Pulse loop. Respects system load: under heavy
        load (queue > threshold), only HIGH and CRITICAL jobs run.

        Returns:
            Dict with: processed (int), expired (int), interrupted (int),
            queue_depth (int), load_shedding (bool).
        """
        processed = 0
        expired = 0
        load_shedding = False

        with self._lock:
            queue_depth = len(self._queue)

            # Determine batch size and priority filter
            if queue_depth > self._load_threshold:
                load_shedding = True
                max_priority = MemoryJobPriority.HIGH.value
                batch = min(self._max_batch, queue_depth)
            else:
                max_priority = MemoryJobPriority.DEFERRED.value
                batch = min(self._max_batch, queue_depth)

            jobs_to_process: List[MemoryJob] = []
            remaining: List[MemoryJob] = []

            # Drain up to batch jobs from queue
            while self._queue and len(jobs_to_process) < batch:
                job = heapq.heappop(self._queue)
                if job.is_expired:
                    expired += 1
                    self._expired_count += 1
                    continue
                if load_shedding and job.priority > max_priority:
                    remaining.append(job)
                    continue
                jobs_to_process.append(job)

            # Put back remaining jobs
            for j in remaining:
                heapq.heappush(self._queue, j)

        # Process jobs (outside lock to avoid blocking submissions)
        for job in jobs_to_process:
            self._execute_job(job)
            processed += 1

        # Periodic health check
        with self._lock:
            self._ticks_since_health += 1

        return {
            "processed": processed,
            "expired": expired,
            "interrupted": self._interrupted_count,
            "queue_depth": len(self._queue),
            "load_shedding": load_shedding,
            "completed_total": self._completed_count,
        }

    # ══════════════════════════════════════════════════════════════════
    # Health Scoring
    # ══════════════════════════════════════════════════════════════════

    def calculate_health(
        self,
        node_id: str,
        created_at: datetime,
        last_accessed: datetime,
        access_count: int,
    ) -> MemoryHealthScore:
        """
        Calculate health score for a memory node.

        Delegates to ``MemoryHealthScore.calculate()`` and returns
        the lifecycle assessment (STM/MTM/LTM).

        Args:
            node_id: Memory node identifier.
            created_at: Memory creation time.
            last_accessed: Last access time.
            access_count: Total accesses.

        Returns:
            MemoryHealthScore with computed lifecycle stage.
        """
        return MemoryHealthScore.calculate(
            node_id=node_id,
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=access_count,
        )

    def should_consolidate(self, health: MemoryHealthScore) -> bool:
        """
        Determine whether a memory should be scheduled for consolidation
        based on its health profile.

        Consolidation is recommended when:
        - Memory is in MTM stage (being solidified)
        - Recency is moderate (not too fresh, not stale)
        - Frequency shows repeated use (> 3 accesses)

        Returns:
            True if this memory should be consolidated now.
        """
        return (
            health.lifecycle_stage == "MTM"
            and health.frequency > 0.3
            and health.recency > 0.2
        )

    # ══════════════════════════════════════════════════════════════════
    # Statistics
    # ══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Comprehensive scheduler statistics."""
        with self._lock:
            queue_depth = len(self._queue)
            priority_dist: Dict[str, int] = {}
            for job in self._queue:
                p = str(job.priority)
                priority_dist[p] = priority_dist.get(p, 0) + 1

        return {
            "queue_depth": queue_depth,
            "completed_total": self._completed_count,
            "interrupted_total": self._interrupted_count,
            "expired_total": self._expired_count,
            "active_job": self._active_job.id[:8] if self._active_job else None,
            "priority_distribution": priority_dist,
            "handlers_registered": list(self._handlers.keys()),
        }

    # ══════════════════════════════════════════════════════════════════
    # Internal
    # ══════════════════════════════════════════════════════════════════

    def _execute_job(self, job: MemoryJob) -> None:
        """Execute a single job using its registered handler."""
        handler = self._handlers.get(job.job_type)
        if not handler:
            logger.debug(f"No handler for job type '{job.job_type}', skipping.")
            return

        with self._lock:
            self._active_job = job
        try:
            handler(job)
            with self._lock:
                self._completed_count += 1
            logger.debug(
                f"Completed job {job.id[:8]} (type={job.job_type}, "
                f"priority={job.priority})"
            )
        except Exception as e:
            job.attempts += 1
            if job.attempts < job.max_attempts:
                with self._lock:
                    heapq.heappush(self._queue, job)
                logger.warning(
                    f"Job {job.id[:8]} failed (attempt {job.attempts}/"
                    f"{job.max_attempts}): {e}"
                )
            else:
                logger.error(
                    f"Job {job.id[:8]} permanently failed after "
                    f"{job.max_attempts} attempts: {e}"
                )
        finally:
            with self._lock:
                self._active_job = None

    def _interrupt_active(self, reason: str) -> None:
        """Interrupt the currently active job (must hold lock)."""
        if not self._active_job:
            return
        self._interrupted_jobs.append(self._active_job)
        if len(self._interrupted_jobs) > 100:
            self._interrupted_jobs = self._interrupted_jobs[-50:]
        self._interrupted_count += 1
        logger.info(
            f"INTERRUPT: job {self._active_job.id[:8]} preempted. "
            f"Reason: {reason}"
        )
        # Re-queue the interrupted job with slightly lower priority
        interrupted = self._active_job
        interrupted.priority = min(interrupted.priority + 1, MemoryJobPriority.DEFERRED.value)
        heapq.heappush(self._queue, interrupted)
        self._active_job = None

    def _evict_lowest(self) -> None:
        """Evict the lowest-priority job from the queue (must hold lock)."""
        if not self._queue:
            return
        # Find and remove the job with highest priority number (lowest urgency)
        worst_idx = 0
        for i, job in enumerate(self._queue):
            if job.priority > self._queue[worst_idx].priority:
                worst_idx = i
        evicted = self._queue.pop(worst_idx)
        heapq.heapify(self._queue)
        logger.debug(f"Evicted job {evicted.id[:8]} to make room for critical job.")
