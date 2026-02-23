"""
Dream Scheduler – Phase 6.0: Idle Detection & Dream Session Management
========================================================================
Manages offline consolidation through idle detection and scheduled dreaming.

This module implements the "sleep" phase of MnemoCore's cognitive cycle:
- Detects periods of inactivity (no user queries, low system load)
- Triggers dream sessions when idle thresholds are met
- Supports cron-like scheduling for recurring consolidation
- Provides graceful shutdown for long-running dream operations

Architecture:
    [IdleDetector] --> [DreamTrigger] --> [DreamSession]
                                       --> [Consolidation Pipeline]
                                       --> [Dream Report]

The scheduler operates as a background daemon that monitors:
1. Query inactivity (time since last memory operation)
2. System resource availability (CPU, memory)
3. Time-based schedules (nightly, hourly, etc.)

Usage:
    scheduler = DreamScheduler(engine, config)
    await scheduler.start()
    # Dream sessions trigger automatically when idle
    await scheduler.stop()
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, time as dt_time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    from ..core.engine import HAIMEngine


# =============================================================================
# Constants
# =============================================================================

DEFAULT_IDLE_THRESHOLD_SECONDS = 300  # 5 minutes of inactivity
DEFAULT_MIN_IDLE_DURATION = 60  # Minimum idle time before dreaming
DEFAULT_MAX_CPU_PERCENT = 25.0  # Don't dream if CPU is busy
DEFAULT_DREAM_SESSION_TIMEOUT = 600  # Max 10 minutes per dream session

# Time-based schedule presets
SCHEDULE_PRESETS = {
    "nightly": "0 2 * * *",  # 2 AM daily
    "hourly": "0 * * * *",   # Top of every hour
    "daily": "0 0 * * *",    # Midnight daily
    "weekly": "0 0 * * 0",   # Sunday midnight
    "never": "",             # Disable scheduled dreaming
}


# =============================================================================
# Enums & Data Classes
# =============================================================================

class DreamTriggerReason(Enum):
    """Why a dream session was triggered."""
    IDLE_TIMEOUT = "idle_timeout"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    LOW_PRIORITY = "low_priority"
    MEMORY_THRESHOLD = "memory_threshold"


@dataclass
class IdleState:
    """Current system idle state."""
    last_activity_time: float
    last_query_time: float
    current_idle_duration: float
    is_idle: bool
    cpu_percent: float
    memory_percent: float
    active_connections: int


@dataclass
class DreamSchedule:
    """A cron-like schedule for dream sessions."""
    name: str
    cron_expression: str  # Simplified cron: "minute hour day month dow"
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


@dataclass
class DreamSessionConfig:
    """Configuration for a single dream session."""
    max_duration_seconds: float = DEFAULT_DREAM_SESSION_TIMEOUT
    max_memories_to_process: int = 1000
    enable_consolidation: bool = True
    enable_contradiction_check: bool = True
    enable_semantic_promotion: bool = True
    enable_recursive_synthesis: bool = True
    tier_filter: Optional[List[str]] = None  # None = all tiers
    min_ltp_threshold: float = 0.0
    generate_report: bool = True


@dataclass
class DreamSessionResult:
    """Result of a completed dream session."""
    session_id: str
    trigger_reason: DreamTriggerReason
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    status: str  # "completed", "timeout", "error", "cancelled"

    # Pipeline metrics
    memories_processed: int = 0
    memories_consolidated: int = 0
    contradictions_found: int = 0
    contradictions_resolved: int = 0
    patterns_extracted: int = 0
    semantic_promotions: int = 0
    recursive_synthesis_calls: int = 0

    # Dream report
    dream_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "trigger_reason": self.trigger_reason.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "status": self.status,
            "memories_processed": self.memories_processed,
            "memories_consolidated": self.memories_consolidated,
            "contradictions_found": self.contradictions_found,
            "contradictions_resolved": self.contradictions_resolved,
            "patterns_extracted": self.patterns_extracted,
            "semantic_promotions": self.semantic_promotions,
            "recursive_synthesis_calls": self.recursive_synthesis_calls,
            "dream_report": self.dream_report,
            "error_message": self.error_message,
        }


@dataclass
class DreamSchedulerConfig:
    """Configuration for the DreamScheduler."""
    # Idle detection
    idle_threshold_seconds: float = DEFAULT_IDLE_THRESHOLD_SECONDS
    min_idle_duration: float = DEFAULT_MIN_IDLE_DURATION
    max_cpu_percent: float = DEFAULT_MAX_CPU_PERCENT
    check_interval_seconds: float = 30.0  # How often to check idle state

    # Scheduling
    schedules: List[DreamSchedule] = field(default_factory=lambda: [
        DreamSchedule(name="nightly", cron_expression=SCHEDULE_PRESETS["nightly"])
    ])

    # Session defaults
    default_session_config: DreamSessionConfig = field(default_factory=DreamSessionConfig)

    # Persistence
    persist_reports: bool = True
    report_path: str = "./data/dream_reports"

    # Safety
    enabled: bool = True
    require_manual_trigger: bool = False  # If True, only manual triggers work


# =============================================================================
# Idle Detector
# =============================================================================

class IdleDetector:
    """
    Monitors system activity to detect idle periods suitable for dreaming.

    Tracks:
    - Time since last memory operation (store/query)
    - System CPU and memory usage
    - Active connection count (if applicable)

    An idle period is when:
    1. No memory operations for > idle_threshold_seconds
    2. CPU usage is below max_cpu_percent
    3. The idle state has persisted for at least min_idle_duration
    """

    def __init__(
        self,
        idle_threshold: float = DEFAULT_IDLE_THRESHOLD_SECONDS,
        min_idle_duration: float = DEFAULT_MIN_IDLE_DURATION,
        max_cpu_percent: float = DEFAULT_MAX_CPU_PERCENT,
    ):
        self.idle_threshold = idle_threshold
        self.min_idle_duration = min_idle_duration
        self.max_cpu_percent = max_cpu_percent

        self._last_activity_time: float = time.monotonic()
        self._last_query_time: float = time.monotonic()
        self._activity_history: deque = deque(maxlen=100)

    def record_activity(self, activity_type: str = "general") -> None:
        """Record a system activity event."""
        now = time.monotonic()
        self._last_activity_time = now

        if activity_type in ("query", "store", "retrieve"):
            self._last_query_time = now

        self._activity_history.append({
            "timestamp": now,
            "type": activity_type,
        })

    def get_idle_state(self) -> IdleState:
        """Get current idle state."""
        now = time.monotonic()
        idle_duration = now - self._last_query_time

        # Get system metrics
        cpu_percent = self._get_cpu_percent()
        memory_percent = self._get_memory_percent()

        is_idle = (
            idle_duration >= self.idle_threshold and
            cpu_percent <= self.max_cpu_percent
        )

        return IdleState(
            last_activity_time=self._last_activity_time,
            last_query_time=self._last_query_time,
            current_idle_duration=idle_duration,
            is_idle=is_idle,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_connections=0,  # Could be enhanced with actual connection tracking
        )

    def should_trigger_dream(self) -> bool:
        """
        Determine if a dream session should be triggered.

        Returns True only if:
        1. Currently idle
        2. Has been idle for at least min_idle_duration
        """
        state = self.get_idle_state()
        return (
            state.is_idle and
            state.current_idle_duration >= self.min_idle_duration
        )

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0  # Assume idle if psutil not available

    def _get_memory_percent(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0


# =============================================================================
# Dream Session
# =============================================================================

class DreamSession:
    """
    A single offline consolidation session.

    Executes the dreaming pipeline:
    1. Episodic clustering
    2. Pattern extraction
    3. Recursive synthesis
    4. Contradiction resolution
    5. Semantic promotion
    6. Dream report generation

    Sessions can be:
    - Triggered automatically (idle detection)
    - Scheduled (cron-like)
    - Manual (explicit trigger)
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        config: Optional[DreamSessionConfig] = None,
        session_id: Optional[str] = None,
    ):
        from uuid import uuid4

        self.engine = engine
        self.config = config or DreamSessionConfig()
        self.session_id = session_id or f"dream_{uuid4().hex[:12]}"

        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._cancelled = False
        self._result: Optional[DreamSessionResult] = None

    async def execute(
        self,
        trigger_reason: DreamTriggerReason = DreamTriggerReason.MANUAL,
    ) -> DreamSessionResult:
        """
        Execute the dream session pipeline.

        Args:
            trigger_reason: Why this session was triggered.

        Returns:
            DreamSessionResult with metrics and report.
        """
        self._started_at = datetime.now(timezone.utc)
        logger.info(f"[DreamSession {self.session_id}] Starting - trigger: {trigger_reason.value}")

        try:
            # Import pipeline components
            from .dream_pipeline import DreamPipeline

            # Create pipeline with timeout
            pipeline = DreamPipeline(self.engine, self.config)

            # Execute with timeout
            result = await asyncio.wait_for(
                pipeline.run(),
                timeout=self.config.max_duration_seconds,
            )

            self._completed_at = datetime.now(timezone.utc)
            duration = (self._completed_at - self._started_at).total_seconds()

            self._result = DreamSessionResult(
                session_id=self.session_id,
                trigger_reason=trigger_reason,
                started_at=self._started_at,
                completed_at=self._completed_at,
                duration_seconds=duration,
                status="completed" if not self._cancelled else "cancelled",
                memories_processed=result.get("memories_processed", 0),
                memories_consolidated=result.get("memories_consolidated", 0),
                contradictions_found=result.get("contradictions_found", 0),
                contradictions_resolved=result.get("contradictions_resolved", 0),
                patterns_extracted=result.get("patterns_extracted", 0),
                semantic_promotions=result.get("semantic_promotions", 0),
                recursive_synthesis_calls=result.get("recursive_synthesis_calls", 0),
                dream_report=result.get("dream_report"),
            )

            logger.info(
                f"[DreamSession {self.session_id}] Completed in {duration:.1f}s - "
                f"processed={self._result.memories_processed} "
                f"consolidated={self._result.memories_consolidated} "
                f"contradictions={self._result.contradictions_found}"
            )

        except asyncio.TimeoutError:
            self._completed_at = datetime.now(timezone.utc)
            duration = (self._completed_at - self._started_at).total_seconds()

            self._result = DreamSessionResult(
                session_id=self.session_id,
                trigger_reason=trigger_reason,
                started_at=self._started_at,
                completed_at=self._completed_at,
                duration_seconds=duration,
                status="timeout",
                error_message=f"Session exceeded max duration of {self.config.max_duration_seconds}s",
            )

            logger.warning(f"[DreamSession {self.session_id}] Timed out after {duration:.1f}s")

        except Exception as e:
            self._completed_at = datetime.now(timezone.utc)
            duration = (self._completed_at - self._started_at).total_seconds()

            self._result = DreamSessionResult(
                session_id=self.session_id,
                trigger_reason=trigger_reason,
                started_at=self._started_at,
                completed_at=self._completed_at,
                duration_seconds=duration,
                status="error",
                error_message=str(e),
            )

            logger.error(f"[DreamSession {self.session_id}] Error: {e}", exc_info=True)

        return self._result

    def cancel(self) -> None:
        """Request cancellation of the current session."""
        self._cancelled = True
        logger.info(f"[DreamSession {self.session_id}] Cancel requested")

    @property
    def result(self) -> Optional[DreamSessionResult]:
        """Get the session result if completed."""
        return self._result

    @property
    def is_running(self) -> bool:
        """Check if session is currently running."""
        return self._started_at is not None and self._completed_at is None


# =============================================================================
# Dream Scheduler
# =============================================================================

class DreamScheduler:
    """
    Orchestrates dream sessions based on idle detection and scheduling.

    The scheduler runs as a background daemon that:
    1. Monitors system activity via IdleDetector
    2. Evaluates cron-like schedules
    3. Triggers DreamSession when conditions are met
    4. Persists dream reports
    5. Provides graceful shutdown

    Integration with SubconsciousAI:
    - Uses SubconsciousAI's model client for LLM-powered synthesis
    - Publishes dream events to the subconscious bus
    - Coordinates with the consolidation worker

    Usage:
        scheduler = DreamScheduler(engine, config)
        await scheduler.start()

        # Record activity (call from API endpoints)
        scheduler.record_activity("query")

        # Manual trigger
        result = await scheduler.trigger_dream()

        await scheduler.stop()
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        config: Optional[DreamSchedulerConfig] = None,
    ):
        self.engine = engine
        self.cfg = config or DreamSchedulerConfig()

        # Components
        self.idle_detector = IdleDetector(
            idle_threshold=self.cfg.idle_threshold_seconds,
            min_idle_duration=self.cfg.min_idle_duration,
            max_cpu_percent=self.cfg.max_cpu_percent,
        )

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._current_session: Optional[DreamSession] = None
        self._session_history: List[DreamSessionResult] = []
        self._max_history = 100

        # Report persistence
        self._report_dir = Path(self.cfg.report_path)
        if self.cfg.persist_reports:
            self._report_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        self._on_dream_complete: Optional[Callable[[DreamSessionResult], None]] = None

        logger.info(
            f"[DreamScheduler] Initialized - "
            f"idle_threshold={self.cfg.idle_threshold}s, "
            f"schedules={len(self.cfg.schedules)}"
        )

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background scheduler loop."""
        if not self.cfg.enabled:
            logger.info("[DreamScheduler] Disabled by configuration")
            return

        if self._running:
            logger.warning("[DreamScheduler] Already running")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._scheduler_loop(),
            name="dream_scheduler"
        )
        logger.info("[DreamScheduler] Started")

    async def stop(self) -> None:
        """Gracefully stop the scheduler and current session."""
        self._running = False

        # Cancel current session if running
        if self._current_session and self._current_session.is_running:
            logger.info("[DreamScheduler] Cancelling current dream session...")
            self._current_session.cancel()

        # Wait for scheduler task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("[DreamScheduler] Stopped")

    # ---------------------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------------------

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Check if we should trigger a dream
                trigger_reason = await self._evaluate_triggers()

                if trigger_reason and not self.cfg.require_manual_trigger:
                    await self._execute_dream_session(trigger_reason)

                # Wait before next check
                await asyncio.sleep(self.cfg.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DreamScheduler] Loop error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Backoff on error

    async def _evaluate_triggers(self) -> Optional[DreamTriggerReason]:
        """Evaluate all trigger conditions."""
        # Check idle detection
        if self.idle_detector.should_trigger_dream():
            return DreamTriggerReason.IDLE_TIMEOUT

        # Check scheduled triggers
        for schedule in self.cfg.schedules:
            if schedule.enabled and self._is_schedule_due(schedule):
                return DreamTriggerReason.SCHEDULED

        return None

    def _is_schedule_due(self, schedule: DreamSchedule) -> bool:
        """Check if a cron schedule is due."""
        if not schedule.cron_expression:
            return False

        now = datetime.now(timezone.utc)

        # Calculate next run if not set
        if schedule.next_run is None:
            schedule.next_run = self._parse_next_run(schedule.cron_expression, now)

        # Check if we've passed the next run time
        if schedule.next_run and now >= schedule.next_run:
            # Update last_run and calculate next
            schedule.last_run = schedule.next_run
            schedule.next_run = self._parse_next_run(schedule.cron_expression, now)
            return True

        return False

    def _parse_next_run(self, cron_expr: str, after: datetime) -> Optional[datetime]:
        """
        Parse a simplified cron expression and return the next run time.

        Format: "minute hour day month dow"
        Supports: "*" for wildcard, numbers, and comma-separated lists.
        """
        try:
            parts = cron_expr.split()
            if len(parts) != 5:
                return None

            minute, hour, day, month, dow = parts

            # Simple implementation: check if current time matches
            # For production, use a proper cron library like croniter
            current_minute = after.minute
            current_hour = after.hour
            current_day = after.day
            current_month = after.month
            current_dow = after.weekday()

            def matches(value: str, current: int) -> bool:
                if value == "*":
                    return True
                if "," in value:
                    return str(current) in value.split(",")
                return int(value) == current

            # If all parts match, next run is tomorrow
            if (
                matches(minute, current_minute) and
                matches(hour, current_hour) and
                matches(day, current_day) and
                matches(month, current_month) and
                matches(dow, current_dow)
            ):
                # Schedule for same time tomorrow
                return after.replace(hour=int(hour) if hour != "*" else 0,
                                     minute=int(minute) if minute != "*" else 0,
                                     second=0, microsecond=0) + timedelta(days=1)

            # Otherwise, schedule for today at the specified time
            target_hour = int(hour) if hour != "*" else current_hour
            target_minute = int(minute) if minute != "*" else current_minute

            next_run = after.replace(
                hour=target_hour,
                minute=target_minute,
                second=0,
                microsecond=0
            )

            # If we've passed that time today, schedule for tomorrow
            if next_run <= after:
                from datetime import timedelta
                next_run += timedelta(days=1)

            return next_run

        except Exception as e:
            logger.debug(f"[DreamScheduler] Failed to parse cron '{cron_expr}': {e}")
            return None

    # ---------------------------------------------------------------------
    # Session Execution
    # ---------------------------------------------------------------------

    async def _execute_dream_session(self, trigger_reason: DreamTriggerReason) -> None:
        """Execute a dream session."""
        if self._current_session and self._current_session.is_running:
            logger.debug("[DreamScheduler] Session already running, skipping")
            return

        logger.info(f"[DreamScheduler] Triggering dream session: {trigger_reason.value}")

        session = DreamSession(self.engine, self.cfg.default_session_config)
        self._current_session = session

        try:
            result = await session.execute(trigger_reason)
            self._session_history.append(result)
            self._session_history = self._session_history[-self._max_history:]

            # Persist report
            if self.cfg.persist_reports:
                self._persist_report(result)

            # Trigger callback
            if self._on_dream_complete:
                try:
                    if asyncio.iscoroutinefunction(self._on_dream_complete):
                        await self._on_dream_complete(result)
                    else:
                        self._on_dream_complete(result)
                except Exception as e:
                    logger.error(f"[DreamScheduler] Callback error: {e}")

        finally:
            self._current_session = None

    def _persist_report(self, result: DreamSessionResult) -> None:
        """Persist dream report to disk."""
        try:
            timestamp = result.started_at.strftime("%Y%m%d_%H%M%S")
            report_path = self._report_dir / f"dream_{result.session_id}_{timestamp}.json"

            from ..utils import json_compat as json
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.debug(f"[DreamScheduler] Report saved: {report_path}")

        except Exception as e:
            logger.error(f"[DreamScheduler] Failed to persist report: {e}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def record_activity(self, activity_type: str = "general") -> None:
        """
        Record system activity.

        Call this from API endpoints when processing queries/stores.
        """
        self.idle_detector.record_activity(activity_type)

    async def trigger_dream(
        self,
        config: Optional[DreamSessionConfig] = None,
    ) -> DreamSessionResult:
        """
        Manually trigger a dream session.

        Args:
            config: Optional session config override.

        Returns:
            DreamSessionResult
        """
        if self._current_session and self._current_session.is_running:
            raise RuntimeError("Dream session already running")

        session_cfg = config or self.cfg.default_session_config
        session = DreamSession(self.engine, session_cfg)
        self._current_session = session

        try:
            result = await session.execute(DreamTriggerReason.MANUAL)
            self._session_history.append(result)

            if self.cfg.persist_reports:
                self._persist_report(result)

            return result

        finally:
            self._current_session = None

    def set_completion_callback(
        self,
        callback: Callable[[DreamSessionResult], None]
    ) -> None:
        """Set a callback to be invoked when a dream session completes."""
        self._on_dream_complete = callback

    def get_idle_state(self) -> IdleState:
        """Get current idle state."""
        return self.idle_detector.get_idle_state()

    def get_recent_sessions(self, limit: int = 10) -> List[DreamSessionResult]:
        """Get recent dream session results."""
        return self._session_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        idle_state = self.idle_detector.get_idle_state()

        return {
            "enabled": self.cfg.enabled,
            "running": self._running,
            "current_session_running": self._current_session is not None,
            "idle_state": {
                "is_idle": idle_state.is_idle,
                "idle_duration_seconds": round(idle_state.current_idle_duration, 1),
                "cpu_percent": idle_state.cpu_percent,
                "memory_percent": idle_state.memory_percent,
            },
            "schedules": [
                {
                    "name": s.name,
                    "enabled": s.enabled,
                    "last_run": s.last_run.isoformat() if s.last_run else None,
                    "next_run": s.next_run.isoformat() if s.next_run else None,
                }
                for s in self.cfg.schedules
            ],
            "recent_sessions_count": len(self._session_history),
            "total_sessions": sum(
                1 for s in self._session_history if s.status == "completed"
            ),
        }


# =============================================================================
# Factory
# =============================================================================

def create_dream_scheduler(
    engine: "HAIMEngine",
    config: Optional["DreamSchedulerConfig"] = None,
) -> DreamScheduler:
    """
    Factory function to create a DreamScheduler from HAIM config.

    Reads dream-related configuration from config.yaml and creates
    an appropriately configured scheduler.
    """
    if config is None:
        from ..core.config import get_config
        haim_cfg = get_config()

        # Build from HAIM config
        cfg = DreamSchedulerConfig(
            idle_threshold_seconds=getattr(
                haim_cfg, "dream_idle_threshold", DEFAULT_IDLE_THRESHOLD_SECONDS
            ),
            min_idle_duration=getattr(
                haim_cfg, "dream_min_idle_duration", DEFAULT_MIN_IDLE_DURATION
            ),
            max_cpu_percent=getattr(
                haim_cfg, "dream_max_cpu_percent", DEFAULT_MAX_CPU_PERCENT
            ),
            schedules=[
                DreamSchedule(
                    name="nightly",
                    cron_expression=SCHEDULE_PRESETS["nightly"]
                )
            ],
            enabled=True,
        )
    else:
        cfg = config

    return DreamScheduler(engine, cfg)
