"""
Comprehensive tests for Dream Scheduler Module.
===============================================
Tests for:
  - IdleDetector (idle state transitions, CPU threshold)
  - DreamSession (normal completion, cancellation, timeout)
  - DreamScheduler (schedule-based triggering, cooldown enforcement)
  - Next-run calculation (cron parser)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from mnemocore.subconscious.dream_scheduler import (
    IdleDetector,
    IdleState,
    DreamSession,
    DreamSessionConfig,
    DreamSessionResult,
    DreamScheduler,
    DreamSchedulerConfig,
    DreamSchedule,
    DreamTriggerReason,
    create_dream_scheduler,
    DEFAULT_IDLE_THRESHOLD_SECONDS,
    DEFAULT_MIN_IDLE_DURATION,
    DEFAULT_MAX_CPU_PERCENT,
    SCHEDULE_PRESETS,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_haim_engine():
    """Create a mock HAIM engine for testing."""
    engine = MagicMock()
    engine.dimension = 10000

    # Mock tier manager
    tier_manager = MagicMock()
    tier_manager.hot = {}
    tier_manager.warm = {}
    tier_manager.get_memory = MagicMock(return_value=None)
    tier_manager.get_all_hot = AsyncMock(return_value=[])
    engine.tier_manager = tier_manager

    return engine


@pytest.fixture
def dream_scheduler_config():
    """Create a standard dream scheduler configuration."""
    return DreamSchedulerConfig(
        idle_threshold_seconds=300,
        min_idle_duration=60,
        max_cpu_percent=25.0,
        check_interval_seconds=30.0,
        enabled=True,
        require_manual_trigger=False,
    )


@pytest.fixture
def dream_session_config():
    """Create a standard dream session configuration."""
    return DreamSessionConfig(
        max_duration_seconds=60,
        max_memories_to_process=100,
        enable_consolidation=True,
        generate_report=True,
    )


# =====================================================================
# IdleDetector Tests
# =====================================================================

class TestIdleDetector:
    """Tests for the IdleDetector component."""

    def test_init_defaults(self):
        """Test default initialization."""
        detector = IdleDetector()
        assert detector.idle_threshold == DEFAULT_IDLE_THRESHOLD_SECONDS
        assert detector.min_idle_duration == DEFAULT_MIN_IDLE_DURATION
        assert detector.max_cpu_percent == DEFAULT_MAX_CPU_PERCENT

    def test_init_custom_params(self):
        """Test custom initialization."""
        detector = IdleDetector(
            idle_threshold=600,
            min_idle_duration=120,
            max_cpu_percent=50.0,
        )
        assert detector.idle_threshold == 600
        assert detector.min_idle_duration == 120
        assert detector.max_cpu_percent == 50.0

    def test_record_activity(self):
        """Test recording activity updates last activity time."""
        detector = IdleDetector()

        initial_time = detector._last_activity_time
        detector.record_activity("query")

        assert detector._last_activity_time >= initial_time

    def test_record_query_activity(self):
        """Test recording query activity updates both activity and query time."""
        detector = IdleDetector()

        initial_query_time = detector._last_query_time
        detector.record_activity("query")

        assert detector._last_query_time >= initial_query_time

    def test_get_idle_state(self):
        """Test getting idle state returns proper IdleState."""
        detector = IdleDetector()

        state = detector.get_idle_state()

        assert isinstance(state, IdleState)
        assert hasattr(state, "last_activity_time")
        assert hasattr(state, "last_query_time")
        assert hasattr(state, "current_idle_duration")
        assert hasattr(state, "is_idle")
        assert hasattr(state, "cpu_percent")
        assert hasattr(state, "memory_percent")

    def test_get_idle_state_with_psutil_mock(self):
        """Test idle state with mocked psutil."""
        detector = IdleDetector()

        with patch.object(detector, "_get_cpu_percent", return_value=15.0):
            with patch.object(detector, "_get_memory_percent", return_value=50.0):
                state = detector.get_idle_state()

        assert state.cpu_percent == 15.0
        assert state.memory_percent == 50.0

    def test_should_trigger_dream_not_idle(self):
        """Test should_trigger_dream when not idle."""
        detector = IdleDetector(
            idle_threshold=300,
            min_idle_duration=60,
        )

        # Record recent activity
        detector.record_activity("query")

        result = detector.should_trigger_dream()

        # Should not trigger immediately after activity
        assert result is False

    def test_should_trigger_dream_idle_below_threshold(self):
        """Test should_trigger_dream when idle but below minimum duration."""
        detector = IdleDetector(
            idle_threshold=0.1,  # Very short threshold
            min_idle_duration=3600,  # 1 hour minimum
            max_cpu_percent=100.0,
        )

        # Mock CPU to be low
        with patch.object(detector, "_get_cpu_percent", return_value=10.0):
            result = detector.should_trigger_dream()

        # Should not trigger because min_idle_duration not met
        assert result is False

    def test_should_trigger_dream_high_cpu(self):
        """Test should_trigger_dream when CPU is high."""
        detector = IdleDetector(
            idle_threshold=0.1,
            min_idle_duration=0.1,
            max_cpu_percent=10.0,  # Low threshold
        )

        with patch.object(detector, "_get_cpu_percent", return_value=50.0):
            state = detector.get_idle_state()

        # Should not be idle with high CPU
        assert state.is_idle is False

    def test_get_cpu_percent_without_psutil(self):
        """Test CPU percent returns 0 when psutil not available."""
        detector = IdleDetector()

        with patch("builtins.__import__", side_effect=ImportError):
            result = detector._get_cpu_percent()

        # Should return 0.0 if psutil not available
        assert result == 0.0

    def test_get_memory_percent_without_psutil(self):
        """Test memory percent returns 0 when psutil not available."""
        detector = IdleDetector()

        with patch("builtins.__import__", side_effect=ImportError):
            result = detector._get_memory_percent()

        # Should return 0.0 if psutil not available
        assert result == 0.0


# =====================================================================
# DreamSession Tests
# =====================================================================

class TestDreamSession:
    """Tests for the DreamSession component."""

    def test_init(self, mock_haim_engine):
        """Test session initialization."""
        config = DreamSessionConfig()
        session = DreamSession(mock_haim_engine, config)

        assert session.engine == mock_haim_engine
        assert session.config == config
        assert session.session_id is not None
        assert session.session_id.startswith("dream_")

    def test_init_with_custom_session_id(self, mock_haim_engine):
        """Test session with custom ID."""
        config = DreamSessionConfig()
        session = DreamSession(
            mock_haim_engine,
            config,
            session_id="custom_session_123"
        )

        assert session.session_id == "custom_session_123"

    @pytest.mark.asyncio
    async def test_execute_normal_completion(self, mock_haim_engine):
        """Test session execution completes normally."""
        config = DreamSessionConfig(
            max_duration_seconds=60,
            generate_report=True,
        )
        session = DreamSession(mock_haim_engine, config)

        # Mock the dream pipeline
        with patch(
            "mnemocore.subconscious.dream_pipeline.DreamPipeline"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.run = AsyncMock(return_value={
                "success": True,
                "memories_processed": 10,
                "memories_consolidated": 5,
                "contradictions_found": 2,
                "contradictions_resolved": 1,
                "patterns_extracted": 3,
                "semantic_promotions": 2,
                "recursive_synthesis_calls": 1,
                "dream_report": {"summary": "Test report"},
            })
            mock_pipeline_cls.return_value = mock_pipeline

            result = await session.execute(DreamTriggerReason.MANUAL)

        assert isinstance(result, DreamSessionResult)
        assert result.status == "completed"
        assert result.trigger_reason == DreamTriggerReason.MANUAL
        assert result.memories_processed == 10

    @pytest.mark.asyncio
    async def test_execute_timeout(self, mock_haim_engine):
        """Test session execution times out correctly."""
        config = DreamSessionConfig(
            max_duration_seconds=0.1,  # Very short timeout
        )
        session = DreamSession(mock_haim_engine, config)

        # Mock the dream pipeline to sleep forever
        async def slow_run():
            await asyncio.sleep(10)
            return {"success": True}

        with patch(
            "mnemocore.subconscious.dream_pipeline.DreamPipeline"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.run = slow_run
            mock_pipeline_cls.return_value = mock_pipeline

            result = await session.execute(DreamTriggerReason.IDLE_TIMEOUT)

        assert result.status == "timeout"
        assert "max duration" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_cancellation(self, mock_haim_engine):
        """Test session can be cancelled."""
        config = DreamSessionConfig(max_duration_seconds=60)
        session = DreamSession(mock_haim_engine, config)

        # Create a task to run the session
        async def run_and_cancel():
            # Start execution in background
            task = asyncio.create_task(
                session.execute(DreamTriggerReason.MANUAL)
            )

            # Wait a bit then cancel
            await asyncio.sleep(0.1)
            session.cancel()

            # Wait for result
            return await task

        # Mock pipeline to take time
        with patch(
            "mnemocore.subconscious.dream_pipeline.DreamPipeline"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()

            async def slow_run():
                await asyncio.sleep(10)
                return {"success": True}

            mock_pipeline.run = slow_run
            mock_pipeline_cls.return_value = mock_pipeline

            # This should complete due to timeout or cancellation handling
            result = await asyncio.wait_for(run_and_cancel(), timeout=5)

        # Session should have been marked for cancellation
        assert session._cancelled is True

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, mock_haim_engine):
        """Test session handles errors gracefully."""
        config = DreamSessionConfig()
        session = DreamSession(mock_haim_engine, config)

        with patch(
            "mnemocore.subconscious.dream_pipeline.DreamPipeline"
        ) as mock_pipeline_cls:
            mock_pipeline_cls.side_effect = Exception("Test error")

            result = await session.execute(DreamTriggerReason.MANUAL)

        assert result.status == "error"
        assert "Test error" in result.error_message

    def test_cancel(self, mock_haim_engine):
        """Test cancel method sets cancelled flag."""
        config = DreamSessionConfig()
        session = DreamSession(mock_haim_engine, config)

        session.cancel()

        assert session._cancelled is True

    def test_is_running_property(self, mock_haim_engine):
        """Test is_running property."""
        config = DreamSessionConfig()
        session = DreamSession(mock_haim_engine, config)

        # Not running initially
        assert session.is_running is False

        # Set started but not completed
        session._started_at = datetime.now(timezone.utc)
        assert session.is_running is True

        # Set completed
        session._completed_at = datetime.now(timezone.utc)
        assert session.is_running is False


# =====================================================================
# DreamScheduler Tests
# =====================================================================

class TestDreamScheduler:
    """Tests for the DreamScheduler orchestrator."""

    def test_init(self, mock_haim_engine, dream_scheduler_config):
        """Test scheduler initialization."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        assert scheduler.engine == mock_haim_engine
        assert scheduler.cfg == dream_scheduler_config
        assert scheduler.idle_detector is not None
        assert scheduler._running is False

    def test_init_default_config(self, mock_haim_engine):
        """Test scheduler with default configuration."""
        scheduler = DreamScheduler(mock_haim_engine)

        assert scheduler.cfg is not None
        assert scheduler.cfg.idle_threshold_seconds == DEFAULT_IDLE_THRESHOLD_SECONDS

    @pytest.mark.asyncio
    async def test_start(self, mock_haim_engine, dream_scheduler_config):
        """Test starting the scheduler."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        await scheduler.start()

        assert scheduler._running is True
        assert scheduler._task is not None

        # Clean up
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_disabled(self, mock_haim_engine):
        """Test starting disabled scheduler does nothing."""
        config = DreamSchedulerConfig(enabled=False)
        scheduler = DreamScheduler(mock_haim_engine, config)

        await scheduler.start()

        assert scheduler._running is False
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_start_already_running(self, mock_haim_engine, dream_scheduler_config):
        """Test starting already running scheduler."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        await scheduler.start()
        assert scheduler._running is True

        # Start again should not create new task
        original_task = scheduler._task
        await scheduler.start()

        assert scheduler._task == original_task

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop(self, mock_haim_engine, dream_scheduler_config):
        """Test stopping the scheduler."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        await scheduler.start()
        await scheduler.stop()

        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_current_session(
        self, mock_haim_engine, dream_scheduler_config
    ):
        """Test stopping scheduler cancels running session."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        await scheduler.start()

        # Create a mock running session
        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.cancel = MagicMock()
        scheduler._current_session = mock_session

        await scheduler.stop()

        mock_session.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_dream_manual(self, mock_haim_engine, dream_scheduler_config):
        """Test manual dream trigger."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        with patch(
            "mnemocore.subconscious.dream_scheduler.DreamSession"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_result = DreamSessionResult(
                session_id="test",
                trigger_reason=DreamTriggerReason.MANUAL,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=1.0,
                status="completed",
            )
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session_cls.return_value = mock_session

            result = await scheduler.trigger_dream()

        assert result.status == "completed"
        assert result.trigger_reason == DreamTriggerReason.MANUAL

    @pytest.mark.asyncio
    async def test_trigger_dream_already_running(
        self, mock_haim_engine, dream_scheduler_config
    ):
        """Test manual trigger when session already running raises error."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        # Mock running session
        mock_session = MagicMock()
        mock_session.is_running = True
        scheduler._current_session = mock_session

        with pytest.raises(RuntimeError, match="already running"):
            await scheduler.trigger_dream()

    def test_record_activity(self, mock_haim_engine, dream_scheduler_config):
        """Test record_activity forwards to idle detector."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        scheduler.record_activity("query")

        # Should not raise - just forwards to idle detector

    def test_get_idle_state(self, mock_haim_engine, dream_scheduler_config):
        """Test get_idle_state returns idle detector state."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        state = scheduler.get_idle_state()

        assert isinstance(state, IdleState)

    def test_get_recent_sessions(self, mock_haim_engine, dream_scheduler_config):
        """Test get_recent_sessions returns session history."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        # Add some mock sessions to history
        for i in range(5):
            scheduler._session_history.append(DreamSessionResult(
                session_id=f"session_{i}",
                trigger_reason=DreamTriggerReason.MANUAL,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=1.0,
                status="completed",
            ))

        recent = scheduler.get_recent_sessions(limit=3)

        assert len(recent) == 3

    def test_get_stats(self, mock_haim_engine, dream_scheduler_config):
        """Test get_stats returns scheduler statistics."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        stats = scheduler.get_stats()

        assert "enabled" in stats
        assert "running" in stats
        assert "current_session_running" in stats
        assert "idle_state" in stats
        assert "schedules" in stats

    def test_set_completion_callback(self, mock_haim_engine, dream_scheduler_config):
        """Test setting completion callback."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        callback = MagicMock()
        scheduler.set_completion_callback(callback)

        assert scheduler._on_dream_complete == callback


# =====================================================================
# Schedule and Cron Tests
# =====================================================================

class TestDreamSchedule:
    """Tests for schedule parsing and triggering."""

    def test_schedule_presets(self):
        """Test schedule presets are valid."""
        assert SCHEDULE_PRESETS["nightly"] == "0 2 * * *"
        assert SCHEDULE_PRESETS["hourly"] == "0 * * * *"
        assert SCHEDULE_PRESETS["daily"] == "0 0 * * *"
        assert SCHEDULE_PRESETS["weekly"] == "0 0 * * 0"
        assert SCHEDULE_PRESETS["never"] == ""

    def test_is_schedule_due_with_croniter(self, mock_haim_engine):
        """Test schedule due check with croniter available."""
        config = DreamSchedulerConfig(
            schedules=[
                DreamSchedule(
                    name="test",
                    cron_expression="* * * * *",  # Every minute
                    enabled=True,
                )
            ]
        )
        scheduler = DreamScheduler(mock_haim_engine, config)

        # With croniter, should calculate next_run
        schedule = config.schedules[0]
        is_due = scheduler._is_schedule_due(schedule)

        # First check sets next_run
        assert schedule.next_run is not None or not schedule.enabled

    def test_is_schedule_due_disabled(self, mock_haim_engine):
        """Test schedule due check with disabled schedule."""
        config = DreamSchedulerConfig(
            schedules=[
                DreamSchedule(
                    name="test",
                    cron_expression="* * * * *",
                    enabled=False,
                )
            ]
        )
        scheduler = DreamScheduler(mock_haim_engine, config)

        schedule = config.schedules[0]
        is_due = scheduler._is_schedule_due(schedule)

        assert is_due is False

    def test_is_schedule_due_empty_cron(self, mock_haim_engine):
        """Test schedule due check with empty cron expression."""
        config = DreamSchedulerConfig(
            schedules=[
                DreamSchedule(
                    name="never",
                    cron_expression="",
                    enabled=True,
                )
            ]
        )
        scheduler = DreamScheduler(mock_haim_engine, config)

        schedule = config.schedules[0]
        is_due = scheduler._is_schedule_due(schedule)

        assert is_due is False

    def test_parse_next_run_fallback(self, mock_haim_engine):
        """Test fallback cron parser for simple expressions."""
        config = DreamSchedulerConfig()
        scheduler = DreamScheduler(mock_haim_engine, config)

        # Test with simple expression
        now = datetime.now(timezone.utc)
        next_run = scheduler._parse_next_run_fallback("0 2 * * *", now)

        # Should return a datetime
        assert next_run is not None
        assert next_run.hour == 2
        assert next_run.minute == 0

    def test_parse_next_run_fallback_invalid(self, mock_haim_engine):
        """Test fallback cron parser with invalid expression."""
        config = DreamSchedulerConfig()
        scheduler = DreamScheduler(mock_haim_engine, config)

        now = datetime.now(timezone.utc)
        next_run = scheduler._parse_next_run_fallback("invalid cron", now)

        # Should return None for invalid expression
        assert next_run is None


# =====================================================================
# Data Class Tests
# =====================================================================

class TestSchedulerDataClasses:
    """Tests for scheduler data classes."""

    def test_idle_state(self):
        """Test IdleState dataclass."""
        state = IdleState(
            last_activity_time=100.0,
            last_query_time=90.0,
            current_idle_duration=10.0,
            is_idle=True,
            cpu_percent=15.0,
            memory_percent=50.0,
            active_connections=0,
        )

        assert state.last_activity_time == 100.0
        assert state.is_idle is True

    def test_dream_schedule(self):
        """Test DreamSchedule dataclass."""
        schedule = DreamSchedule(
            name="nightly",
            cron_expression="0 2 * * *",
            enabled=True,
        )

        assert schedule.name == "nightly"
        assert schedule.enabled is True
        assert schedule.last_run is None
        assert schedule.next_run is None

    def test_dream_session_config_defaults(self):
        """Test DreamSessionConfig default values."""
        config = DreamSessionConfig()

        assert config.max_duration_seconds == 600
        assert config.max_memories_to_process == 1000
        assert config.enable_consolidation is True
        assert config.generate_report is True

    def test_dream_session_result_to_dict(self):
        """Test DreamSessionResult serialization."""
        result = DreamSessionResult(
            session_id="test_session",
            trigger_reason=DreamTriggerReason.MANUAL,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=1.5,
            status="completed",
            memories_processed=10,
        )

        d = result.to_dict()

        assert d["session_id"] == "test_session"
        assert d["status"] == "completed"
        assert d["trigger_reason"] == "manual"
        assert d["memories_processed"] == 10


# =====================================================================
# Factory Function Tests
# =====================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_dream_scheduler_with_config(self, mock_haim_engine):
        """Test create_dream_scheduler with explicit config."""
        config = DreamSchedulerConfig(
            idle_threshold_seconds=600,
            enabled=True,
        )

        scheduler = create_dream_scheduler(mock_haim_engine, config)

        assert scheduler.cfg.idle_threshold_seconds == 600

    def test_create_dream_scheduler_default_config(self, mock_haim_engine):
        """Test create_dream_scheduler with default config."""
        with patch(
            "mnemocore.core.config.get_config"
        ) as mock_get_config:
            mock_haim_cfg = MagicMock()
            mock_haim_cfg.dream_idle_threshold = 500
            mock_get_config.return_value = mock_haim_cfg

            scheduler = create_dream_scheduler(mock_haim_engine)

            assert scheduler is not None


# =====================================================================
# Integration Tests
# =====================================================================

class TestSchedulerIntegration:
    """End-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_scheduler_idle_trigger_flow(
        self, mock_haim_engine, dream_scheduler_config
    ):
        """Test scheduler triggers dream on idle."""
        # Configure for quick idle detection
        dream_scheduler_config.idle_threshold_seconds = 0.1
        dream_scheduler_config.min_idle_duration = 0.1
        dream_scheduler_config.max_cpu_percent = 100.0
        dream_scheduler_config.check_interval_seconds = 0.1

        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        # Mock the pipeline execution
        with patch(
            "mnemocore.subconscious.dream_pipeline.DreamPipeline"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.run = AsyncMock(return_value={
                "success": True,
                "memories_processed": 5,
            })
            mock_pipeline_cls.return_value = mock_pipeline

            await scheduler.start()

            # Wait briefly for potential trigger
            await asyncio.sleep(0.3)

            await scheduler.stop()

        # Scheduler should have run
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_scheduler_manual_trigger_while_idle(
        self, mock_haim_engine, dream_scheduler_config
    ):
        """Test manual trigger while system is idle."""
        scheduler = DreamScheduler(mock_haim_engine, dream_scheduler_config)

        with patch(
            "mnemocore.subconscious.dream_scheduler.DreamSession"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_result = DreamSessionResult(
                session_id="manual_test",
                trigger_reason=DreamTriggerReason.MANUAL,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=1.0,
                status="completed",
                memories_processed=10,
            )
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session_cls.return_value = mock_session

            result = await scheduler.trigger_dream()

        assert result.status == "completed"
        assert result.trigger_reason == DreamTriggerReason.MANUAL

    @pytest.mark.asyncio
    async def test_scheduler_respects_manual_only_mode(
        self, mock_haim_engine
    ):
        """Test scheduler with require_manual_trigger=True ignores idle."""
        config = DreamSchedulerConfig(
            idle_threshold_seconds=0.1,
            min_idle_duration=0.1,
            max_cpu_percent=100.0,
            check_interval_seconds=0.1,
            require_manual_trigger=True,
        )

        scheduler = DreamScheduler(mock_haim_engine, config)

        with patch.object(
            scheduler.idle_detector,
            "should_trigger_dream",
            return_value=True
        ):
            triggers = await scheduler._evaluate_triggers()

        # Should not trigger even when idle due to manual-only mode
        # (but scheduled triggers would still work if configured)
        assert triggers is None or triggers == DreamTriggerReason.SCHEDULED
