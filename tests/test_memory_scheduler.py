"""
Tests for MemoryOS Scheduler + Health Score (memory_scheduler.py)
==================================================================
Covers MemoryScheduler, MemoryHealthScore, MemoryJob,
MemoryJobType, MemoryJobPriority.

Research basis: MemoryOS (EMNLP 2025), Neuroca health scoring.
"""

import time
from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta

import pytest

from mnemocore.core.memory_scheduler import (
    MemoryHealthScore,
    MemoryJob,
    MemoryJobPriority,
    MemoryJobType,
    MemoryScheduler,
)


# ═══════════════════════════════════════════════════════════════════════
# MemoryJobPriority / MemoryJobType
# ═══════════════════════════════════════════════════════════════════════

class TestEnums:

    def test_priority_ordering(self):
        assert MemoryJobPriority.CRITICAL.value < MemoryJobPriority.HIGH.value
        assert MemoryJobPriority.HIGH.value < MemoryJobPriority.NORMAL.value
        assert MemoryJobPriority.NORMAL.value < MemoryJobPriority.LOW.value
        assert MemoryJobPriority.LOW.value < MemoryJobPriority.DEFERRED.value

    def test_job_types(self):
        assert MemoryJobType.CONSOLIDATION.value == "consolidation"
        assert MemoryJobType.INTERRUPT.value == "interrupt"


# ═══════════════════════════════════════════════════════════════════════
# MemoryJob
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryJob:

    def test_default_job(self):
        job = MemoryJob(
            job_type=MemoryJobType.CONSOLIDATION.value,
            priority=MemoryJobPriority.NORMAL.value,
        )
        assert job.job_type == MemoryJobType.CONSOLIDATION.value
        assert job.priority == MemoryJobPriority.NORMAL.value
        assert job.attempts == 0
        assert job.is_expired is False

    def test_job_ordering(self):
        """Higher priority (lower value) should sort first."""
        high = MemoryJob(
            job_type=MemoryJobType.PRUNING.value,
            priority=MemoryJobPriority.HIGH.value,
        )
        # Sleep tiny amount so created_at differs
        time.sleep(0.001)
        low = MemoryJob(
            job_type=MemoryJobType.DECAY.value,
            priority=MemoryJobPriority.LOW.value,
        )
        # In a min-heap, high priority should come before low
        assert (high.priority, high.created_at) <= (low.priority, low.created_at)

    def test_expired_job(self):
        job = MemoryJob(
            job_type=MemoryJobType.HEALTH_CHECK.value,
            priority=MemoryJobPriority.NORMAL.value,
            deadline_seconds=0.001,  # Expires immediately
        )
        time.sleep(0.01)
        assert job.is_expired is True

    def test_not_expired(self):
        job = MemoryJob(
            job_type=MemoryJobType.HEALTH_CHECK.value,
            priority=MemoryJobPriority.NORMAL.value,
            deadline_seconds=3600.0,  # 1 hour
        )
        assert job.is_expired is False


# ═══════════════════════════════════════════════════════════════════════
# MemoryHealthScore
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryHealthScore:

    def test_calculate_fresh_memory(self):
        """Recently accessed memory should have high health."""
        score = MemoryHealthScore.calculate(
            node_id="test-1",
            last_accessed=datetime.now(timezone.utc),
            access_count=10,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert score.health > 0.3

    def test_calculate_stale_memory(self):
        """Old, rarely accessed memory should have lower health."""
        score = MemoryHealthScore.calculate(
            node_id="test-2",
            last_accessed=datetime.now(timezone.utc) - timedelta(days=30),
            access_count=1,
            created_at=datetime.now(timezone.utc) - timedelta(days=60),
        )
        assert score.health < 0.5

    def test_lifecycle_fresh(self):
        score = MemoryHealthScore(
            node_id="t", health=0.75, recency=0.9,
            frequency=0.8, stability=0.5, lifecycle_stage="LTM",
        )
        assert score.lifecycle_stage in ("STM", "MTM", "LTM")

    def test_health_components_bounded(self):
        score = MemoryHealthScore.calculate(
            node_id="test-3",
            last_accessed=datetime.now(timezone.utc),
            access_count=100,
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
        )
        assert 0.0 <= score.recency <= 1.0
        assert 0.0 <= score.frequency <= 1.0
        assert 0.0 <= score.stability <= 1.0
        assert 0.0 <= score.health <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# MemoryScheduler
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryScheduler:

    def _make_job(self, job_type: MemoryJobType = MemoryJobType.CONSOLIDATION,
                  priority: MemoryJobPriority = MemoryJobPriority.NORMAL,
                  **kwargs) -> MemoryJob:
        return MemoryJob(
            job_type=job_type.value,
            priority=priority.value,
            **kwargs,
        )

    def test_submit_and_process(self):
        scheduler = MemoryScheduler()
        results = []
        scheduler.register_handler(
            MemoryJobType.CONSOLIDATION.value,
            lambda job: results.append(job.job_type),
        )
        job = self._make_job()
        scheduler.submit(job)
        tick_result = scheduler.process_tick()
        assert tick_result["processed"] == 1
        assert len(results) == 1

    def test_priority_ordering_in_processing(self):
        scheduler = MemoryScheduler()
        order = []
        scheduler.register_handler(
            MemoryJobType.CONSOLIDATION.value,
            lambda job: order.append("high"),
        )
        scheduler.register_handler(
            MemoryJobType.DECAY.value,
            lambda job: order.append("low"),
        )
        # Submit low first, then high
        scheduler.submit(self._make_job(MemoryJobType.DECAY, MemoryJobPriority.LOW))
        scheduler.submit(self._make_job(MemoryJobType.CONSOLIDATION, MemoryJobPriority.HIGH))
        scheduler.process_tick()
        # High priority should be processed first
        assert order[0] == "high"

    def test_submit_interrupt(self):
        scheduler = MemoryScheduler()
        results = []
        scheduler.register_handler(
            MemoryJobType.INTERRUPT.value,
            lambda job: results.append("interrupt"),
        )
        # submit_interrupt takes (target_id, payload, agent_id) and returns job id
        job_id = scheduler.submit_interrupt(
            target_id="mem-urgent",
            payload={"reason": "urgent"},
        )
        assert isinstance(job_id, str)
        tick_result = scheduler.process_tick()
        assert tick_result["processed"] >= 1
        assert "interrupt" in results

    def test_expired_jobs_skipped(self):
        scheduler = MemoryScheduler()
        results = []
        scheduler.register_handler(
            MemoryJobType.HEALTH_CHECK.value,
            lambda job: results.append("processed"),
        )
        job = self._make_job(
            MemoryJobType.HEALTH_CHECK,
            MemoryJobPriority.NORMAL,
            deadline_seconds=0.001,
        )
        scheduler.submit(job)
        time.sleep(0.01)  # Let deadline expire
        scheduler.process_tick()
        assert len(results) == 0  # Expired job should be skipped

    def test_should_consolidate(self):
        scheduler = MemoryScheduler()
        health = MemoryHealthScore(
            node_id="t", health=0.5, recency=0.5,
            frequency=0.5, stability=0.5, lifecycle_stage="MTM",
        )
        result = scheduler.should_consolidate(health)
        assert isinstance(result, bool)

    def test_calculate_health(self):
        scheduler = MemoryScheduler()
        health = scheduler.calculate_health(
            node_id="test-node",
            last_accessed=datetime.now(timezone.utc),
            access_count=5,
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        assert isinstance(health, MemoryHealthScore)
        assert health.health > 0

    def test_get_stats(self):
        scheduler = MemoryScheduler()
        scheduler.submit(self._make_job())
        stats = scheduler.get_stats()
        assert stats["queue_depth"] >= 1
        assert "completed_total" in stats

    def test_max_retries(self):
        scheduler = MemoryScheduler()
        fail_count = [0]

        def failing_handler(job):
            fail_count[0] += 1
            raise ValueError("simulated failure")

        scheduler.register_handler(MemoryJobType.PRUNING.value, failing_handler)
        scheduler.submit(self._make_job(MemoryJobType.PRUNING))

        # Process multiple ticks to exhaust retries
        for _ in range(5):
            scheduler.process_tick()

        assert fail_count[0] >= 1

    def test_multiple_handlers(self):
        scheduler = MemoryScheduler()
        results = {"con": 0, "prune": 0}
        scheduler.register_handler(
            MemoryJobType.CONSOLIDATION.value,
            lambda j: results.__setitem__("con", results["con"] + 1),
        )
        scheduler.register_handler(
            MemoryJobType.PRUNING.value,
            lambda j: results.__setitem__("prune", results["prune"] + 1),
        )
        scheduler.submit(self._make_job(MemoryJobType.CONSOLIDATION))
        scheduler.submit(self._make_job(MemoryJobType.PRUNING))
        scheduler.process_tick()
        assert results["con"] == 1
        assert results["prune"] == 1
