"""
Subconscious Consolidation Worker (Phase 4.0+)
==============================================
Periodic background worker that runs semantic consolidation on memory tiers.

This worker operates autonomously in the background, consolidating similar
memories at configurable intervals. It is designed to run continuously
alongside the main HAIMEngine.

Usage:
    worker = SubconsciousConsolidationWorker(engine)
    await worker.start()       # Launches background task
    await worker.run_once()    # One-shot execution (for testing)
    await worker.stop()        # Graceful shutdown
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, TYPE_CHECKING

from loguru import logger

from ..core.consolidation import SemanticConsolidator

if TYPE_CHECKING:
    from ..core.engine import HAIMEngine


@dataclass
class ConsolidationWorkerConfig:
    """Configuration for the subconscious consolidation worker."""
    interval_seconds: float = 3600.0  # 1 hour default
    hot_tier_enabled: bool = True
    warm_tier_enabled: bool = True
    similarity_threshold: float = 0.85
    min_cluster_size: int = 2
    enabled: bool = True


class SubconsciousConsolidationWorker:
    """
    Periodic consolidation worker that runs in the background.

    This worker:
    1. Wakes up at configurable intervals
    2. Runs semantic consolidation on HOT and/or WARM tiers
    3. Reports statistics to the engine
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        config: Optional[ConsolidationWorkerConfig] = None,
    ):
        """
        Initialize the consolidation worker.

        Args:
            engine: HAIMEngine instance to consolidate.
            config: Optional configuration overrides.
        """
        self.engine = engine
        self.cfg = config or ConsolidationWorkerConfig()

        # Create the consolidator
        self.consolidator = SemanticConsolidator(
            tier_manager=engine.tier_manager,
            similarity_threshold=self.cfg.similarity_threshold,
            min_cluster_size=self.cfg.min_cluster_size,
        )

        # Lifecycle state
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.last_run: Optional[datetime] = None
        self.stats: Dict = {}

    async def start(self) -> None:
        """Launch the background consolidation loop."""
        if not self.cfg.enabled:
            logger.info("SubconsciousConsolidationWorker disabled by config.")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._consolidation_loop(),
            name="subconscious_consolidation"
        )
        logger.info(
            f"SubconsciousConsolidationWorker started — "
            f"interval={self.cfg.interval_seconds}s"
        )

    async def stop(self) -> None:
        """Gracefully stop the worker."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SubconsciousConsolidationWorker stopped.")

    async def _consolidation_loop(self) -> None:
        """Main loop: sleep, consolidate, repeat."""
        while self._running:
            try:
                await asyncio.sleep(self.cfg.interval_seconds)
                if self._running:
                    await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    f"SubconsciousConsolidationWorker error: {exc}",
                    exc_info=True
                )
                await asyncio.sleep(60)  # Backoff on error

    async def run_once(self) -> Dict:
        """
        Execute one consolidation cycle.

        Consolidates both HOT and WARM tiers (if enabled) and aggregates
        statistics.

        Returns:
            Dict with consolidation statistics.
        """
        t0 = time.monotonic()
        logger.info("=== Subconscious Consolidation — start ===")

        total_stats = {
            "hot": {},
            "warm": {},
            "elapsed_seconds": 0.0,
            "timestamp": None,
        }

        # Consolidate HOT tier
        if self.cfg.hot_tier_enabled:
            try:
                hot_stats = await self.consolidator.consolidate_tier(
                    tier="hot",
                    threshold=self.cfg.similarity_threshold,
                )
                total_stats["hot"] = hot_stats
            except Exception as e:
                logger.error(f"HOT tier consolidation failed: {e}")
                total_stats["hot"] = {"error": str(e)}

        # Consolidate WARM tier
        if self.cfg.warm_tier_enabled:
            try:
                warm_stats = await self.consolidator.consolidate_tier(
                    tier="warm",
                    threshold=self.cfg.similarity_threshold,
                )
                total_stats["warm"] = warm_stats
            except Exception as e:
                logger.error(f"WARM tier consolidation failed: {e}")
                total_stats["warm"] = {"error": str(e)}

        elapsed = time.monotonic() - t0
        self.last_run = datetime.now(timezone.utc)

        total_stats["elapsed_seconds"] = round(elapsed, 2)
        total_stats["timestamp"] = self.last_run.isoformat()

        self.stats = total_stats

        # Log summary
        hot_merged = total_stats["hot"].get("nodes_merged", 0)
        warm_merged = total_stats["warm"].get("nodes_merged", 0)
        logger.info(
            f"=== Subconscious Consolidation — done in {elapsed:.1f}s "
            f"| HOT merged={hot_merged} WARM merged={warm_merged} ==="
        )

        return total_stats


# Factory function for creating from config
def create_consolidation_worker(
    engine: "HAIMEngine",
    interval_seconds: Optional[float] = None,
) -> SubconsciousConsolidationWorker:
    """
    Create a consolidation worker with optional interval override.

    Args:
        engine: HAIMEngine instance.
        interval_seconds: Optional interval override (reads from config if not provided).

    Returns:
        Configured SubconsciousConsolidationWorker instance.
    """
    from ..core.config import get_config

    config = get_config()

    # Read interval from config if not provided
    if interval_seconds is None:
        interval_seconds = getattr(
            config,
            "consolidation_interval_seconds",
            3600.0
        )

    worker_config = ConsolidationWorkerConfig(
        interval_seconds=interval_seconds,
        similarity_threshold=0.85,
        min_cluster_size=2,
        enabled=True,
    )

    return SubconsciousConsolidationWorker(engine, worker_config)
