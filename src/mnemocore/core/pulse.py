"""
Pulse Heartbeat Loop
====================
The central background orchestrator that binds together the AGI cognitive cycles.
Triggers working memory maintenance, episodic sequence linking, gap tracking, and subconscious inferences.
"""

from typing import Optional
from enum import Enum
import threading
import asyncio
import logging
import traceback
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PulseTick(Enum):
    WM_MAINTENANCE = "wm_maintenance"
    EPISODIC_CHAINING = "episodic_chaining"
    SEMANTIC_REFRESH = "semantic_refresh"
    GAP_DETECTION = "gap_detection"
    INSIGHT_GENERATION = "insight_generation"
    PROCEDURE_REFINEMENT = "procedure_refinement"
    META_SELF_REFLECTION = "meta_self_reflection"


class PulseLoop:
    def __init__(self, container, config):
        """
        Args:
            container: The fully built DI Container containing all memory sub-services.
            config: Specifically the `config.pulse` section settings.
        """
        self.container = container
        self.config = config
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Begin the background pulse orchestrator."""
        if not getattr(self.config, "enabled", False):
            logger.info("Pulse loop is disabled via configuration.")
            return

        self._running = True
        interval = getattr(self.config, "interval_seconds", 30)
        logger.info(f"Starting AGI Pulse Loop (interval={interval}s).")

        while self._running:
            start_time = datetime.now(timezone.utc)
            try:
                await self.tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during Pulse tick: {e}", exc_info=True)

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            sleep_time = max(0.1, interval - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        """Gracefully interrupt and unbind the Pulse loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("AGI Pulse Loop stopped.")

    async def tick(self) -> None:
        """Execute a full iteration across the cognitive architecture planes."""
        await self._wm_maintenance()
        await self._episodic_chaining()
        await self._semantic_refresh()
        await self._gap_detection()
        await self._insight_generation()
        await self._procedure_refinement()
        await self._meta_self_reflection()

    async def _wm_maintenance(self) -> None:
        """Prune overloaded short-term buffers and cull expired items."""
        if hasattr(self.container, "working_memory") and self.container.working_memory:
            self.container.working_memory.prune_all()
            logger.debug(f"Pulse: [{PulseTick.WM_MAINTENANCE.value}] Executed.")

    async def _episodic_chaining(self) -> None:
        """Retroactively verify event streams and apply temporal links between episodic contexts."""
        logger.debug(f"Pulse: [{PulseTick.EPISODIC_CHAINING.value}] Stubbed.")

    async def _semantic_refresh(self) -> None:
        """Prompt Qdrant abstractions or run `semantic_consolidation` loops over episodic data."""
        logger.debug(f"Pulse: [{PulseTick.SEMANTIC_REFRESH.value}] Stubbed.")

    async def _gap_detection(self) -> None:
        """Unearth missing knowledge vectors (GapDetector integration)."""
        logger.debug(f"Pulse: [{PulseTick.GAP_DETECTION.value}] Stubbed.")

    async def _insight_generation(self) -> None:
        """Forward memory patterns to LLM for spontaneous inference generation."""
        logger.debug(f"Pulse: [{PulseTick.INSIGHT_GENERATION.value}] Stubbed.")

    async def _procedure_refinement(self) -> None:
        """Modify procedure reliabilities directly depending on episode occurrences."""
        logger.debug(f"Pulse: [{PulseTick.PROCEDURE_REFINEMENT.value}] Stubbed.")

    async def _meta_self_reflection(self) -> None:
        """Collate macro anomalies and submit SelfImprovementProposals. Throttled to save CPU."""
        if not getattr(self, "_meta_tick_count", None):
            self._meta_tick_count = 0
            
        self._meta_tick_count += 1
        
        # Only run heavy reflection every 10 ticks (e.g., every 5 minutes) to save compute
        if self._meta_tick_count % 10 != 0:
            return
            
        meta_memory = getattr(self.container, "meta_memory", None)
        engine = getattr(self.container, "engine", None) # The DI container in API sets this, or we fallback
        
        if meta_memory and engine:
            logger.debug(f"Pulse: [{PulseTick.META_SELF_REFLECTION.value}] Executing resource-aware reflection.")
            # engine might be accessible via app.state depending on how DI is wired, 
            # assuming `engine` is passed somehow, otherwise we gracefully skip.
            await meta_memory.generate_proposals_from_metrics(engine)
        else:
            logger.debug(f"Pulse: [{PulseTick.META_SELF_REFLECTION.value}] Skipped (missing dependencies).")

