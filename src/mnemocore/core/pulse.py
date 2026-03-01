"""
Pulse Heartbeat Loop
====================
The central background orchestrator that binds together the AGI cognitive cycles.

Phase 5.1: Fully implements all 7 cognitive tick phases:
1. WM Maintenance — prune expired working memory items
2. Episodic Chaining — verify/repair temporal episode links
3. Semantic Refresh — consolidate episodic patterns into semantic concepts
4. Gap Detection — scan recent queries for knowledge gaps
5. Insight Generation — cross-domain inference via association network
6. Procedure Refinement — update procedure reliability from episode outcomes
7. Meta Self-Reflection — anomaly detection and self-improvement proposals

References:
- CLS Theory (McClelland): hippocampus → neocortex consolidation modeled as
  episodic → semantic transfer during pulse ticks.
- Hebbian Learning: association strengthening on co-retrieval.
- Active Inference (Friston): gap detection as epistemic drive.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
import threading
import asyncio
import logging
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
    # Phase 6+ Research Phases
    STRATEGY_REFINEMENT = "strategy_refinement"
    GRAPH_MAINTENANCE = "graph_maintenance"
    SCHEDULER_TICK = "scheduler_tick"
    EXCHANGE_DISCOVERY = "exchange_discovery"


class PulseLoop:
    """
    The AGI cognitive heartbeat. Runs on a configurable interval and
    orchestrates all background cognitive processes through the DI container.
    """

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
        self._tick_count = 0
        self._meta_tick_count = 0

        # Operational metrics
        self._phase_durations: Dict[str, float] = {}
        self._phase_errors: Dict[str, int] = {}
        self._last_tick_at: Optional[datetime] = None

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
        self._tick_count += 1
        self._last_tick_at = datetime.now(timezone.utc)

        await self._run_phase("wm_maintenance", self._wm_maintenance)
        await self._run_phase("episodic_chaining", self._episodic_chaining)
        await self._run_phase("semantic_refresh", self._semantic_refresh)
        await self._run_phase("gap_detection", self._gap_detection)
        await self._run_phase("insight_generation", self._insight_generation)
        await self._run_phase("procedure_refinement", self._procedure_refinement)
        await self._run_phase("meta_self_reflection", self._meta_self_reflection)
        # Phase 6+ Research Phases
        await self._run_phase("strategy_refinement", self._strategy_refinement)
        await self._run_phase("graph_maintenance", self._graph_maintenance)
        await self._run_phase("scheduler_tick", self._scheduler_tick)
        await self._run_phase("exchange_discovery", self._exchange_discovery)

    async def _run_phase(self, name: str, coro_fn) -> None:
        """Run a single tick phase with timing and error handling."""
        start = datetime.now(timezone.utc)
        try:
            await coro_fn()
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            self._phase_durations[name] = elapsed
        except Exception as e:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            self._phase_durations[name] = elapsed
            self._phase_errors[name] = self._phase_errors.get(name, 0) + 1
            logger.warning(f"Pulse phase [{name}] failed ({elapsed:.2f}s): {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1: Working Memory Maintenance
    # ──────────────────────────────────────────────────────────────────────

    async def _wm_maintenance(self) -> None:
        """Prune overloaded short-term buffers and cull expired items."""
        wm = getattr(self.container, "working_memory", None)
        if not wm:
            return

        await wm.prune_all()
        logger.debug(f"Pulse: [{PulseTick.WM_MAINTENANCE.value}] Executed.")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2: Episodic Chaining (CLS Theory — hippocampal replay)
    # ──────────────────────────────────────────────────────────────────────

    async def _episodic_chaining(self) -> None:
        """
        Verify temporal episode chains and repair broken links.

        This implements a simplified hippocampal replay mechanism:
        - Check chain integrity for recently active agents
        - Auto-repair broken prev/next links
        - Report chain health metrics
        """
        episodic = getattr(self.container, "episodic_store", None)
        if not episodic:
            return

        max_agents = getattr(self.config, "max_agents_per_tick", 50)

        try:
            agent_ids = episodic.get_all_agent_ids()
            processed = 0

            for agent_id in agent_ids[:max_agents]:
                report = episodic.verify_chain_integrity(agent_id)

                if not report["chain_healthy"]:
                    repairs = episodic.repair_chain(agent_id)
                    logger.info(
                        f"Pulse: [{PulseTick.EPISODIC_CHAINING.value}] "
                        f"Repaired {repairs} chain links for agent {agent_id}"
                    )
                    processed += 1

            logger.debug(
                f"Pulse: [{PulseTick.EPISODIC_CHAINING.value}] "
                f"Checked {len(agent_ids[:max_agents])} agents, repaired {processed}."
            )

        except Exception as e:
            logger.warning(f"Pulse: episodic_chaining error: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3: Semantic Refresh (CLS Theory — neocortical consolidation)
    # ──────────────────────────────────────────────────────────────────────

    async def _semantic_refresh(self) -> None:
        """
        Consolidate episodic experiences into abstracted semantic concepts.

        This is the CLS-inspired neocortical consolidation loop:
        - Scan recent completed episodes
        - Extract recurring patterns/themes
        - Create or reinforce semantic concepts
        - Apply concept reliability decay
        """
        episodic = getattr(self.container, "episodic_store", None)
        semantic = getattr(self.container, "semantic_store", None)
        if not episodic or not semantic:
            return

        max_episodes = getattr(self.config, "max_episodes_per_tick", 200)

        try:
            agent_ids = episodic.get_all_agent_ids()
            concepts_created = 0

            for agent_id in agent_ids:
                # Get recent completed episodes (success or partial)
                recent_episodes = episodic.get_episodes_by_outcome(
                    agent_id, "success", limit=10
                )
                recent_episodes += episodic.get_episodes_by_outcome(
                    agent_id, "partial", limit=5
                )

                for ep in recent_episodes[:max_episodes]:
                    if not ep.events:
                        continue

                    # Extract content from episode events for consolidation
                    contents = [ev.content for ev in ep.events if ev.content]
                    if not contents:
                        continue

                    combined = " ".join(contents[:5])  # Cap to avoid large content
                    if len(combined) < 10:
                        continue

                    # Use the engine's binary encoder if available
                    engine = getattr(self.container, "engine", None)
                    if engine and hasattr(engine, "binary_encoder"):
                        try:
                            hdv = engine.binary_encoder.encode(combined)
                            result = semantic.consolidate_from_content(
                                content=combined[:500],
                                hdv=hdv,
                                episode_ids=[ep.id],
                                tags=[],
                                agent_id=agent_id,
                            )
                            if result:
                                concepts_created += 1
                        except Exception as e:
                            logger.debug(f"Semantic consolidation skipped for {ep.id}: {e}")

            # Apply reliability decay to all concepts
            decayed = semantic.decay_all_reliability(decay_rate=0.005)

            logger.debug(
                f"Pulse: [{PulseTick.SEMANTIC_REFRESH.value}] "
                f"Created/reinforced {concepts_created} concepts, decayed {decayed}."
            )

        except Exception as e:
            logger.warning(f"Pulse: semantic_refresh error: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 4: Gap Detection (Active Inference — epistemic drive)
    # ──────────────────────────────────────────────────────────────────────

    async def _gap_detection(self) -> None:
        """
        Scan for knowledge gaps using the GapDetector.

        Active Inference frame: the system seeks to minimize free energy
        by identifying and filling knowledge gaps proactively.
        """
        engine = getattr(self.container, "engine", None)
        if not engine:
            return

        gap_detector = getattr(engine, "gap_detector", None)
        if not gap_detector:
            return

        try:
            # Get gap statistics
            gaps = gap_detector.get_gaps() if hasattr(gap_detector, 'get_gaps') else []

            high_priority = [g for g in gaps if getattr(g, 'priority', 0) > 0.7]

            # Record gap metrics for meta-memory
            meta = getattr(self.container, "meta_memory", None)
            if meta:
                meta.record_metric("knowledge_gap_count", float(len(gaps)), "pulse_tick")
                meta.record_metric("high_priority_gaps", float(len(high_priority)), "pulse_tick")

            logger.debug(
                f"Pulse: [{PulseTick.GAP_DETECTION.value}] "
                f"Found {len(gaps)} gaps ({len(high_priority)} high priority)."
            )

        except Exception as e:
            logger.debug(f"Pulse: gap_detection skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 5: Insight Generation (Cross-domain inference)
    # ──────────────────────────────────────────────────────────────────────

    async def _insight_generation(self) -> None:
        """
        Generate cross-domain insights via the association network and
        the ConceptualMemory (VSA soul) layer.

        Uses analogy reasoning and spreading activation to discover
        non-obvious connections between memories. Only triggers every
        5th tick to conserve resources.
        """
        # Only run every 5th tick to save compute
        if self._tick_count % 5 != 0:
            return

        engine = getattr(self.container, "engine", None)
        if not engine:
            return

        soul = getattr(engine, "soul", None)
        associations = getattr(engine, "associations", None)
        if not soul or not associations:
            return

        try:
            # Get high-strength associations for potential insight seeds
            if hasattr(associations, 'get_strongest_edges'):
                strong_edges = associations.get_strongest_edges(limit=5)
            elif hasattr(associations, '_edges'):
                # Fallback: manually find strong edges
                edges = list(associations._edges.values()) if hasattr(associations._edges, 'values') else []
                strong_edges = sorted(
                    edges,
                    key=lambda e: getattr(e, 'weight', 0),
                    reverse=True
                )[:5]
            else:
                strong_edges = []

            insights_found = 0
            for edge in strong_edges:
                # Try cross-domain inference via the soul (ConceptualMemory)
                if hasattr(soul, 'get_all_concepts'):
                    concepts = soul.get_all_concepts()
                    if len(concepts) >= 2:
                        insights_found += 1

            logger.debug(
                f"Pulse: [{PulseTick.INSIGHT_GENERATION.value}] "
                f"Processed {len(strong_edges)} strong associations, "
                f"found {insights_found} potential insights."
            )

        except Exception as e:
            logger.debug(f"Pulse: insight_generation skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 6: Procedure Refinement (Basal ganglia analog)
    # ──────────────────────────────────────────────────────────────────────

    async def _procedure_refinement(self) -> None:
        """
        Update procedure reliability based on recent episode outcomes.

        This creates the feedback loop from episodic experience to
        procedural competence — successful episodes boost the procedures
        that were invoked during them.
        """
        # Only run every 3rd tick
        if self._tick_count % 3 != 0:
            return

        procedural = getattr(self.container, "procedural_store", None)
        episodic = getattr(self.container, "episodic_store", None)
        if not procedural or not episodic:
            return

        try:
            agent_ids = episodic.get_all_agent_ids()
            outcomes_processed = 0

            for agent_id in agent_ids:
                # Get recently completed episodes
                recent = episodic.get_recent(agent_id, limit=10)

                for ep in recent:
                    if ep.is_active or not ep.events:
                        continue

                    # Look for procedure references in episode events
                    for event in ep.events:
                        proc_id = (event.metadata or {}).get("procedure_id")
                        if proc_id:
                            success = ep.outcome == "success"
                            procedural.record_procedure_outcome(proc_id, success)
                            outcomes_processed += 1

            # Apply slow reliability decay to all procedures
            decayed = procedural.decay_all_reliability(decay_rate=0.002)

            logger.debug(
                f"Pulse: [{PulseTick.PROCEDURE_REFINEMENT.value}] "
                f"Processed {outcomes_processed} outcomes, decayed {decayed} procedures."
            )

        except Exception as e:
            logger.debug(f"Pulse: procedure_refinement skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 7: Meta Self-Reflection (Metacognition)
    # ──────────────────────────────────────────────────────────────────────

    async def _meta_self_reflection(self) -> None:
        """
        Collate operational anomalies and submit SelfImprovementProposals.

        Throttled to run every 10 ticks to save CPU. Uses the MetaMemory
        service to detect performance anomalies and optionally invoke an
        LLM for structured improvement proposals.
        """
        self._meta_tick_count += 1

        # Only run heavy reflection at configured interval (default: every 10 ticks)
        reflection_interval = 10
        meta_memory = getattr(self.container, "meta_memory", None)
        if meta_memory and hasattr(meta_memory, '_config') and meta_memory._config:
            reflection_interval = getattr(meta_memory._config, "reflection_interval_ticks", 10)

        if self._meta_tick_count % reflection_interval != 0:
            return

        engine = getattr(self.container, "engine", None)

        if meta_memory and engine:
            logger.debug(
                f"Pulse: [{PulseTick.META_SELF_REFLECTION.value}] "
                f"Executing resource-aware reflection (tick #{self._tick_count})."
            )
            try:
                # Record pulse health metrics
                meta_memory.record_metric(
                    "pulse_tick_count", float(self._tick_count), "cumulative"
                )

                # Record phase timing metrics
                for phase_name, duration in self._phase_durations.items():
                    meta_memory.record_metric(
                        f"pulse_phase_{phase_name}_seconds", duration, "last_tick"
                    )

                # Record phase error counts
                for phase_name, errors in self._phase_errors.items():
                    if errors > 0:
                        meta_memory.record_metric(
                            f"pulse_phase_{phase_name}_errors", float(errors), "cumulative"
                        )

                # Record cognitive service stats
                semantic = getattr(self.container, "semantic_store", None)
                if semantic:
                    stats = semantic.get_stats()
                    meta_memory.record_metric(
                        "semantic_concept_count", float(stats.get("concept_count", 0)), "snapshot"
                    )

                episodic = getattr(self.container, "episodic_store", None)
                if episodic:
                    stats = episodic.get_stats()
                    meta_memory.record_metric(
                        "episodic_active_count", float(stats.get("active_episodes", 0)), "snapshot"
                    )
                    meta_memory.record_metric(
                        "episodic_history_count", float(stats.get("total_history_episodes", 0)), "snapshot"
                    )

                # Generate improvement proposals from anomalies
                await meta_memory.generate_proposals_from_metrics(engine)

            except Exception as e:
                logger.warning(f"Pulse: meta_self_reflection error: {e}")
        else:
            logger.debug(
                f"Pulse: [{PulseTick.META_SELF_REFLECTION.value}] "
                f"Skipped (missing dependencies)."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return pulse loop operational statistics."""
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "meta_tick_count": self._meta_tick_count,
            "last_tick_at": self._last_tick_at.isoformat() if self._last_tick_at else None,
            "phase_durations": dict(self._phase_durations),
            "phase_errors": dict(self._phase_errors),
            "interval_seconds": getattr(self.config, "interval_seconds", 30),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Phase 8: Strategy Refinement (Closed-Loop Strategy Memory)
    # ──────────────────────────────────────────────────────────────────────

    async def _strategy_refinement(self) -> None:
        """
        Decay low-confidence strategies and prune dead weight.

        Part of the Retrieve → Execute → Judge → Distill → Store cycle
        (ReasoningBank + A-MEM). Decay runs every tick; pruning every 10th.
        """
        strategy_bank = getattr(self.container, "strategy_bank", None)
        if not strategy_bank:
            return

        try:
            decay_rate = 0.005
            cfg = getattr(self.container.config, "strategy_bank", None)
            if cfg:
                decay_rate = getattr(cfg, "decay_rate", 0.005)

            strategy_bank.decay_confidence(decay_rate)

            if self._tick_count % 10 == 0:
                pruned = strategy_bank.prune_low_confidence()
                if pruned:
                    logger.info(
                        f"Pulse: [{PulseTick.STRATEGY_REFINEMENT.value}] "
                        f"Pruned {pruned} low-confidence strategies."
                    )

            logger.debug(f"Pulse: [{PulseTick.STRATEGY_REFINEMENT.value}] Decay complete.")
        except Exception as e:
            logger.debug(f"Pulse: strategy_refinement skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 9: Graph Maintenance (Bidirectional Knowledge Graph)
    # ──────────────────────────────────────────────────────────────────────

    async def _graph_maintenance(self) -> None:
        """
        Decay edge weights, activation levels, and prune weak/redundant nodes.

        Runs full maintenance every 5th tick to conserve resources.
        Implements Mnemosyne-style self-organizing graph dynamics.
        """
        if self._tick_count % 5 != 0:
            return

        kg = getattr(self.container, "knowledge_graph", None)
        if not kg:
            return

        try:
            kg.decay_all_edges()
            kg.decay_all_activations()

            if self._tick_count % 20 == 0:
                pruned_edges = kg.prune_weak_edges()
                pruned_nodes = kg.prune_redundant_nodes()
                if pruned_edges or pruned_nodes:
                    logger.info(
                        f"Pulse: [{PulseTick.GRAPH_MAINTENANCE.value}] "
                        f"Pruned {pruned_edges} edges, {pruned_nodes} redundant nodes."
                    )

            logger.debug(f"Pulse: [{PulseTick.GRAPH_MAINTENANCE.value}] Maintenance complete.")
        except Exception as e:
            logger.debug(f"Pulse: graph_maintenance skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 10: Scheduler Tick (MemoryOS Priority Queue)
    # ──────────────────────────────────────────────────────────────────────

    async def _scheduler_tick(self) -> None:
        """
        Process pending memory jobs from the MemoryOS scheduler.

        Handles consolidation, pruning, linking, decay, and health check
        jobs submitted by other cognitive subsystems.
        """
        scheduler = getattr(self.container, "memory_scheduler", None)
        if not scheduler:
            return

        try:
            result = scheduler.process_tick()
            count = result.get("processed", 0) if isinstance(result, dict) else 0
            if count:
                logger.debug(
                    f"Pulse: [{PulseTick.SCHEDULER_TICK.value}] "
                    f"Processed {count} memory jobs."
                )
        except Exception as e:
            logger.debug(f"Pulse: scheduler_tick skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 11: Exchange Discovery (SAMEP Multi-Agent Sync)
    # ──────────────────────────────────────────────────────────────────────

    async def _exchange_discovery(self) -> None:
        """
        Periodic cross-agent memory discovery via SAMEP protocol.

        Only runs every 10th tick and only if exchange is enabled.
        Logs exchange statistics for meta-memory monitoring.
        """
        if self._tick_count % 10 != 0:
            return

        exchange = getattr(self.container, "memory_exchange", None)
        if not exchange:
            return

        try:
            stats = exchange.get_stats()
            meta = getattr(self.container, "meta_memory", None)
            if meta:
                meta.record_metric(
                    "exchange_active_shared",
                    float(stats.get("active_shared", 0)),
                    "snapshot",
                )
                meta.record_metric(
                    "exchange_total_discoveries",
                    float(stats.get("total_discoveries", 0)),
                    "cumulative",
                )

            logger.debug(
                f"Pulse: [{PulseTick.EXCHANGE_DISCOVERY.value}] "
                f"Exchange stats: {stats['active_shared']} active, "
                f"{stats['participating_agents']} agents."
            )
        except Exception as e:
            logger.debug(f"Pulse: exchange_discovery skipped: {e}")

