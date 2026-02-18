"""
Autonomous Gap-Filling via LLM Integration (Phase 4.0)
======================================================
Bridges the GapDetector with the existing HAIMLLMIntegrator to autonomously
fill detected knowledge gaps by generating and storing synthetic memories.

Pipeline:
  1. GapFiller polls GapDetector for high-priority open gaps.
  2. For each gap, it constructs a prompt asking the LLM to fill it.
  3. The LLM response is parsed into discrete factual statements.
  4. Each statement is stored in the engine as a new memory node, tagged
     with metadata: {"source": "llm_gap_fill", "gap_id": ..., "query": ...}.
  5. The gap record is marked as filled in the detector registry.

Safety controls:
  - Rate-limiting: max N gap-fill calls per hour (configurable).
  - Confidence gate: only fill gaps that stay unresolved after min_reqs queries.
  - Dry-run mode: generate responses but don't store them.
  - Minimum priority threshold before triggering LLM calls.

Usage:
    filler = GapFiller(engine, integrator, detector)
    await filler.start()                   # background task
    results = await filler.fill_now(n=5)   # immediate fill of top-5 gaps
    await filler.stop()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger

from .gap_detector import GapDetector, GapRecord


# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

@dataclass
class GapFillerConfig:
    """Controls how/when the gap filler triggers LLM calls."""
    poll_interval_seconds: float = 600.0     # check for gaps every 10 min
    max_fills_per_hour: int = 20             # rate limit
    min_priority_to_fill: float = 0.3        # skip low-priority gaps
    min_seen_before_fill: int = 2            # gap must be seen ≥ N times
    max_statements_per_gap: int = 5          # slice LLM response into pieces
    dry_run: bool = False                    # if True: generate but don't store
    store_tag: str = "llm_gap_fill"          # metadata tag on stored memories
    enabled: bool = True


# ------------------------------------------------------------------ #
#  Prompt templates                                                   #
# ------------------------------------------------------------------ #

_FILL_PROMPT_TEMPLATE = """You are an expert knowledge assistant integrated into a cognitive memory system.
A user recently queried for information that the system could not adequately answer.

Query topic: "{query}"

Please provide a concise, factual response about this topic. Structure your answer as
{max_statements} distinct, standalone factual statements (one per line, no numbering needed).
Each statement should be directly useful for answering future questions about this topic.
Keep each statement under 150 words. Be objective and accurate.

Statements:"""

_REFINE_PROMPT_TEMPLATE = """You are helping fill a knowledge gap in a memory system.

The topic "{query}" was queried {seen} times without a satisfactory answer.

Provide {max_statements} concise factual statements that would help answer this topic.
One statement per line. Be specific, factual, and succinct (max 120 words each).

Statements:"""


# ------------------------------------------------------------------ #
#  Gap filler                                                         #
# ------------------------------------------------------------------ #

class GapFiller:
    """
    Autonomous LLM-driven knowledge gap filler.
    
    Integrates with GapDetector (finds gaps) and HAIMLLMIntegrator (fills them).
    """

    def __init__(
        self,
        engine,                       # HAIMEngine
        llm_integrator,               # HAIMLLMIntegrator
        gap_detector: GapDetector,
        config: Optional[GapFillerConfig] = None,
    ):
        self.engine = engine
        self.llm = llm_integrator
        self.detector = gap_detector
        self.cfg = config or GapFillerConfig()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._fill_timestamps: List[float] = []  # for rate limiting
        self.stats: Dict = {
            "gaps_filled": 0,
            "statements_stored": 0,
            "llm_calls": 0,
            "errors": 0,
        }

    # ---- Lifecycle ----------------------------------------------- #

    async def start(self) -> None:
        if not self.cfg.enabled:
            logger.info("GapFiller disabled by config.")
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="gap_filler")
        logger.info(
            f"GapFiller started — polling every {self.cfg.poll_interval_seconds}s"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("GapFiller stopped.")

    # ---- Poll loop ----------------------------------------------- #

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.cfg.poll_interval_seconds)
                if self._running:
                    await self.fill_now(n=5)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"GapFiller poll error: {exc}", exc_info=True)
                self.stats["errors"] += 1
                await asyncio.sleep(60)

    # ---- Fill on demand ------------------------------------------ #

    async def fill_now(self, n: int = 5) -> List[Dict]:
        """
        Immediately fill the top-n open gaps.

        Returns:
            List of fill result dicts.
        """
        if not self._rate_check():
            logger.warning("GapFiller rate limit reached — skipping fill cycle.")
            return []

        open_gaps = self.detector.get_open_gaps(top_n=n * 3)  # over-fetch to filter
        eligible = [
            g for g in open_gaps
            if g.priority_score >= self.cfg.min_priority_to_fill
            and g.seen_count >= self.cfg.min_seen_before_fill
        ][:n]

        results = []
        for gap in eligible:
            if not self._rate_check():
                break
            result = await self._fill_gap(gap)
            results.append(result)

        return results

    # ---- Single gap fill ----------------------------------------- #

    async def _fill_gap(self, gap: GapRecord) -> Dict:
        """Generate and store knowledge for a single gap."""
        logger.info(
            f"Filling gap '{gap.query_text[:60]}' "
            f"(priority={gap.priority_score:.3f} seen={gap.seen_count})"
        )

        # Build prompt
        prompt = _REFINE_PROMPT_TEMPLATE.format(
            query=gap.query_text,
            seen=gap.seen_count,
            max_statements=self.cfg.max_statements_per_gap,
        )

        # Call LLM (runs sync _call_llm in executor)
        try:
            loop = asyncio.get_running_loop()
            raw_response = await loop.run_in_executor(
                None, self.llm._call_llm, prompt, 512
            )
            self._record_call()
            self.stats["llm_calls"] += 1
        except Exception as exc:
            logger.error(f"LLM call failed for gap {gap.gap_id}: {exc}")
            self.stats["errors"] += 1
            return {"gap_id": gap.gap_id, "status": "error", "error": str(exc)}

        # Parse into statements
        statements = self._parse_statements(raw_response)
        if not statements:
            logger.warning(f"LLM returned no parseable statements for gap {gap.gap_id}")
            return {"gap_id": gap.gap_id, "status": "empty_response"}

        # Store each statement as a memory node
        stored_ids = []
        if not self.cfg.dry_run:
            for stmt in statements:
                if not stmt.strip():
                    continue
                meta = {
                    "source": self.cfg.store_tag,
                    "gap_id": gap.gap_id,
                    "gap_query": gap.query_text,
                    "gap_signal": gap.signal,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "tags": ["gap_fill", "llm_generated"],
                }
                try:
                    node_id = await self.engine.store(stmt.strip(), metadata=meta)
                    stored_ids.append(node_id)
                    self.stats["statements_stored"] += 1
                except Exception as exc:
                    logger.error(f"Failed to store gap-fill statement: {exc}")

        # Mark gap as filled
        self.detector.mark_filled(gap.gap_id)
        self.stats["gaps_filled"] += 1

        result = {
            "gap_id": gap.gap_id,
            "query": gap.query_text,
            "status": "filled" if not self.cfg.dry_run else "dry_run",
            "statements": statements,
            "stored_node_ids": stored_ids,
        }

        logger.info(
            f"Gap filled: '{gap.query_text[:50]}' "
            f"→ {len(stored_ids)} statements stored"
        )
        return result

    # ---- Helpers ------------------------------------------------- #

    def _parse_statements(self, raw: str) -> List[str]:
        """
        Split LLM response into individual factual statements.
        Handles bullet points, numbered lists, and plain line-breaks.
        """
        import re
        lines = raw.strip().split("\n")
        statements = []
        for line in lines:
            # Strip bullets / numbering
            clean = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            if len(clean) > 20:  # skip header lines / blanks
                statements.append(clean)
        return statements[: self.cfg.max_statements_per_gap]

    def _rate_check(self) -> bool:
        """True if under the hourly rate limit."""
        now = time.time()
        # Keep only calls within the last hour
        self._fill_timestamps = [t for t in self._fill_timestamps if now - t < 3600]
        return len(self._fill_timestamps) < self.cfg.max_fills_per_hour

    def _record_call(self) -> None:
        """Record a fill call timestamp for rate limiting."""
        self._fill_timestamps.append(time.time())
