"""
Self-Improvement Worker (Phase 5.4 — Phase 0: Dry-Run Only)
============================================================
Background worker that identifies memory improvement candidates,
generates proposals, validates them through safety gates, and logs
results — all without writing any data (Phase 0 = dry-run only).

Based on:  docs/SELF_IMPROVEMENT_DEEP_DIVE.md
Pattern:   SubconsciousConsolidationWorker (lifecycle template)

Responsibilities:
  1. Candidate selection from HOT/WARM tiers (cheap heuristics).
  2. Proposal generation (rule-based, no LLM in Phase 0).
  3. Validation through 5 safety gates (semantic drift, fact safety,
     structure, policy, resource).
  4. Decision logging and metric recording.
  5. NO commits — all proposals are dry-run logged.

Usage:
    worker = SelfImprovementWorker(engine)
    await worker.start()       # Background loop
    await worker.run_once()    # One-shot (testing)
    await worker.stop()        # Graceful shutdown
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..core.engine import HAIMEngine
    from ..core.config import SelfImprovementConfig


# ---------------------------------------------------------------------------
# Internal models
# ---------------------------------------------------------------------------

@dataclass
class ImprovementCandidate:
    """A memory node selected for potential improvement."""
    node_id: str
    content: str
    metadata: Dict[str, Any]
    tier: str  # "hot" | "warm"
    selection_reason: str  # why it was selected
    score: float = 0.0


@dataclass
class ImprovementProposal:
    """A proposed improvement to a candidate memory."""
    candidate: ImprovementCandidate
    improvement_type: str  # normalize | summarize | deduplicate | metadata_repair
    proposed_content: str
    rationale: str
    expected_improvement_score: float
    validator_results: Dict[str, bool] = field(default_factory=dict)
    accepted: bool = False
    rejection_reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metrics tracker (in-process counters, Prometheus-compatible names)
# ---------------------------------------------------------------------------

class SelfImprovementMetrics:
    """In-process counters matching the 7 proposed Prometheus metrics."""

    def __init__(self):
        self.attempts_total: int = 0
        self.commits_total: int = 0  # always 0 in Phase 0 (dry-run)
        self.rejects_total: int = 0
        self.cycle_durations: List[float] = []
        self.candidates_per_cycle: List[int] = []
        self.quality_deltas: List[float] = []
        self.backpressure_skips_total: int = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "mnemocore_self_improve_attempts_total": self.attempts_total,
            "mnemocore_self_improve_commits_total": self.commits_total,
            "mnemocore_self_improve_rejects_total": self.rejects_total,
            "mnemocore_self_improve_cycle_duration_seconds": (
                round(self.cycle_durations[-1], 3)
                if self.cycle_durations else 0.0
            ),
            "mnemocore_self_improve_candidates_in_cycle": (
                self.candidates_per_cycle[-1]
                if self.candidates_per_cycle else 0
            ),
            "mnemocore_self_improve_quality_delta": (
                round(self.quality_deltas[-1], 4)
                if self.quality_deltas else 0.0
            ),
            "mnemocore_self_improve_backpressure_skips_total": (
                self.backpressure_skips_total
            ),
            "total_cycles": len(self.cycle_durations),
        }


# ---------------------------------------------------------------------------
# Validation Gates
# ---------------------------------------------------------------------------

class ValidationGates:
    """Five validation gates per SELF_IMPROVEMENT_DEEP_DIVE.md §9."""

    def __init__(self, config: "SelfImprovementConfig"):
        self.min_semantic_similarity = config.min_semantic_similarity
        self.min_improvement_score = config.min_improvement_score
        self.safety_mode = config.safety_mode

    # Gate 1: Semantic drift — proposed content must stay close to original
    def semantic_drift_gate(self, proposal: ImprovementProposal) -> bool:
        """Check that improvement doesn't drift too far from original."""
        # Phase 0: simple character-level Jaccard as proxy (no embeddings)
        original_tokens = set(proposal.candidate.content.lower().split())
        proposed_tokens = set(proposal.proposed_content.lower().split())
        if not original_tokens and not proposed_tokens:
            return True
        union = original_tokens | proposed_tokens
        intersection = original_tokens & proposed_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard >= self.min_semantic_similarity

    # Gate 2: Fact safety — no new unsupported claims
    def fact_safety_gate(self, proposal: ImprovementProposal) -> bool:
        """In strict mode, proposed content must not introduce new factual claims."""
        if self.safety_mode != "strict":
            return True
        # Phase 0 heuristic: proposed content must be a subset or near-subset
        # of original tokens (no new information injected)
        original_tokens = set(proposal.candidate.content.lower().split())
        proposed_tokens = set(proposal.proposed_content.lower().split())
        new_tokens = proposed_tokens - original_tokens
        # Allow up to 10% new tokens (formatting words, articles etc.)
        max_new = max(3, int(len(proposed_tokens) * 0.10))
        return len(new_tokens) <= max_new

    # Gate 3: Structure — must improve readability/compactness
    def structure_gate(self, proposal: ImprovementProposal) -> bool:
        """Check minimum improvement score threshold."""
        return proposal.expected_improvement_score >= self.min_improvement_score

    # Gate 4: Policy — block forbidden metadata changes
    def policy_gate(self, proposal: ImprovementProposal) -> bool:
        """Ensure no forbidden fields are modified."""
        forbidden_fields = {"agent_id", "unix_timestamp", "previous_id", "id"}
        candidate_meta = proposal.candidate.metadata
        # In Phase 0 we don't modify metadata, so always passes
        for fld in forbidden_fields:
            if fld in candidate_meta:
                pass  # read-only check
        return True

    # Gate 5: Resource — cycle budget / backpressure
    def resource_gate(
        self, cycle_elapsed: float, max_cycle_seconds: float
    ) -> bool:
        """Check that cycle hasn't exceeded time budget."""
        return cycle_elapsed < max_cycle_seconds

    def validate(
        self,
        proposal: ImprovementProposal,
        cycle_elapsed: float,
        max_cycle_seconds: float,
    ) -> bool:
        """Run all 5 gates. Records results on the proposal."""
        gates = {
            "semantic_drift": self.semantic_drift_gate(proposal),
            "fact_safety": self.fact_safety_gate(proposal),
            "structure": self.structure_gate(proposal),
            "policy": self.policy_gate(proposal),
            "resource": self.resource_gate(cycle_elapsed, max_cycle_seconds),
        }
        proposal.validator_results = gates

        rejections = [name for name, passed in gates.items() if not passed]
        proposal.rejection_reasons = rejections
        proposal.accepted = len(rejections) == 0
        return proposal.accepted


# ---------------------------------------------------------------------------
# Main Worker
# ---------------------------------------------------------------------------

class SelfImprovementWorker:
    """
    Phase 0 Self-Improvement Worker.

    Runs as a background asyncio task. Each cycle:
      1. Select candidates from tier_manager (HOT tier).
      2. Generate rule-based proposals (normalize, metadata_repair, deduplicate).
      3. Validate proposals through 5 gates.
      4. Log decisions — NO writes (dry-run).
      5. Record metrics.

    Follows the SubconsciousConsolidationWorker lifecycle pattern.
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        config: Optional["SelfImprovementConfig"] = None,
    ):
        from ..core.config import SelfImprovementConfig as _SIC

        self.engine = engine
        self.cfg = config or _SIC()

        # Validation pipeline
        self.gates = ValidationGates(self.cfg)

        # Metrics
        self.metrics = SelfImprovementMetrics()

        # Cooldown tracker: node_id → last_evaluated_timestamp
        self._cooldown: Dict[str, float] = {}

        # Decision log (kept in-memory, capped)
        self._decision_log: List[Dict[str, Any]] = []
        self._max_log_entries = 500

        # Lifecycle
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.last_run: Optional[datetime] = None
        self.stats: Dict[str, Any] = {}

        logger.info(
            f"SelfImprovementWorker initialized — "
            f"enabled={self.cfg.enabled} dry_run={self.cfg.dry_run} "
            f"safety_mode={self.cfg.safety_mode} "
            f"interval={self.cfg.interval_seconds}s"
        )

    # ------------------------------------------------------------------
    # Lifecycle (mirrors ConsolidationWorker)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the background improvement loop."""
        if not self.cfg.enabled:
            logger.info("SelfImprovementWorker disabled by config.")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._improvement_loop(),
            name="self_improvement_worker",
        )
        logger.info(
            f"SelfImprovementWorker started — "
            f"interval={self.cfg.interval_seconds}s "
            f"batch_size={self.cfg.batch_size}"
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
        logger.info("SelfImprovementWorker stopped.")

    async def _improvement_loop(self) -> None:
        """Main loop: sleep → run_once → repeat."""
        while self._running:
            try:
                await asyncio.sleep(self.cfg.interval_seconds)
                if self._running:
                    await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    f"SelfImprovementWorker error: {exc}", exc_info=True
                )
                # Backoff on error
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    async def run_once(self) -> Dict[str, Any]:
        """
        Execute one improvement cycle (dry-run in Phase 0).

        Returns dict with cycle statistics.
        """
        t0 = time.monotonic()
        logger.info("=== Self-Improvement Cycle — start ===")

        cycle_result: Dict[str, Any] = {
            "candidates_found": 0,
            "proposals_generated": 0,
            "proposals_accepted": 0,
            "proposals_rejected": 0,
            "backpressure_skip": False,
            "elapsed_seconds": 0.0,
            "dry_run": self.cfg.dry_run,
            "timestamp": None,
        }

        # ---- Backpressure check ----
        if self._check_backpressure():
            self.metrics.backpressure_skips_total += 1
            cycle_result["backpressure_skip"] = True
            logger.info("Self-Improvement skipped — backpressure detected.")
            return cycle_result

        # ---- Step 1: Candidate Selection ----
        candidates = await self._select_candidates()
        cycle_result["candidates_found"] = len(candidates)
        self.metrics.candidates_per_cycle.append(len(candidates))

        if not candidates:
            logger.info("Self-Improvement: no candidates this cycle.")
            elapsed = time.monotonic() - t0
            cycle_result["elapsed_seconds"] = round(elapsed, 3)
            cycle_result["timestamp"] = datetime.now(timezone.utc).isoformat()
            self.metrics.cycle_durations.append(elapsed)
            self.last_run = datetime.now(timezone.utc)
            self.stats = cycle_result
            return cycle_result

        # ---- Step 2 + 3: Generate & Validate Proposals ----
        accepted_proposals: List[ImprovementProposal] = []
        rejected_proposals: List[ImprovementProposal] = []

        for candidate in candidates:
            cycle_elapsed = time.monotonic() - t0

            # Resource gate pre-check
            if cycle_elapsed >= self.cfg.max_cycle_seconds:
                logger.info(
                    f"Self-Improvement cycle time budget exhausted "
                    f"({cycle_elapsed:.1f}s >= {self.cfg.max_cycle_seconds}s)"
                )
                break

            proposals = self._generate_proposals(candidate)
            self.metrics.attempts_total += len(proposals)

            for proposal in proposals:
                passed = self.gates.validate(
                    proposal, cycle_elapsed, self.cfg.max_cycle_seconds
                )
                if passed:
                    accepted_proposals.append(proposal)
                else:
                    rejected_proposals.append(proposal)

                self._log_decision(proposal)

        cycle_result["proposals_generated"] = len(accepted_proposals) + len(
            rejected_proposals
        )
        cycle_result["proposals_accepted"] = len(accepted_proposals)
        cycle_result["proposals_rejected"] = len(rejected_proposals)

        self.metrics.rejects_total += len(rejected_proposals)

        # ---- Step 4: Commit (Phase 0 = dry-run, NO writes) ----
        if self.cfg.dry_run:
            if accepted_proposals:
                logger.info(
                    f"Self-Improvement DRY-RUN: "
                    f"{len(accepted_proposals)} proposals would be committed. "
                    f"(dry_run=True, no writes performed)"
                )
                for p in accepted_proposals:
                    logger.debug(
                        f"  [DRY-RUN ACCEPT] type={p.improvement_type} "
                        f"node={p.candidate.node_id} "
                        f"score={p.expected_improvement_score:.3f} "
                        f"reason={p.candidate.selection_reason}"
                    )
        else:
            # Phase 1+ would call engine.store() here
            # For now, log a warning if someone disables dry_run prematurely
            logger.warning(
                "SelfImprovementWorker: dry_run=False but Phase 0 "
                "does not support live commits. Skipping writes."
            )

        # ---- Finalize ----
        elapsed = time.monotonic() - t0
        self.last_run = datetime.now(timezone.utc)
        cycle_result["elapsed_seconds"] = round(elapsed, 3)
        cycle_result["timestamp"] = self.last_run.isoformat()

        self.metrics.cycle_durations.append(elapsed)
        quality_delta = sum(
            p.expected_improvement_score for p in accepted_proposals
        ) / max(len(accepted_proposals), 1)
        self.metrics.quality_deltas.append(quality_delta)

        self.stats = cycle_result

        logger.info(
            f"=== Self-Improvement Cycle — done in {elapsed:.2f}s | "
            f"candidates={len(candidates)} "
            f"accepted={len(accepted_proposals)} "
            f"rejected={len(rejected_proposals)} "
            f"dry_run={self.cfg.dry_run} ==="
        )

        # Record to meta_memory if available
        if hasattr(self.engine, "meta_memory") and self.engine.meta_memory:
            try:
                self.engine.meta_memory.record_metric(
                    "self_improvement_cycle",
                    cycle_result["proposals_accepted"],
                    metadata={
                        "rejected": cycle_result["proposals_rejected"],
                        "elapsed": elapsed,
                    },
                )
            except Exception:
                pass

        return cycle_result

    # ------------------------------------------------------------------
    # Candidate Selection
    # ------------------------------------------------------------------

    async def _select_candidates(self) -> List[ImprovementCandidate]:
        """
        Select memory nodes from HOT tier as improvement candidates.

        Heuristics (Phase 0):
          - Short content (potential metadata-only nodes or stubs)
          - Missing metadata fields
          - Near-duplicate content hashes
        """
        candidates: List[ImprovementCandidate] = []
        now = time.time()
        cooldown_seconds = self.cfg.cooldown_minutes * 60

        try:
            # Get recent memories from HOT tier
            hot_memories = self.engine.tier_manager.get_hot_recent(
                limit=self.cfg.batch_size * 4  # over-fetch, then filter
            )
        except Exception as e:
            logger.debug(f"Self-Improvement: could not read HOT tier: {e}")
            return candidates

        # Content hash tracker for near-duplicate detection
        content_hashes: Dict[str, List[str]] = defaultdict(list)

        for mem in hot_memories:
            node_id = mem.get("id", mem.get("node_id", ""))
            content = mem.get("content", "")
            metadata = mem.get("metadata", {})

            if not node_id or not content:
                continue

            # Cooldown check
            if node_id in self._cooldown:
                if now - self._cooldown[node_id] < cooldown_seconds:
                    continue

            # Hash for duplicate detection
            content_hash = hashlib.md5(
                content.strip().lower().encode()
            ).hexdigest()[:12]
            content_hashes[content_hash].append(node_id)

            # Heuristic 1: short content (< 20 chars) likely a stub
            if len(content.strip()) < 20:
                candidates.append(
                    ImprovementCandidate(
                        node_id=node_id,
                        content=content,
                        metadata=metadata,
                        tier="hot",
                        selection_reason="short_content",
                        score=0.3,
                    )
                )
                self._cooldown[node_id] = now
                continue

            # Heuristic 2: missing key metadata fields
            expected_fields = {"agent_id", "importance"}
            missing = expected_fields - set(metadata.keys())
            if missing:
                candidates.append(
                    ImprovementCandidate(
                        node_id=node_id,
                        content=content,
                        metadata=metadata,
                        tier="hot",
                        selection_reason=f"missing_metadata:{','.join(missing)}",
                        score=0.2,
                    )
                )
                self._cooldown[node_id] = now

            if len(candidates) >= self.cfg.batch_size:
                break

        # Heuristic 3: near-duplicates (same content hash)
        for chash, node_ids in content_hashes.items():
            if len(node_ids) > 1 and len(candidates) < self.cfg.batch_size:
                # Mark the first one as a dedup candidate
                for nid in node_ids[1:]:
                    if len(candidates) >= self.cfg.batch_size:
                        break
                    candidates.append(
                        ImprovementCandidate(
                            node_id=nid,
                            content="",  # filled later if needed
                            metadata={},
                            tier="hot",
                            selection_reason=f"near_duplicate_of:{node_ids[0]}",
                            score=0.4,
                        )
                    )
                    self._cooldown[nid] = now

        return candidates[: self.cfg.batch_size]

    # ------------------------------------------------------------------
    # Proposal Generation (Phase 0: rule-based only, no LLM)
    # ------------------------------------------------------------------

    def _generate_proposals(
        self, candidate: ImprovementCandidate
    ) -> List[ImprovementProposal]:
        """Generate rule-based improvement proposals for a candidate."""
        proposals: List[ImprovementProposal] = []

        if candidate.selection_reason == "short_content":
            proposals.append(self._propose_metadata_repair(candidate))

        elif candidate.selection_reason.startswith("missing_metadata"):
            proposals.append(self._propose_metadata_repair(candidate))

        elif candidate.selection_reason.startswith("near_duplicate"):
            proposals.append(self._propose_deduplicate(candidate))

        else:
            # Default: normalize
            proposals.append(self._propose_normalize(candidate))

        return proposals

    def _propose_normalize(
        self, candidate: ImprovementCandidate
    ) -> ImprovementProposal:
        """Normalize whitespace, casing, formatting."""
        content = candidate.content
        # Simple normalization: collapse whitespace, strip
        normalized = " ".join(content.split()).strip()
        improvement = (
            abs(len(content) - len(normalized)) / max(len(content), 1)
        )
        return ImprovementProposal(
            candidate=candidate,
            improvement_type="normalize",
            proposed_content=normalized,
            rationale="Normalize whitespace and formatting",
            expected_improvement_score=min(improvement + 0.1, 1.0),
        )

    def _propose_metadata_repair(
        self, candidate: ImprovementCandidate
    ) -> ImprovementProposal:
        """Propose adding missing metadata fields."""
        return ImprovementProposal(
            candidate=candidate,
            improvement_type="metadata_repair",
            proposed_content=candidate.content,  # content unchanged
            rationale=(
                f"Repair missing metadata: {candidate.selection_reason}"
            ),
            expected_improvement_score=0.2,
        )

    def _propose_deduplicate(
        self, candidate: ImprovementCandidate
    ) -> ImprovementProposal:
        """Propose marking a duplicate for supersedence."""
        return ImprovementProposal(
            candidate=candidate,
            improvement_type="deduplicate",
            proposed_content=candidate.content,
            rationale=(
                f"Near-duplicate detected: {candidate.selection_reason}"
            ),
            expected_improvement_score=0.4,
        )

    # ------------------------------------------------------------------
    # Decision Logging
    # ------------------------------------------------------------------

    def _log_decision(self, proposal: ImprovementProposal) -> None:
        """Record a decision for audit trail."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_id": proposal.candidate.node_id,
            "tier": proposal.candidate.tier,
            "improvement_type": proposal.improvement_type,
            "selection_reason": proposal.candidate.selection_reason,
            "expected_score": proposal.expected_improvement_score,
            "accepted": proposal.accepted,
            "gate_results": proposal.validator_results,
            "rejection_reasons": proposal.rejection_reasons,
            "dry_run": self.cfg.dry_run,
        }
        self._decision_log.append(entry)

        # Cap log size
        if len(self._decision_log) > self._max_log_entries:
            self._decision_log = self._decision_log[-self._max_log_entries:]

        level = "DEBUG" if proposal.accepted else "TRACE"
        logger.log(
            level,
            f"SelfImprovement decision: "
            f"{'ACCEPT' if proposal.accepted else 'REJECT'} "
            f"type={proposal.improvement_type} "
            f"node={proposal.candidate.node_id} "
            f"score={proposal.expected_improvement_score:.3f} "
            f"reasons={proposal.rejection_reasons}"
        )

    # ------------------------------------------------------------------
    # Backpressure
    # ------------------------------------------------------------------

    def _check_backpressure(self) -> bool:
        """
        Check if the main engine is under load and we should skip.
        Phase 0: simple heuristic based on available attributes.
        """
        try:
            # If engine has a pending queue, check depth
            if hasattr(self.engine, "_subconscious_queue"):
                q = self.engine._subconscious_queue
                if hasattr(q, "qsize") and q.qsize() > 50:
                    return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return combined stats and metrics snapshot."""
        return {
            "last_run": (
                self.last_run.isoformat() if self.last_run else None
            ),
            "running": self._running,
            "config": {
                "enabled": self.cfg.enabled,
                "dry_run": self.cfg.dry_run,
                "safety_mode": self.cfg.safety_mode,
                "interval_seconds": self.cfg.interval_seconds,
                "batch_size": self.cfg.batch_size,
            },
            "metrics": self.metrics.snapshot(),
            "decision_log_size": len(self._decision_log),
            "cooldown_entries": len(self._cooldown),
            "last_cycle": self.stats,
        }

    def get_decision_log(
        self, last_n: int = 50
    ) -> List[Dict[str, Any]]:
        """Return recent decision log entries."""
        return self._decision_log[-last_n:]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_self_improvement_worker(
    engine: "HAIMEngine",
) -> SelfImprovementWorker:
    """
    Create a SelfImprovementWorker using engine config.

    Args:
        engine: HAIMEngine instance.

    Returns:
        Configured SelfImprovementWorker instance.
    """
    from ..core.config import get_config

    config = get_config()
    si_config = getattr(config, "self_improvement", None)
    return SelfImprovementWorker(engine, config=si_config)
