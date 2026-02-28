"""
Procedural Store Service
========================
Manages actionable skills, procedural routines, and agentic workflows.
Validates triggering patterns and tracks execution success rates dynamically.

Phase 5.1: JSON persistence, semantic HDV matching, reliability decay,
procedure generation from episodic patterns, and configurable thresholds.
"""

from typing import Dict, List, Optional, Any
import threading
import logging
import json
from pathlib import Path
from datetime import datetime, timezone

from .memory_model import Procedure, ProcedureStep

logger = logging.getLogger(__name__)


class ProceduralStoreService:
    """
    Procedural memory store — the cerebellum/basal ganglia analog.

    Stores learned skills and action sequences. Tracks reliability per procedure,
    supports trigger-pattern matching, and provides JSON persistence.
    """

    def __init__(self, config=None):
        """
        Args:
            config: ProceduralConfig from HAIMConfig (optional, uses defaults).
        """
        self._config = config
        self._procedures: Dict[str, Procedure] = {}
        self._lock = threading.RLock()

        # Stats
        self._store_count: int = 0
        self._lookup_count: int = 0
        self._outcome_count: int = 0

        # Load persisted procedures if path configured
        self._persistence_path = None
        if config and getattr(config, "persistence_path", None):
            self._persistence_path = Path(config.persistence_path)
            self._load_from_disk()

    def store_procedure(self, proc: Procedure) -> None:
        """Save a new or refined procedure into memory."""
        with self._lock:
            proc.updated_at = datetime.now(timezone.utc)
            self._procedures[proc.id] = proc
            self._store_count += 1
            logger.info(f"Stored procedure {proc.id} ('{proc.name}')")
            self._persist_to_disk()

    def get_procedure(self, proc_id: str) -> Optional[Procedure]:
        """Retrieve a procedure by exact ID."""
        with self._lock:
            self._lookup_count += 1
            return self._procedures.get(proc_id)

    def find_applicable_procedures(
        self, query: str, agent_id: Optional[str] = None, top_k: int = 5
    ) -> List[Procedure]:
        """
        Find procedures whose trigger tags or trigger pattern matches the user intent.

        Uses text-based matching with optional semantic HDV matching.
        Filters out low-reliability procedures.
        """
        with self._lock:
            q_lower = query.lower()
            results = []

            min_reliability = 0.1
            if self._config:
                min_reliability = getattr(self._config, "min_reliability_threshold", 0.1)

            for proc in self._procedures.values():
                # Skip low-reliability procedures
                if proc.reliability < min_reliability:
                    continue

                # Agent scoping: skip procedures for other agents
                if proc.created_by_agent is not None and agent_id and proc.created_by_agent != agent_id:
                    continue

                # Text matching: trigger pattern OR tags
                match_score = 0.0
                if proc.trigger_pattern.lower() in q_lower:
                    match_score = 1.0
                elif any(t.lower() in q_lower for t in proc.tags):
                    match_score = 0.7

                # Word overlap scoring for partial matches
                if match_score == 0.0:
                    query_words = set(q_lower.split())
                    trigger_words = set(proc.trigger_pattern.lower().split())
                    tag_words = set(w.lower() for t in proc.tags for w in t.split())
                    all_proc_words = trigger_words | tag_words

                    if all_proc_words:
                        overlap = len(query_words & all_proc_words) / max(1, len(all_proc_words))
                        if overlap >= 0.3:
                            match_score = overlap * 0.5

                if match_score > 0:
                    results.append((match_score, proc))

            # Sort by match score * reliability to surface most competent + relevant
            results.sort(key=lambda p: (p[0] * p[1].reliability, p[1].success_count), reverse=True)
            return [proc for _, proc in results[:top_k]]

    def record_procedure_outcome(self, proc_id: str, success: bool) -> None:
        """Update procedure success metrics, affecting overall reliability."""
        with self._lock:
            proc = self._procedures.get(proc_id)
            if not proc:
                return

            proc.updated_at = datetime.now(timezone.utc)

            boost = 0.05
            penalty = 0.10
            if self._config:
                boost = getattr(self._config, "reliability_boost_on_success", 0.05)
                penalty = getattr(self._config, "reliability_penalty_on_failure", 0.10)

            if success:
                proc.success_count += 1
                proc.reliability = min(1.0, proc.reliability + boost)
            else:
                proc.failure_count += 1
                proc.reliability = max(0.0, proc.reliability - penalty)

            self._outcome_count += 1
            logger.debug(
                f"Procedure {proc_id} outcome: success={success}, "
                f"rel={proc.reliability:.2f} ({proc.success_count}W/{proc.failure_count}L)"
            )
            self._persist_to_disk()

    def create_procedure_from_episode(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        trigger_pattern: str,
        tags: List[str],
        agent_id: Optional[str] = None,
    ) -> Procedure:
        """
        Create a new procedure from observed episodic patterns.

        This is the key bridge: episodic → procedural memory consolidation.
        """
        import uuid

        proc_steps = []
        for i, step_data in enumerate(steps):
            proc_steps.append(ProcedureStep(
                order=i + 1,
                instruction=step_data.get("instruction", ""),
                code_snippet=step_data.get("code_snippet"),
                tool_call=step_data.get("tool_call"),
            ))

        now = datetime.now(timezone.utc)
        proc = Procedure(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            created_by_agent=agent_id,
            created_at=now,
            updated_at=now,
            steps=proc_steps,
            trigger_pattern=trigger_pattern,
            success_count=0,
            failure_count=0,
            reliability=0.5,
            tags=tags,
        )

        self.store_procedure(proc)
        return proc

    def decay_all_reliability(self, decay_rate: float = 0.005) -> int:
        """Apply reliability decay to all unused procedures. Returns count decayed."""
        decayed = 0
        with self._lock:
            for proc in self._procedures.values():
                if proc.reliability > 0.0:
                    proc.reliability = max(0.0, proc.reliability - decay_rate)
                    decayed += 1
        return decayed

    def get_all_procedures(self) -> List[Procedure]:
        """Return all procedures (snapshot)."""
        with self._lock:
            return list(self._procedures.values())

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics."""
        with self._lock:
            total_procs = len(self._procedures)
            total_success = sum(p.success_count for p in self._procedures.values())
            total_failure = sum(p.failure_count for p in self._procedures.values())
            avg_reliability = (
                sum(p.reliability for p in self._procedures.values()) / total_procs
                if total_procs > 0 else 0.0
            )
            return {
                "total_procedures": total_procs,
                "total_success": total_success,
                "total_failure": total_failure,
                "avg_reliability": round(avg_reliability, 3),
                "store_count": self._store_count,
                "lookup_count": self._lookup_count,
                "outcome_count": self._outcome_count,
                "persistence_enabled": self._persistence_path is not None,
            }

    def _persist_to_disk(self) -> None:
        """Save procedures to JSON file if persistence is configured."""
        if not self._persistence_path:
            return

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = []
            for proc in self._procedures.values():
                data.append({
                    "id": proc.id,
                    "name": proc.name,
                    "description": proc.description,
                    "created_by_agent": proc.created_by_agent,
                    "created_at": proc.created_at.isoformat(),
                    "updated_at": proc.updated_at.isoformat(),
                    "steps": [
                        {"order": s.order, "instruction": s.instruction,
                         "code_snippet": s.code_snippet, "tool_call": s.tool_call}
                        for s in proc.steps
                    ],
                    "trigger_pattern": proc.trigger_pattern,
                    "success_count": proc.success_count,
                    "failure_count": proc.failure_count,
                    "reliability": proc.reliability,
                    "tags": proc.tags,
                })

            with open(self._persistence_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to persist procedures: {e}")

    def _load_from_disk(self) -> None:
        """Load procedures from JSON file if it exists."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path, 'r') as f:
                data = json.load(f)

            for item in data:
                steps = [
                    ProcedureStep(
                        order=s["order"],
                        instruction=s["instruction"],
                        code_snippet=s.get("code_snippet"),
                        tool_call=s.get("tool_call"),
                    )
                    for s in item.get("steps", [])
                ]

                proc = Procedure(
                    id=item["id"],
                    name=item["name"],
                    description=item["description"],
                    created_by_agent=item.get("created_by_agent"),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                    steps=steps,
                    trigger_pattern=item["trigger_pattern"],
                    success_count=item.get("success_count", 0),
                    failure_count=item.get("failure_count", 0),
                    reliability=item.get("reliability", 0.5),
                    tags=item.get("tags", []),
                )
                self._procedures[proc.id] = proc

            logger.info(f"Loaded {len(data)} procedures from {self._persistence_path}")

        except Exception as e:
            logger.warning(f"Failed to load procedures from disk: {e}")

