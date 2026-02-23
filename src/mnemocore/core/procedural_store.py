"""
Procedural Store Service
========================
Manages actionable skills, procedural routines, and agentic workflows.
Validates triggering patterns and tracks execution success rates dynamically.
"""

from typing import Dict, List, Optional
import threading
import logging
from datetime import datetime, timezone

from .memory_model import Procedure

logger = logging.getLogger(__name__)


class ProceduralStoreService:
    def __init__(self):
        # Local dictionary for Procedures mapping by ID
        # Would typically be serialized to SQLite, JSON, or Qdrant for retrieval.
        self._procedures: Dict[str, Procedure] = {}
        self._lock = threading.RLock()

    def store_procedure(self, proc: Procedure) -> None:
        """Save a new or refined procedure into memory."""
        with self._lock:
            proc.updated_at = datetime.now(timezone.utc)
            self._procedures[proc.id] = proc
            logger.info(f"Stored procedure {proc.id} ('{proc.name}')")

    def get_procedure(self, proc_id: str) -> Optional[Procedure]:
        """Retrieve a procedure by exact ID."""
        with self._lock:
            return self._procedures.get(proc_id)

    def find_applicable_procedures(
        self, query: str, agent_id: Optional[str] = None, top_k: int = 5
    ) -> List[Procedure]:
        """
        Find procedures whose trigger tags or trigger pattern matches the user intent.
        Simple local text-matching for the prototype layout.
        """
        with self._lock:
            q_lower = query.lower()
            results = []
            for proc in self._procedures.values():
                # Prefer procedures meant directly for this agent, or system globals
                if proc.created_by_agent is not None and agent_id and proc.created_by_agent != agent_id:
                    continue

                if proc.trigger_pattern.lower() in q_lower or any(t.lower() in q_lower for t in proc.tags):
                    results.append(proc)

            # Sort by reliability and usage history to surface most competent tools
            results.sort(key=lambda p: (p.reliability, p.success_count), reverse=True)
            return results[:top_k]

    def record_procedure_outcome(self, proc_id: str, success: bool) -> None:
        """Update procedure success metrics, affecting overall reliability."""
        with self._lock:
            proc = self._procedures.get(proc_id)
            if not proc:
                return

            proc.updated_at = datetime.now(timezone.utc)
            if success:
                proc.success_count += 1
                # Increase reliability slightly on success
                proc.reliability = min(1.0, proc.reliability + 0.05)
            else:
                proc.failure_count += 1
                # Decrease reliability heavily on failure
                proc.reliability = max(0.0, proc.reliability - 0.1)

            logger.debug(f"Procedure {proc_id} outcome recorded: success={success}, new rel={proc.reliability:.2f}")

