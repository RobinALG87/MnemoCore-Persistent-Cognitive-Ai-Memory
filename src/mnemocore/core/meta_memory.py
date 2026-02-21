"""
Meta Memory Service
===================
Maintains a self-model of the memory substrate, gathering metrics and surfacing self-improvement proposals.
Plays a crucial role in enabling an AGI system to observe and upgrade its own thinking architectures over time.
"""

from typing import Dict, List, Optional
import threading
import logging
from datetime import datetime

from .memory_model import SelfMetric, SelfImprovementProposal

logger = logging.getLogger(__name__)


class MetaMemoryService:
    def __init__(self):
        self._metrics: List[SelfMetric] = []
        self._proposals: Dict[str, SelfImprovementProposal] = {}
        self._lock = threading.RLock()

    def record_metric(self, name: str, value: float, window: str) -> None:
        """Log a new performance or algorithmic metric reading."""
        with self._lock:
            # We strictly bind this to metrics history for Subconscious AI trend analysis.
            metric = SelfMetric(
                name=name, value=value, window=window, updated_at=datetime.utcnow()
            )
            self._metrics.append(metric)

            # Cap local metrics storage bounds 
            if len(self._metrics) > 10000:
                self._metrics = self._metrics[-5000:]
                
            logger.debug(f"Recorded meta-metric: {name}={value} ({window})")

    def list_metrics(self, limit: int = 100, window: Optional[str] = None) -> List[SelfMetric]:
        """Fetch historical metric footprints."""
        with self._lock:
            filtered = [m for m in self._metrics if (not window) or m.window == window]
            filtered.sort(key=lambda x: x.updated_at, reverse=True)
            return filtered[:limit]

    def create_proposal(self, proposal: SelfImprovementProposal) -> str:
        """Inject a formally modeled improvement prompt into the queue."""
        with self._lock:
            self._proposals[proposal.id] = proposal
            logger.info(f"New self-improvement proposal created by {proposal.author}: {proposal.title}")
            return proposal.id

    def update_proposal_status(self, proposal_id: str, status: str) -> None:
        """Mark a proposal as accepted, rejected, or implemented by the oversight entity."""
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if not proposal:
                logger.warning(f"Could not update unknown proposal ID: {proposal_id}")
                return
            
            proposal.status = status # type: ignore
            logger.info(f"Proposal {proposal_id} status escalated to: {status}")

    def list_proposals(self, status: Optional[str] = None) -> List[SelfImprovementProposal]:
        """Retrieve proposals matching a given state."""
        with self._lock:
            if status:
                return [p for p in self._proposals.values() if p.status == status]
            return list(self._proposals.values())

