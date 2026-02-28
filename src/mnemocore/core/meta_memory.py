"""
Meta Memory Service
===================
Maintains a self-model of the memory substrate, gathering metrics and surfacing self-improvement proposals.
Plays a crucial role in enabling an AGI system to observe and upgrade its own thinking architectures over time.
"""

from typing import Dict, List, Optional
import threading
import logging
from datetime import datetime, timezone

from .memory_model import SelfMetric, SelfImprovementProposal

logger = logging.getLogger(__name__)


class MetaMemoryService:
    def __init__(self, config=None):
        self._config = config
        self._metrics: List[SelfMetric] = []
        self._proposals: Dict[str, SelfImprovementProposal] = {}
        self._lock = threading.RLock()

    def record_metric(self, name: str, value: float, window: str) -> None:
        """Log a new performance or algorithmic metric reading."""
        with self._lock:
            # We strictly bind this to metrics history for Subconscious AI trend analysis.
            metric = SelfMetric(
                name=name, value=value, window=window, updated_at=datetime.now(timezone.utc)
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

    async def generate_proposals_from_metrics(self, engine) -> Optional[str]:
        """
        Analyze recent metrics. Only invoke the LLM (SubconsciousAI) if anomalies are detected,
        ensuring minimal CPU overhead.
        """
        # 1. Analyze for anomalies (cheap operation to save CPU)
        anomalies = []
        with self._lock:
            # Check last 50 metrics
            recent = self._metrics[-50:]
            for m in recent:
                if "failure_rate" in m.name and m.value > 0.1:
                    anomalies.append(f"High failure rate: {m.value*100}% in {m.window}")
                elif "hit_rate" in m.name and m.value < 0.5:
                    anomalies.append(f"Low cache hit rate: {m.value*100}% in {m.window}")
                elif "latency" in m.name and m.value > 1000:
                    anomalies.append(f"High latency spike: {m.value}ms in {m.window}")
                    
        # Remove duplicates
        anomalies = list(set(anomalies))
        
        # 2. If no issues, skip heavy LLM processing
        if not anomalies:
            return None
            
        # 3. If we have the engine and LLM subsystem, generate proposal
        if not hasattr(engine, "subconscious") or not engine.subconscious:
            logger.debug("SubconsciousAI unavailable for meta-reflection.")
            return None
            
        prompt = (
            "You are the Meta-Cognitive module of MnemoCore. Analyze these anomalies "
            f"and propose an architectural or configuration improvement:\n{anomalies}\n"
            "Respond in strictly structured JSON: {\"title\": \"...\", \"rationale\": \"...\", \"expected_effect\": \"...\"}"
        )
        
        try:
            # We use analyze_dream directly to bypass store tracking, just a raw LLM inference
            response = await engine.subconscious.analyze_dream(prompt, [])
            if not response:
                return None
                
            import json as _json
            import re
            import uuid
            
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return None
                
            parsed = _json.loads(match.group())
            
            proposal = SelfImprovementProposal(
                id=f"prop_{uuid.uuid4().hex[:8]}",
                created_at=datetime.now(timezone.utc),
                author="system",
                title=parsed.get("title", "Optimization Proposal"),
                description=str(anomalies),
                rationale=parsed.get("rationale", "Address observed anomalies."),
                expected_effect=parsed.get("expected_effect", "Stabilize metrics."),
                status="pending",
                metadata={"anomalies": anomalies}
            )
            
            return self.create_proposal(proposal)
            
        except Exception as e:
            logger.error(f"Failed to generate meta-proposal: {e}")
            return None


