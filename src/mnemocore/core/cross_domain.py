"""
Cross-Domain Association Builder (Phase 5.0 — Agent 3)
=======================================================
Automatically links memories across three semantic domains:

    strategic   – goals, decisions, roadmaps, strategies
    operational – code, bugs, documentation, tasks
    personal    – preferences, habits, relationships, context

Cross-domain synapses improve holistic reasoning: when a strategic
goal changes, the system can surface related operational tasks or
personal context without being explicitly queried.

Implementation:
  - Each memory is tagged with a `domain` in its metadata (or inferred)
  - CrossDomainSynapseBuilder monitors recently stored memories
  - Co-occurrence within a time window → create a cross-domain synapse
  - Synapse weight is damped (0.2×) relative to intra-domain (1.0×)

Integration with RippleContext:
  - ripple_context.py uses domain_weight when propagating context
  - Cross-domain propagation uses CROSS_DOMAIN_WEIGHT as the multiplier

Public API:
    builder = CrossDomainSynapseBuilder(engine)
    await builder.process_new_memory(node)  # call after /store
    pairs = await builder.scan_recent(hours=1)  # background scan
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple
from loguru import logger

if TYPE_CHECKING:
    from .node import MemoryNode


# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

DOMAINS = {"strategic", "operational", "personal"}
DEFAULT_DOMAIN = "operational"

# Weight applied to cross-domain synapses (vs 1.0 for intra-domain)
CROSS_DOMAIN_WEIGHT: float = 0.2

# Time window for co-occurrence detection (hours)
COOCCURRENCE_WINDOW_HOURS: float = 2.0

# Keywords used to infer domain automatically if not tagged
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "strategic": [
        "goal", "strategy", "roadmap", "vision", "mission", "objective",
        "decision", "priority", "kpi", "okr", "plan", "budget", "market",
    ],
    "personal": [
        "prefer", "habit", "feel", "emotion", "prefer", "like", "dislike",
        "relationship", "trust", "colleague", "friend", "name", "remember me",
    ],
    "operational": [
        "code", "bug", "fix", "implement", "test", "deploy", "api",
        "function", "class", "module", "error", "exception", "task", "ticket",
    ],
}


# ------------------------------------------------------------------ #
#  Domain inference                                                  #
# ------------------------------------------------------------------ #

def infer_domain(content: str, metadata: Optional[Dict] = None) -> str:
    """
    Infer the semantic domain of a memory from its content and metadata.

    Priority:
    1. metadata["domain"] if set
    2. keyword match in content (highest score wins)
    3. DEFAULT_DOMAIN ("operational")
    """
    if metadata and "domain" in metadata:
        d = metadata["domain"].lower()
        return d if d in DOMAINS else DEFAULT_DOMAIN

    content_lower = content.lower()
    best_domain = DEFAULT_DOMAIN
    best_count = 0

    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in content_lower)
        if count > best_count:
            best_count = count
            best_domain = domain

    return best_domain


# ------------------------------------------------------------------ #
#  CrossDomainSynapseBuilder                                         #
# ------------------------------------------------------------------ #

class CrossDomainSynapseBuilder:
    """
    Detects cross-domain co-occurrences and requests synapse creation.

    Works by maintaining a rolling buffer of recently stored memories,
    then pairing memories from different domains that appeared within
    COOCCURRENCE_WINDOW_HOURS of each other.
    """

    def __init__(
        self,
        engine=None,  # HAIMEngine
        window_hours: float = COOCCURRENCE_WINDOW_HOURS,
        cross_domain_weight: float = CROSS_DOMAIN_WEIGHT,
    ) -> None:
        self.engine = engine
        self.window = timedelta(hours=window_hours)
        self.weight = cross_domain_weight
        # Buffer: list of (node_id, domain, stored_at)
        self._buffer: List[Tuple[str, str, datetime]] = []

    # ---- Domain helpers ------------------------------------------ #

    def tag_domain(self, node: "MemoryNode") -> str:
        """Infer and write domain tag to node.metadata. Returns domain string."""
        domain = infer_domain(node.content, getattr(node, "metadata", {}))
        if hasattr(node, "metadata"):
            node.metadata["domain"] = domain
        return domain

    # ---- Synapse creation --------------------------------------- #

    async def _create_synapse(self, id_a: str, id_b: str) -> None:
        """
        Request synapse creation between two nodes via the engine's synapse index.
        Weight is damped by CROSS_DOMAIN_WEIGHT.
        """
        if self.engine is None:
            logger.debug(f"CrossDomain: no engine, skipping synapse {id_a[:8]} ↔ {id_b[:8]}")
            return
        try:
            synapse_index = getattr(self.engine, "synapse_index", None)
            if synapse_index is not None:
                synapse_index.add_or_strengthen(id_a, id_b, delta=self.weight)
                logger.debug(
                    f"CrossDomain synapse created: {id_a[:8]} ↔ {id_b[:8]} weight={self.weight}"
                )
        except Exception as exc:
            logger.debug(f"CrossDomain synapse creation failed: {exc}")

    # ---- Main API ----------------------------------------------- #

    async def process_new_memory(self, node: "MemoryNode") -> List[Tuple[str, str]]:
        """
        Called after a new memory is stored.
        Tags its domain and checks for cross-domain co-occurrences in the buffer.

        Returns list of (id_a, id_b) pairs for which synapses were created.
        """
        domain = self.tag_domain(node)
        now = datetime.now(timezone.utc)

        # Cut stale entries from buffer
        cutoff = now - self.window
        self._buffer = [(nid, d, ts) for nid, d, ts in self._buffer if ts >= cutoff]

        # Find cross-domain pairs with current node
        pairs: List[Tuple[str, str]] = []
        already_seen: Set[str] = set()
        for existing_id, existing_domain, _ts in self._buffer:
            if existing_domain != domain and existing_id not in already_seen:
                await self._create_synapse(node.id, existing_id)
                pairs.append((node.id, existing_id))
                already_seen.add(existing_id)

        # Add current node to buffer
        self._buffer.append((node.id, domain, now))

        if pairs:
            logger.info(
                f"CrossDomain: {len(pairs)} cross-domain synapses created for node {node.id[:8]} (domain={domain})"
            )
        return pairs

    async def scan_recent(self, hours: float = 1.0) -> List[Tuple[str, str]]:
        """
        Scan the current buffer for any unpaired cross-domain co-occurrences.
        Returns all cross-domain pairs.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        recent = [(nid, d, ts) for nid, d, ts in self._buffer if ts >= cutoff]

        pairs: List[Tuple[str, str]] = []
        n = len(recent)
        for i in range(n):
            for j in range(i + 1, n):
                id_i, dom_i, _ = recent[i]
                id_j, dom_j, _ = recent[j]
                if dom_i != dom_j:
                    await self._create_synapse(id_i, id_j)
                    pairs.append((id_i, id_j))

        return pairs

    def clear_buffer(self) -> None:
        """Reset the co-occurrence buffer."""
        self._buffer.clear()
