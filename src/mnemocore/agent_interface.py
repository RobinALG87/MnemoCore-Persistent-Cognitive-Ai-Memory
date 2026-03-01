"""
Cognitive Memory Client
=======================
The high-level facade for autonomous agents to interact with the MnemoCore AGI Memory Substrate.
Provides easy methods for observation, episodic sequence tracking, and working memory recall.

All primary methods are async for consistency. Sync wrappers are provided for convenience.
"""

from typing import List, Optional, Any, Tuple
import asyncio
import logging

from .core.engine import HAIMEngine
from .core.working_memory import WorkingMemoryService, WorkingMemoryItem
from .core.episodic_store import EpisodicStoreService
from .core.semantic_store import SemanticStoreService
from .core.procedural_store import ProceduralStoreService
from .core.meta_memory import MetaMemoryService, SelfImprovementProposal
from .core.memory_model import Procedure

logger = logging.getLogger(__name__)


def _run_sync(coro):
    """
    Helper to run an async coroutine synchronously.
    Used by sync wrapper methods.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're in an async context, create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # No running loop, we can use asyncio.run
        return asyncio.run(coro)


class CognitiveMemoryClient:
    """
    Plug-and-play cognitive memory facade for agent frameworks (LangGraph, AutoGen, OpenClaw, etc.).

    All primary methods are async for consistency with the underlying async engine.
    Sync wrapper methods (suffixed with _sync) are provided for legacy compatibility.
    """

    def __init__(
        self,
        engine: HAIMEngine,
        wm: WorkingMemoryService,
        episodic: EpisodicStoreService,
        semantic: SemanticStoreService,
        procedural: ProceduralStoreService,
        meta: MetaMemoryService,
    ):
        self.engine = engine
        self.wm = wm
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.meta = meta

    # --- Observation & WM ---

    async def observe(
        self,
        agent_id: str,
        content: str,
        kind: str = "observation",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        **meta
    ) -> str:
        """
        Push a new observation or thought directly into the agent's short-term Working Memory.
        Async version.
        """
        import uuid
        from datetime import datetime, timezone
        item_id = f"wm_{uuid.uuid4().hex[:8]}"

        item = WorkingMemoryItem(
            id=item_id,
            agent_id=agent_id,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=3600,  # 1 hour default
            content=content,
            kind=kind,  # type: ignore
            importance=importance,
            tags=tags or [],
            hdv=None  # Could encode via engine in future
        )
        await self.wm.push_item(agent_id, item)
        logger.debug(f"Agent {agent_id} observed: {content[:30]}...")
        return item_id

    def observe_sync(
        self,
        agent_id: str,
        content: str,
        kind: str = "observation",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        **meta
    ) -> str:
        """Sync wrapper for observe()."""
        return _run_sync(self.observe(
            agent_id=agent_id,
            content=content,
            kind=kind,
            importance=importance,
            tags=tags,
            **meta
        ))

    async def get_working_context(
        self,
        agent_id: str,
        limit: int = 16
    ) -> List[WorkingMemoryItem]:
        """
        Read the active, un-pruned context out of the agent's working memory buffer.
        Async version.
        """
        state = await self.wm.get_state(agent_id)
        if not state:
            return []

        return state.items[-limit:]

    def get_working_context_sync(
        self,
        agent_id: str,
        limit: int = 16
    ) -> List[WorkingMemoryItem]:
        """Sync wrapper for get_working_context()."""
        return _run_sync(self.get_working_context(agent_id, limit=limit))

    # --- Episodic ---

    async def start_episode(
        self,
        agent_id: str,
        goal: str,
        context: Optional[str] = None
    ) -> str:
        """Begin a new temporally-linked event sequence. Async version."""
        return self.episodic.start_episode(agent_id, goal=goal, context=context)

    def start_episode_sync(
        self,
        agent_id: str,
        goal: str,
        context: Optional[str] = None
    ) -> str:
        """Sync wrapper for start_episode()."""
        return _run_sync(self.start_episode(agent_id, goal=goal, context=context))

    async def append_event(
        self,
        episode_id: str,
        kind: str,
        content: str,
        **meta
    ) -> None:
        """Log an action or outcome to an ongoing episode. Async version."""
        self.episodic.append_event(episode_id, kind, content, meta)

    def append_event_sync(
        self,
        episode_id: str,
        kind: str,
        content: str,
        **meta
    ) -> None:
        """Sync wrapper for append_event()."""
        return _run_sync(self.append_event(episode_id, kind, content, **meta))

    async def end_episode(
        self,
        episode_id: str,
        outcome: str,
        reward: Optional[float] = None
    ) -> None:
        """Seal an episode, logging its final success or failure state. Async version."""
        self.episodic.end_episode(episode_id, outcome, reward)

    def end_episode_sync(
        self,
        episode_id: str,
        outcome: str,
        reward: Optional[float] = None
    ) -> None:
        """Sync wrapper for end_episode()."""
        return _run_sync(self.end_episode(episode_id, outcome, reward))

    # --- Semantic / Retrieval ---

    async def recall(
        self,
        agent_id: str,
        query: str,
        context: Optional[str] = None,
        top_k: int = 8,
        modes: Tuple[str, ...] = ("episodic", "semantic")
    ) -> List[dict]:
        """
        A unified query interface that checks Working Memory, Episodic History, and the Semantic Vector Store.
        Currently delegates heavily to the backing HAIMEngine, but can be augmented to return semantic concepts.
        """
        results = []

        # 1. Broad retrieval via existing HAIM engine (SM / general memories)
        if "semantic" in modes:
            engine_results = await self.engine.query(query, top_k=top_k)
            for mem_id, score in engine_results:
                node = await self.engine.tier_manager.get_memory(mem_id)
                if node:
                    results.append({"source": "semantic/engine", "content": node.content, "score": score})

        # 2. Local episodic retrieval
        if "episodic" in modes:
            recent_eps = self.episodic.get_recent(agent_id, limit=top_k, context=context)
            for ep in recent_eps:
                summary = f"Episode(goal={ep.goal}, outcome={ep.outcome}, events={len(ep.events)})"
                results.append({"source": "episodic", "content": summary, "score": ep.reliability})

        # Sort and trim mixed results
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:top_k]

    def recall_sync(
        self,
        agent_id: str,
        query: str,
        context: Optional[str] = None,
        top_k: int = 8,
        modes: Tuple[str, ...] = ("episodic", "semantic")
    ) -> List[dict]:
        """Sync wrapper for recall()."""
        return _run_sync(self.recall(
            agent_id=agent_id,
            query=query,
            context=context,
            top_k=top_k,
            modes=modes
        ))

    # --- Procedural ---

    async def suggest_procedures(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Procedure]:
        """Fetch executable tool-patterns based on the agent's intent. Async version."""
        return self.procedural.find_applicable_procedures(query, agent_id=agent_id, top_k=top_k)

    def suggest_procedures_sync(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Procedure]:
        """Sync wrapper for suggest_procedures()."""
        return _run_sync(self.suggest_procedures(agent_id=agent_id, query=query, top_k=top_k))

    async def record_procedure_outcome(self, proc_id: str, success: bool) -> None:
        """Report on the utility of a chosen procedure. Async version."""
        self.procedural.record_procedure_outcome(proc_id, success)

    def record_procedure_outcome_sync(self, proc_id: str, success: bool) -> None:
        """Sync wrapper for record_procedure_outcome()."""
        return _run_sync(self.record_procedure_outcome(proc_id, success))

    # --- Meta / Self-awareness ---

    async def get_knowledge_gaps(self, agent_id: str, lookback_hours: int = 24) -> List[dict]:
        """Return currently open knowledge gaps identified by the Pulse loop. Async version."""
        # Stubbed: Would interact with gap_detector
        return []

    def get_knowledge_gaps_sync(self, agent_id: str, lookback_hours: int = 24) -> List[dict]:
        """Sync wrapper for get_knowledge_gaps()."""
        return _run_sync(self.get_knowledge_gaps(agent_id, lookback_hours=lookback_hours))

    async def get_self_improvement_proposals(self) -> List[SelfImprovementProposal]:
        """Retrieve system-generated proposals to improve operation or prompt alignment. Async version."""
        return self.meta.list_proposals()

    def get_self_improvement_proposals_sync(self) -> List[SelfImprovementProposal]:
        """Sync wrapper for get_self_improvement_proposals()."""
        return _run_sync(self.get_self_improvement_proposals())


# Backward compatibility aliases - these are the old sync method names
# that now delegate to the async versions internally
class CognitiveMemoryClientLegacy(CognitiveMemoryClient):
    """
    Legacy compatibility wrapper that provides sync method names.

    DEPRECATED: Use CognitiveMemoryClient with async methods or explicit _sync suffixed methods.
    """

    # Map old sync names to new sync wrappers for backward compatibility
    # These override the async methods with sync versions
    def observe(self, agent_id: str, content: str, kind: str = "observation",
                importance: float = 0.5, tags: Optional[List[str]] = None, **meta) -> str:
        """Legacy sync observe - calls observe_sync."""
        return self.observe_sync(agent_id, content, kind, importance, tags, **meta)

    def get_working_context(self, agent_id: str, limit: int = 16) -> List[WorkingMemoryItem]:
        """Legacy sync get_working_context - calls get_working_context_sync."""
        return self.get_working_context_sync(agent_id, limit)

    def start_episode(self, agent_id: str, goal: str, context: Optional[str] = None) -> str:
        """Legacy sync start_episode - calls start_episode_sync."""
        return self.start_episode_sync(agent_id, goal, context)

    def append_event(self, episode_id: str, kind: str, content: str, **meta) -> None:
        """Legacy sync append_event - calls append_event_sync."""
        return self.append_event_sync(episode_id, kind, content, **meta)

    def end_episode(self, episode_id: str, outcome: str, reward: Optional[float] = None) -> None:
        """Legacy sync end_episode - calls end_episode_sync."""
        return self.end_episode_sync(episode_id, outcome, reward)

    def suggest_procedures(self, agent_id: str, query: str, top_k: int = 5) -> List[Procedure]:
        """Legacy sync suggest_procedures - calls suggest_procedures_sync."""
        return self.suggest_procedures_sync(agent_id, query, top_k)

    def record_procedure_outcome(self, proc_id: str, success: bool) -> None:
        """Legacy sync record_procedure_outcome - calls record_procedure_outcome_sync."""
        return self.record_procedure_outcome_sync(proc_id, success)

    def get_knowledge_gaps(self, agent_id: str, lookback_hours: int = 24) -> List[dict]:
        """Legacy sync get_knowledge_gaps - calls get_knowledge_gaps_sync."""
        return self.get_knowledge_gaps_sync(agent_id, lookback_hours)

    def get_self_improvement_proposals(self) -> List[SelfImprovementProposal]:
        """Legacy sync get_self_improvement_proposals - calls get_self_improvement_proposals_sync."""
        return self.get_self_improvement_proposals_sync()
