"""
Cognitive Memory Client
=======================
The high-level facade for autonomous agents to interact with the MnemoCore AGI Memory Substrate.
Provides easy methods for observation, episodic sequence tracking, and working memory recall.
"""

from typing import List, Optional, Any, Tuple
import logging

from .core.engine import HAIMEngine
from .core.working_memory import WorkingMemoryService, WorkingMemoryItem
from .core.episodic_store import EpisodicStoreService
from .core.semantic_store import SemanticStoreService
from .core.procedural_store import ProceduralStoreService
from .core.meta_memory import MetaMemoryService, SelfImprovementProposal
from .core.memory_model import Procedure

logger = logging.getLogger(__name__)

class CognitiveMemoryClient:
    """
    Plug-and-play cognitive memory facade for agent frameworks (LangGraph, AutoGen, OpenClaw, etc.).
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

    def observe(self, agent_id: str, content: str, kind: str = "observation", importance: float = 0.5, tags: Optional[List[str]] = None, **meta) -> str:
        """
        Push a new observation or thought directly into the agent's short-term Working Memory.
        """
        import uuid
        from datetime import datetime
        item_id = f"wm_{uuid.uuid4().hex[:8]}"
        
        item = WorkingMemoryItem(
            id=item_id,
            agent_id=agent_id,
            created_at=datetime.utcnow(),
            ttl_seconds=3600, # 1 hour default
            content=content,
            kind=kind, # type: ignore
            importance=importance,
            tags=tags or [],
            hdv=None # Could encode via engine in future
        )
        self.wm.push_item(agent_id, item)
        logger.debug(f"Agent {agent_id} observed: {content[:30]}...")
        return item_id

    def get_working_context(self, agent_id: str, limit: int = 16) -> List[WorkingMemoryItem]:
        """
        Read the active, un-pruned context out of the agent's working memory buffer.
        """
        state = self.wm.get_state(agent_id)
        if not state:
            return []
        
        return state.items[-limit:]

    # --- Episodic ---

    def start_episode(self, agent_id: str, goal: str, context: Optional[str] = None) -> str:
        """Begin a new temporally-linked event sequence."""
        return self.episodic.start_episode(agent_id, goal=goal, context=context)

    def append_event(self, episode_id: str, kind: str, content: str, **meta) -> None:
        """Log an action or outcome to an ongoing episode."""
        self.episodic.append_event(episode_id, kind, content, meta)

    def end_episode(self, episode_id: str, outcome: str, reward: Optional[float] = None) -> None:
        """Seal an episode, logging its final success or failure state."""
        self.episodic.end_episode(episode_id, outcome, reward)

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
                node = await self.engine.tier_manager.get_memory(mem_id)  # Fix: tier_manager.get_memory is async
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

    # --- Procedural ---

    def suggest_procedures(self, agent_id: str, query: str, top_k: int = 5) -> List[Procedure]:
        """Fetch executable tool-patterns based on the agent's intent."""
        return self.procedural.find_applicable_procedures(query, agent_id=agent_id, top_k=top_k)

    def record_procedure_outcome(self, proc_id: str, success: bool) -> None:
        """Report on the utility of a chosen procedure."""
        self.procedural.record_procedure_outcome(proc_id, success)

    # --- Meta / Self-awareness ---

    def get_knowledge_gaps(self, agent_id: str, lookback_hours: int = 24) -> List[dict]:
        """Return currently open knowledge gaps identified by the Pulse loop."""
        # Stubbed: Would interact with gap_detector
        return []

    def get_self_improvement_proposals(self) -> List[SelfImprovementProposal]:
        """Retrieve system-generated proposals to improve operation or prompt alignment."""
        return self.meta.list_proposals()

