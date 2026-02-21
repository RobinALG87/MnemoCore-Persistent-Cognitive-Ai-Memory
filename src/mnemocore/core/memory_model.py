"""
Memory Models
=============
Data classes mapping the Cognitive Architecture Phase 5 entities:
Working Memory (WM), Episodic Memory (EM), Semantic Memory (SM),
Procedural Memory (PM), and Meta-Memory (MM).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional, List

from .binary_hdv import BinaryHDV


# --- Working Memory (WM) ---

@dataclass
class WorkingMemoryItem:
    id: str
    agent_id: str
    created_at: datetime
    ttl_seconds: int
    content: str
    kind: Literal["thought", "observation", "goal", "plan_step", "action", "meta"]
    importance: float
    tags: List[str]
    hdv: Optional[BinaryHDV] = None


@dataclass
class WorkingMemoryState:
    agent_id: str
    max_items: int
    items: List[WorkingMemoryItem] = field(default_factory=list)


# --- Episodic Memory (EM) ---

@dataclass
class EpisodeEvent:
    timestamp: datetime
    kind: Literal["observation", "action", "thought", "reward", "error", "system"]
    content: str
    metadata: dict[str, Any]
    hdv: Optional[BinaryHDV] = None


@dataclass
class Episode:
    id: str
    agent_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    goal: Optional[str]
    context: Optional[str]
    events: List[EpisodeEvent]
    outcome: Literal["success", "failure", "partial", "unknown", "in_progress"]
    reward: Optional[float]
    links_prev: List[str]
    links_next: List[str]
    ltp_strength: float
    reliability: float

    @property
    def is_active(self) -> bool:
        return self.ended_at is None


# --- Semantic Memory (SM) ---

@dataclass
class SemanticConcept:
    id: str
    label: str
    description: str
    tags: List[str]
    prototype_hdv: BinaryHDV
    support_episode_ids: List[str]
    reliability: float
    last_updated_at: datetime
    metadata: dict[str, Any]


# --- Procedural Memory (PM) ---

@dataclass
class ProcedureStep:
    order: int
    instruction: str
    code_snippet: Optional[str] = None
    tool_call: Optional[dict[str, Any]] = None


@dataclass
class Procedure:
    id: str
    name: str
    description: str
    created_by_agent: Optional[str]
    created_at: datetime
    updated_at: datetime
    steps: List[ProcedureStep]
    trigger_pattern: str
    success_count: int
    failure_count: int
    reliability: float
    tags: List[str]


# --- Meta-Memory (MM) ---

@dataclass
class SelfMetric:
    name: str
    value: float
    window: str  # e.g. "5m", "1h", "24h"
    updated_at: datetime


@dataclass
class SelfImprovementProposal:
    id: str
    created_at: datetime
    author: Literal["system", "agent", "human"]
    title: str
    description: str
    rationale: str
    expected_effect: str
    status: Literal["pending", "accepted", "rejected", "implemented"]
    metadata: dict[str, Any]

