"""
Goal Tree
=========
Hierarchical goal decomposition with autonomous sub-goal generation.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

GOALS_PATH = "./data/goals.json"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """A goal with potential sub-goals."""
    id: str
    title: str
    description: str
    parent_id: Optional[str] = None
    status: str = "active"
    priority: float = 0.5  # 0.0 - 1.0
    progress: float = 0.0  # 0.0 - 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    deadline: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    
    def is_leaf(self, all_goals: Dict[str, 'Goal']) -> bool:
        """Check if this goal has no children."""
        return not any(g.parent_id == self.id for g in all_goals.values())


class GoalTree:
    """Hierarchical goal management."""
    
    def __init__(self, path: str = GOALS_PATH):
        self.path = path
        self.goals: Dict[str, Goal] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                for gid, goal_data in data.items():
                    self.goals[gid] = Goal(**goal_data)
    
    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.goals.items()}, f, indent=2)
    
    def add(
        self,
        title: str,
        description: str,
        parent_id: Optional[str] = None,
        priority: float = 0.5,
        deadline: Optional[str] = None,
        tags: List[str] = None
    ) -> str:
        """Add a new goal."""
        goal_id = f"goal_{len(self.goals)}"
        goal = Goal(
            id=goal_id,
            title=title,
            description=description,
            parent_id=parent_id,
            priority=priority,
            deadline=deadline,
            tags=tags or []
        )
        self.goals[goal_id] = goal
        self._save()
        return goal_id
    
    def decompose(self, goal_id: str, sub_goals: List[Dict]) -> List[str]:
        """Break a goal into sub-goals."""
        if goal_id not in self.goals:
            return []
        
        created = []
        for sg in sub_goals:
            sub_id = self.add(
                title=sg.get("title", "Untitled"),
                description=sg.get("description", ""),
                parent_id=goal_id,
                priority=sg.get("priority", 0.5),
                tags=sg.get("tags", [])
            )
            created.append(sub_id)
        
        return created
    
    def complete(self, goal_id: str):
        """Mark a goal as completed and update parent progress."""
        if goal_id not in self.goals:
            return
        
        self.goals[goal_id].status = GoalStatus.COMPLETED.value
        self.goals[goal_id].progress = 1.0
        
        # Update parent progress
        parent_id = self.goals[goal_id].parent_id
        if parent_id and parent_id in self.goals:
            self._update_parent_progress(parent_id)
        
        self._save()
    
    def _update_parent_progress(self, goal_id: str):
        """Recalculate parent progress based on children."""
        children = [g for g in self.goals.values() if g.parent_id == goal_id]
        if not children:
            return
        
        total_progress = sum(c.progress for c in children)
        self.goals[goal_id].progress = total_progress / len(children)
    
    def block(self, goal_id: str, reason: str):
        """Mark a goal as blocked."""
        if goal_id in self.goals:
            self.goals[goal_id].status = GoalStatus.BLOCKED.value
            self.goals[goal_id].blockers.append(reason)
            self._save()
    
    def get_active(self) -> List[Goal]:
        """Get all active goals."""
        return [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE.value]
    
    def get_next_actions(self, limit: int = 5) -> List[Goal]:
        """Get actionable leaf goals sorted by priority."""
        leaves = [
            g for g in self.goals.values()
            if g.status == GoalStatus.ACTIVE.value and g.is_leaf(self.goals)
        ]
        leaves.sort(key=lambda g: g.priority, reverse=True)
        return leaves[:limit]
    
    def get_tree(self, root_id: Optional[str] = None, depth: int = 0) -> List[Dict]:
        """Get goal tree as nested structure."""
        if root_id is None:
            roots = [g for g in self.goals.values() if g.parent_id is None]
        else:
            roots = [self.goals[root_id]] if root_id in self.goals else []
        
        result = []
        for goal in roots:
            children = [g for g in self.goals.values() if g.parent_id == goal.id]
            node = {
                "id": goal.id,
                "title": goal.title,
                "status": goal.status,
                "progress": goal.progress,
                "priority": goal.priority,
                "depth": depth,
                "children": self.get_tree(goal.id, depth + 1) if children else []
            }
            result.append(node)
        
        return result
    
    def stats(self) -> Dict:
        return {
            "total_goals": len(self.goals),
            "active": sum(1 for g in self.goals.values() if g.status == "active"),
            "completed": sum(1 for g in self.goals.values() if g.status == "completed"),
            "blocked": sum(1 for g in self.goals.values() if g.status == "blocked"),
            "avg_progress": sum(g.progress for g in self.goals.values()) / max(1, len(self.goals))
        }
