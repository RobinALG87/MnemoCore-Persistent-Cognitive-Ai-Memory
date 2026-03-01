"""
Review Schedule – Review Entry Data Structure
==============================================
A scheduled review for a single memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .sm2 import SM2State


@dataclass
class ReviewEntry:
    """A scheduled review for a single memory."""
    memory_id: str
    agent_id: str
    due_at: datetime
    current_retention: float
    stability: float
    sm2_state: Optional[SM2State] = None
    emotional_salience: float = 0.0
    action: str = "review"

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "due_at": self.due_at.isoformat(),
            "current_retention": round(self.current_retention, 4),
            "stability": round(self.stability, 4),
            "sm2_state": self.sm2_state.to_dict() if self.sm2_state else None,
            "emotional_salience": round(self.emotional_salience, 4),
            "action": self.action,
        }


__all__ = ["ReviewEntry"]
