"""
Learning Journal
================
Tracks what works and what doesn't. Meta-learning layer for HAIM.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger

JOURNAL_PATH = "./data/learning_journal.json"


@dataclass
class LearningEntry:
    """A single learning."""
    id: str
    lesson: str
    context: str
    outcome: str  # "success" | "failure" | "mixed"
    confidence: float  # 0.0 - 1.0
    applications: int = 0  # Times this learning was applied
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = field(default_factory=list)


class LearningJournal:
    """Meta-learning storage."""
    
    def __init__(self, path: str = JOURNAL_PATH):
        self.path = path
        self.entries: Dict[str, LearningEntry] = {}
        self.predictions: Dict[str, dict] = {}  # In-memory prediction buffer
        self._load()
    
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                    for eid, entry_data in data.items():
                        self.entries[eid] = LearningEntry(**entry_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse learning journal JSON from {self.path}: {e}")
                self.entries = {}
            except Exception as e:
                logger.warning(f"Failed to load learning journal from {self.path}: {e}")
                self.entries = {}
    
    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.entries.items()}, f, indent=2)
    
    def record(
        self,
        lesson: str,
        context: str,
        outcome: str = "success",
        confidence: float = 0.7,
        tags: List[str] = None,
        surprise: float = 0.0
    ) -> str:
        """Record a new learning."""
        entry_id = f"learn_{uuid.uuid4().hex[:12]}"
        # Boost confidence if high surprise (flashbulb learning)
        if surprise > 0.5:
            confidence = min(1.0, confidence * (1.0 + surprise))
            
        entry = LearningEntry(
            id=entry_id,
            lesson=lesson,
            context=context,
            outcome=outcome,
            confidence=confidence,
            tags=tags or []
        )
        if surprise > 0:
            entry.tags.append(f"surprise_{surprise:.2f}")
            if surprise > 0.7:
                entry.tags.append("flashbulb_memory")
            
        self.entries[entry_id] = entry
        self._save()
        return entry_id
        
    def register_prediction(self, context: str, expectation: str) -> str:
        """Start a prediction cycle (v1.6)."""
        pred_id = f"pred_{datetime.now(timezone.utc).timestamp()}"
        self.predictions[pred_id] = {
            "context": context,
            "expectation": expectation,
            "timestamp": datetime.now(timezone.utc)
        }
        return pred_id

    def evaluate_surprise(self, expectation: str, actual: str) -> float:
        """
        Calculate surprise metric (0.0 to 1.0).
        Simple heuristic: Length difference + keyword overlap.
        In v1.8 this should use semantic embedding distance.
        """
        if expectation == actual:
            return 0.0
            
        # Jaccard similarity of words
        exp_words = set(expectation.lower().split())
        act_words = set(actual.lower().split())
        
        if not exp_words or not act_words:
            return 1.0
            
        intersection = len(exp_words.intersection(act_words))
        union = len(exp_words.union(act_words))
        
        similarity = intersection / union
        surprise = 1.0 - similarity
        return surprise

    def resolve_prediction(self, pred_id: str, actual_result: str) -> Optional[str]:
        """
        Close the loop: Compare expectation vs reality, record learning if surprised.
        """
        pred = self.predictions.pop(pred_id, None)
        if not pred:
            return None
            
        surprise = self.evaluate_surprise(pred['expectation'], actual_result)
        
        # Auto-record if significant surprise
        if surprise > 0.3:
            return self.record(
                lesson=f"Expectation '{pred['expectation']}' differed from '{actual_result}'",
                context=pred['context'],
                outcome="mixed" if surprise < 0.8 else "failure",
                confidence=0.8,
                surprise=surprise,
                tags=["prediction_error", "auto_generated"]
            )
        return None

    
    def apply(self, entry_id: str):
        """Mark a learning as applied (reinforcement)."""
        if entry_id in self.entries:
            self.entries[entry_id].applications += 1
            self.entries[entry_id].confidence = min(1.0, self.entries[entry_id].confidence * 1.05)
            self._save()
    
    def contradict(self, entry_id: str):
        """Mark a learning as contradicted (weakening)."""
        if entry_id in self.entries:
            self.entries[entry_id].confidence *= 0.8
            self._save()
    
    def query(self, context: str, top_k: int = 5) -> List[LearningEntry]:
        """Find relevant learnings for a context."""
        # Simple keyword matching for now
        context_lower = context.lower()
        scored = []
        
        for entry in self.entries.values():
            score = 0
            for word in context_lower.split():
                if word in entry.context.lower() or word in entry.lesson.lower():
                    score += 1
                if word in entry.tags:
                    score += 2
            
            score *= entry.confidence
            if score > 0:
                scored.append((score, entry))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]
    
    def get_top_learnings(self, n: int = 10) -> List[LearningEntry]:
        """Get most confident/applied learnings."""
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.confidence * (1 + e.applications * 0.1),
            reverse=True
        )
        return sorted_entries[:n]
    
    def stats(self) -> Dict:
        return {
            "total_learnings": len(self.entries),
            "successes": sum(1 for e in self.entries.values() if e.outcome == "success"),
            "failures": sum(1 for e in self.entries.values() if e.outcome == "failure"),
            "avg_confidence": sum(e.confidence for e in self.entries.values()) / max(1, len(self.entries)),
            "total_applications": sum(e.applications for e in self.entries.values())
        }
