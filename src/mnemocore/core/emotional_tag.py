"""
Emotional Tagging Module (Phase 5.0 — Agent 3)
================================================
Adds valence/arousal emotional metadata to MemoryNode storage.

Based on affective computing research (Russell's circumplex model):
  - emotional_valence: float in [-1.0, 1.0]
      -1.0 = extremely negative (fear, grief)
       0.0 = neutral
      +1.0 = extremely positive (joy, excitement)

  - emotional_arousal: float in [0.0, 1.0]
       0.0 = calm / low energy
       1.0 = highly activated / intense

These signals are used by the SubconsciousAI dream cycle to prioritize
consolidation of high-valence, high-arousal memories (the most
biologically significant ones).

Public API:
    tag = EmotionalTag(valence=0.8, arousal=0.9)
    meta = tag.to_metadata_dict()
    tag_back = EmotionalTag.from_metadata(node.metadata)
    score = tag.salience()  # combined importance weight
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import MemoryNode


# ------------------------------------------------------------------ #
#  EmotionalTag                                                      #
# ------------------------------------------------------------------ #

@dataclass
class EmotionalTag:
    """
    Two-dimensional emotional metadata for a memory.

    valence  ∈ [-1.0, 1.0]  (-1 = very negative, +1 = very positive)
    arousal  ∈ [ 0.0, 1.0]  ( 0 = calm, 1 = highly activated)
    """

    valence: float = 0.0
    arousal: float = 0.0

    def __post_init__(self) -> None:
        self.valence = float(max(-1.0, min(1.0, self.valence)))
        self.arousal = float(max(0.0, min(1.0, self.arousal)))

    # ---- Salience ------------------------------------------------ #

    def salience(self) -> float:
        """
        Combined salience score for dream cycle prioritization.
        High |valence| AND high arousal = most memorable / worth consolidating.

        Returns a float in [0.0, 1.0].
        """
        return abs(self.valence) * self.arousal

    def is_emotionally_significant(self, threshold: float = 0.3) -> bool:
        """True if the salience is above the given threshold."""
        return self.salience() >= threshold

    # ---- Serialization ------------------------------------------- #

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "emotional_valence": self.valence,
            "emotional_arousal": self.arousal,
            "emotional_salience": round(self.salience(), 4),
        }

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "EmotionalTag":
        """Extract an EmotionalTag from a MemoryNode's metadata dict."""
        return cls(
            valence=float(metadata.get("emotional_valence", 0.0)),
            arousal=float(metadata.get("emotional_arousal", 0.0)),
        )

    @classmethod
    def from_node(cls, node: "MemoryNode") -> "EmotionalTag":
        """Extract emotional tag directly from a MemoryNode."""
        return cls.from_metadata(getattr(node, "metadata", {}))

    # ---- Helpers -------------------------------------------------- #

    @classmethod
    def neutral(cls) -> "EmotionalTag":
        return cls(valence=0.0, arousal=0.0)

    @classmethod
    def high_positive(cls) -> "EmotionalTag":
        """Factory for highly positive, highly aroused tags (e.g. breakthrough)."""
        return cls(valence=1.0, arousal=1.0)

    @classmethod
    def high_negative(cls) -> "EmotionalTag":
        """Factory for highly negative, highly aroused tags (e.g. critical failure)."""
        return cls(valence=-1.0, arousal=1.0)

    def __repr__(self) -> str:
        return f"EmotionalTag(valence={self.valence:+.2f}, arousal={self.arousal:.2f}, salience={self.salience():.2f})"


# ------------------------------------------------------------------ #
#  Node helpers                                                      #
# ------------------------------------------------------------------ #

def attach_emotional_tag(node: "MemoryNode", tag: EmotionalTag) -> None:
    """Write emotional metadata into node.metadata in place."""
    node.metadata.update(tag.to_metadata_dict())


def get_emotional_tag(node: "MemoryNode") -> EmotionalTag:
    """Read the emotional tag from a node's metadata (returns neutral if absent)."""
    return EmotionalTag.from_node(node)
