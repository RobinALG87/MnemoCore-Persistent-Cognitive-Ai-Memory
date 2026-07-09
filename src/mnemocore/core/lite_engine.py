"""LiteEngine: synchronous lightweight core using HDV for store/query.

Preserves core functionality (HDV encode + Hamming similarity) without heavy deps/workers.
"""
import uuid
from typing import Dict, List, Tuple, Optional, Any

from .binary_hdv import TextEncoder, BinaryHDV
from .node import MemoryNode


class LiteEngine:
    """Synchronous in-memory engine for lite profile.

    Uses only BinaryHDV/TextEncoder + MemoryNode dict.
    No asyncio, no TierManager, no initialize, no workers, no disk.
    """

    def __init__(self, dimension: int = 16384):
        self.dimension = dimension
        self.encoder = TextEncoder(dimension)
        self._memories: Dict[str, MemoryNode] = {}

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Store content as HDV-encoded MemoryNode. Returns id.
        Accepts metadata=dict or flat kwargs (user_id=..., etc) which are folded in.
        """
        if metadata is None:
            metadata = {}
        # Fold any flat meta kwargs into metadata (e.g. direct user_id=)
        for k, v in kwargs.items():
            if k not in ("metadata",):
                metadata[k] = v
        hdv = self.encoder.encode(content)
        mid = uuid.uuid4().hex[:12]
        node = MemoryNode(
            id=mid,
            hdv=hdv,
            content=content,
            metadata=metadata
        )
        self._memories[mid] = node
        return mid

    def query(self, text: str, top_k: int = 5, **_ignored) -> List[Tuple[str, float]]:
        """Rank by Hamming similarity (higher better). Returns [(id, score), ...]"""
        if not self._memories or not text:
            return []
        q_hdv = self.encoder.encode(text)
        scored = []
        for mid, node in self._memories.items():
            sim = q_hdv.similarity(node.hdv)  # 1.0 identical, ~0.5 random
            scored.append((mid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
