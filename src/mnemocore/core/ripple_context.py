"""
RippleContext – Phase 4.5: External Memory Environment
=======================================================
Implements the "Ripple" concept from MIT's Recursive Language Models paper.

Instead of loading all memory content into an LLM's context window (causing
"Context Rot"), RippleContext holds arbitrarily large text as an external
environment. The AI can programmatically search and slice it, fetching only
the relevant portions.

This is the MnemoCore equivalent of the RLM "REPL environment" — our tiered
storage (Redis/Qdrant/FileSystem) is the Ripple, and this class provides the
tool interface to search it without loading everything.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class RippleChunk:
    """A single chunk of text from the Ripple environment."""

    index: int
    text: str
    start_char: int
    end_char: int
    # Simple TF index for keyword search
    term_freq: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.term_freq:
            self.term_freq = self._build_tf(self.text)

    @staticmethod
    def _build_tf(text: str) -> Dict[str, int]:
        """Build term frequency index for this chunk."""
        tokens = re.findall(r"\b[a-zA-ZåäöÅÄÖ]{2,}\b", text.lower())
        return dict(Counter(tokens))

    def score_query(self, query_terms: List[str]) -> float:
        """BM25-inspired relevance score for a list of query terms."""
        if not query_terms or not self.term_freq:
            return 0.0
        total_terms = sum(self.term_freq.values()) or 1
        score = 0.0
        for term in query_terms:
            tf = self.term_freq.get(term, 0)
            if tf > 0:
                # Normalized TF with saturation (BM25-style)
                k1 = 1.5
                norm_tf = (tf * (k1 + 1)) / (tf + k1 * (total_terms / 100))
                score += norm_tf
        return score


class RippleContext:
    """
    External memory environment for Phase 4.5 Recursive Synthesis.

    Holds large text corpora outside the LLM context window. The AI
    interacts with it via search() and slice() — never loading everything
    at once.

    Usage:
        ctx = RippleContext(large_text, chunk_size=500)
        snippets = ctx.search("quantum computing", top_k=3)
        raw = ctx.slice(0, 1000)
    """

    def __init__(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        source_label: str = "external",
    ):
        """
        Args:
            text: The large text to hold as external context.
            chunk_size: Characters per chunk (default 500).
            chunk_overlap: Overlap between adjacent chunks (default 50).
            source_label: Label for logging/tracing.
        """
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.source_label = source_label
        self.chunks: List[RippleChunk] = []

        self._build_index()
        logger.debug(
            f"RippleContext '{source_label}': {len(self.text)} chars, "
            f"{len(self.chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})"
        )

    def _build_index(self) -> None:
        """Chunk the text and build the search index."""
        text = self.text
        step = max(1, self.chunk_size - self.chunk_overlap)
        idx = 0
        pos = 0

        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            chunk_text = text[pos:end]
            self.chunks.append(
                RippleChunk(
                    index=idx,
                    text=chunk_text,
                    start_char=pos,
                    end_char=end,
                )
            )
            idx += 1
            pos += step

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search the external context for relevant snippets.

        Uses BM25-inspired keyword scoring. Returns the top_k most
        relevant text chunks.

        Args:
            query: The search query.
            top_k: Number of chunks to return.

        Returns:
            List of relevant text snippets (strings).
        """
        if not self.chunks:
            return []

        query_terms = re.findall(r"\b[a-zA-ZåäöÅÄÖ]{2,}\b", query.lower())
        if not query_terms:
            # Fallback: return first top_k chunks
            return [c.text for c in self.chunks[:top_k]]

        scored = [(chunk, chunk.score_query(query_terms)) for chunk in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = [chunk.text for chunk, score in scored[:top_k] if score > 0]
        if not results:
            # No keyword matches — return first chunks as fallback
            results = [c.text for c in self.chunks[:top_k]]

        logger.debug(
            f"RippleContext.search('{query[:40]}...'): "
            f"top score={scored[0][1]:.2f}, returned {len(results)} chunks"
        )
        return results

    def slice(self, start_char: int, end_char: int) -> str:
        """
        Extract a raw slice of the external context by character position.

        Args:
            start_char: Start character index (inclusive).
            end_char: End character index (exclusive).

        Returns:
            The text slice.
        """
        start_char = max(0, start_char)
        end_char = min(len(self.text), end_char)
        return self.text[start_char:end_char]

    def get_chunk_by_index(self, index: int) -> Optional[RippleChunk]:
        """Get a specific chunk by its index."""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about this context."""
        return {
            "source": self.source_label,
            "total_chars": len(self.text),
            "total_chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "approx_tokens": len(self.text) // 4,  # rough estimate
        }

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "RippleContext":
        """Load a RippleContext from a text file."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(text=text, source_label=path, **kwargs)

    @classmethod
    def from_memory_jsonl(cls, path: str, **kwargs) -> "RippleContext":
        """
        Load a RippleContext from MnemoCore's memory.jsonl (Cold tier).
        Concatenates all memory content fields into a searchable corpus.
        """
        import json

        lines = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        content = obj.get("content", "")
                        mem_id = obj.get("id", "?")
                        if content:
                            lines.append(f"[{mem_id}] {content}")
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            logger.warning(f"memory.jsonl not found at {path}, creating empty context")
            return cls(text="", source_label=path, **kwargs)

        text = "\n".join(lines)
        logger.info(f"RippleContext loaded {len(lines)} memories from {path}")
        return cls(text=text, source_label=path, **kwargs)

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return (
            f"RippleContext(source='{self.source_label}', "
            f"chars={len(self.text)}, chunks={len(self.chunks)})"
        )
