"""Deterministic lexical and BinaryHDV retrieval over AgentMemory records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from mnemocore.agent_memory import MemoryRecord

from .contracts import HybridRecallResult, RetrievalRequest

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_HDV_BITS = 256


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


def lexical_similarity(query: str, content: str) -> float:
    """Return query-token coverage, a deterministic lexical retrieval score."""
    query_tokens = frozenset(_tokens(query))
    if not query_tokens:
        return 0.0
    return len(query_tokens.intersection(_tokens(content))) / len(query_tokens)


class BinaryHDV:
    """A local deterministic binary hypervector encoder with no external model."""

    bits = _HDV_BITS

    @classmethod
    def encode(cls, text: str) -> int:
        """Bundle token hashes into one fixed-width binary hypervector."""
        votes = [0] * cls.bits
        for token in _tokens(text):
            value = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest(), "big")
            for bit in range(cls.bits):
                votes[bit] += 1 if value & (1 << bit) else -1
        vector = 0
        for bit, vote in enumerate(votes):
            if vote >= 0:
                vector |= 1 << bit
        return vector

    @classmethod
    def similarity(cls, left: str, right: str) -> float:
        """Return normalized Hamming similarity between two encoded texts."""
        distance = (cls.encode(left) ^ cls.encode(right)).bit_count()
        return 1.0 - distance / cls.bits


class DeterministicHybridRetriever:
    """Exact-scope lexical candidate selection and deterministic BinaryHDV reranking."""

    def retrieve(
        self,
        request: RetrievalRequest,
        records: Iterable[MemoryRecord],
    ) -> tuple[HybridRecallResult, ...]:
        results: list[HybridRecallResult] = []
        for memory in records:
            if memory.scope != request.scope:
                continue
            lexical_score = lexical_similarity(request.query, memory.content)
            if lexical_score == 0.0:
                continue
            hdv_score = BinaryHDV.similarity(request.query, memory.content)
            hybrid_score = (lexical_score + hdv_score) / 2.0
            results.append(
                HybridRecallResult(
                    memory=memory,
                    lexical_score=lexical_score,
                    hdv_score=hdv_score,
                    score=hybrid_score,
                )
            )
        results.sort(key=lambda result: (-result.score, -result.lexical_score, result.memory.id))
        return tuple(results[: request.limit])
