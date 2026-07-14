"""Deterministic lexical and BinaryHDV retrieval over AgentMemory records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from mnemocore.agent_memory import MemoryRecord

from .contracts import HybridRecallResult, RetrievalRequest

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_HDV_BITS = 256
_HDV_MINIMUM_SIMILARITY = 0.60


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


def _hdv_features(text: str) -> tuple[str, ...]:
    """Return stable token and character features for local HDV encoding.

    Character trigrams let the binary strategy recognize close word forms
    (for example ``apple`` and ``apples``) even when lexical token matching
    has no candidate.  Feature prefixes keep token and character namespaces
    distinct.
    """
    features: list[str] = []
    for token in _tokens(text):
        features.append(f"token:{token}")
        padded = f"^{token}$"
        features.extend(
            f"gram:{padded[index : index + 3]}"
            for index in range(max(1, len(padded) - 2))
        )
    return tuple(features)


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
        for feature in _hdv_features(text):
            value = int.from_bytes(
                hashlib.sha256(feature.encode("utf-8")).digest(), "big"
            )
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
    """Exact-scope union of HDV and lexical candidates with hybrid ranking."""

    def retrieve(
        self,
        request: RetrievalRequest,
        records: Iterable[MemoryRecord],
    ) -> tuple[HybridRecallResult, ...]:
        candidates: dict[str, HybridRecallResult] = {}
        for memory in records:
            if memory.scope != request.scope:
                continue
            lexical_score = lexical_similarity(request.query, memory.content)
            hdv_score = BinaryHDV.similarity(request.query, memory.content)
            hybrid_score = (lexical_score + hdv_score) / 2.0
            result = HybridRecallResult(
                memory=memory,
                lexical_score=lexical_score,
                hdv_score=hdv_score,
                score=hybrid_score,
            )
            if hdv_score >= _HDV_MINIMUM_SIMILARITY:
                candidates[memory.id] = result
            if lexical_score > 0.0:
                candidates[memory.id] = result

        results = list(candidates.values())
        results.sort(
            key=lambda result: (
                -result.score,
                -result.hdv_score,
                -result.lexical_score,
                result.memory.id,
            )
        )
        return tuple(results[: request.limit])
