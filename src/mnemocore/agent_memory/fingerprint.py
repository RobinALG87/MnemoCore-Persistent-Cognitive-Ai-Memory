"""Small deterministic text fingerprint used by AgentMemory reranking."""

from __future__ import annotations

import hashlib

_FINGERPRINT_BYTES = 32


def fingerprint_similarity(left: str, right: str) -> float:
    """Return normalized bit similarity for deterministic local fingerprints."""
    left_digest = hashlib.blake2b(
        left.encode("utf-8"), digest_size=_FINGERPRINT_BYTES
    ).digest()
    right_digest = hashlib.blake2b(
        right.encode("utf-8"), digest_size=_FINGERPRINT_BYTES
    ).digest()
    distance = sum((a ^ b).bit_count() for a, b in zip(left_digest, right_digest))
    return 1.0 - distance / (_FINGERPRINT_BYTES * 8)
