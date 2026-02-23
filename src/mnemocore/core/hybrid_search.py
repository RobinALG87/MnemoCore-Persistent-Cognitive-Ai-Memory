"""
Hybrid Search Engine (Dense + Sparse)
=====================================
Combines dense (semantic) and sparse (lexical) search using
Reciprocal Rank Fusion (RRF) and weighted alpha blending.

Phase 4.6: Enhanced semantic retrieval with multi-modal search support.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import re
from loguru import logger

import numpy as np


@dataclass
class SearchResult:
    """A single search result with combined scores."""
    id: str
    score: float
    payload: Dict[str, Any]
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None

    def __post_init__(self):
        self.score = float(self.score)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    mode: str = "hybrid"  # "dense", "sparse", or "hybrid"
    hybrid_alpha: float = 0.7  # Weight for dense search (0-1)
    rrf_k: int = 60  # RRF constant (higher = more ranking focused)
    sparse_model: str = "naver/splade-cocondenser-ensemble-distil"
    enable_query_expansion: bool = True
    expansion_terms: int = 3
    min_dense_score: float = 0.0
    min_sparse_score: float = 0.0


class SparseEncoder:
    """
    BM25-based sparse encoder for lexical search.

    Uses a simplified BM25 implementation that can work without
    external dependencies for basic keyword matching.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: defaultdict = defaultdict(int)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.vocabulary: Set[str] = set()

    def index_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Index documents for BM25 scoring.

        Args:
            documents: List of (doc_id, text) tuples
        """
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.vocabulary = set()

        for doc_id, text in documents:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            self.doc_lengths.append(len(tokens))
            self.vocabulary.update(unique_tokens)

            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.total_docs = len(documents)
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 1.0

        logger.info(
            f"SparseEncoder indexed {self.total_docs} documents, "
            f"vocab size: {len(self.vocabulary)}, "
            f"avg doc length: {self.avg_doc_length:.1f}"
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization supporting alphanumeric sequences."""
        # Extract alphanumeric sequences
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def encode(self, text: str) -> Dict[str, float]:
        """
        Encode text to sparse vector (token -> weight dict).

        Uses simplified TF-IDF style encoding without requiring
        document corpus statistics for query encoding.
        """
        tokens = self._tokenize(text)
        weights = {}

        # Use TF (term frequency) for query
        tf_counts = defaultdict(int)
        for token in tokens:
            tf_counts[token] += 1

        max_tf = max(tf_counts.values()) if tf_counts else 1

        for token, count in tf_counts.items():
            # Normalized TF with sublinear scaling
            weights[token] = (1 + np.log(count)) / (1 + np.log(max_tf))

        return weights

    def compute_scores(
        self,
        query: str,
        document_texts: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Compute BM25 scores for documents given a query.

        Args:
            query: Search query string
            document_texts: Dict of doc_id -> text

        Returns:
            Dict of doc_id -> BM25 score
        """
        query_tokens = set(self._tokenize(query))
        scores = {}

        for doc_id, text in document_texts.items():
            doc_tokens = self._tokenize(text)
            doc_length = len(doc_tokens)
            token_counts = defaultdict(int)

            for token in doc_tokens:
                token_counts[token] += 1

            score = 0.0
            for token in query_tokens:
                if token not in token_counts:
                    continue

                # IDF component
                df = self.doc_freqs.get(token, 1)
                idf = np.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

                # TF component
                tf = token_counts[token]
                tf_normalized = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                )

                score += idf * tf_normalized

            scores[doc_id] = max(0.0, score)

        return scores


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

    RRF is robust to score scale differences and works well when
    combining results from different retrieval methods.

    Formula: score(d) = sum(rank_i(d) + k) ^ -1
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        ranked_lists: List[List[Tuple[str, float]]],
        limit: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked lists into a single ranked list.

        Args:
            ranked_lists: List of ranked result lists [(id, score), ...]
            limit: Maximum number of results to return

        Returns:
            Fused ranked list [(id, fused_score), ...]
        """
        fused_scores: Dict[str, float] = defaultdict(float)

        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list):
                fused_scores[doc_id] += 1.0 / (self.k + rank + 1)

        # Sort by fused score (descending)
        result = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        if limit:
            result = result[:limit]

        return result


class HybridSearchEngine:
    """
    Main hybrid search engine combining dense and sparse retrieval.

    Supports three modes:
    - "dense": Semantic search only (vector similarity)
    - "sparse": Lexical search only (BM25/keywords)
    - "hybrid": Combined using RRF or weighted alpha blending
    """

    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
    ):
        self.config = config or HybridSearchConfig()
        self.sparse_encoder = SparseEncoder()
        self.rrf = ReciprocalRankFusion(k=self.config.rrf_k)
        self._indexed = False

        # Document registry for sparse search
        self._documents: Dict[str, str] = {}

    def index_document(self, doc_id: str, text: str) -> None:
        """
        Index a document for sparse search.

        Args:
            doc_id: Document identifier
            text: Document text content
        """
        self._documents[doc_id] = text

    def index_batch(self, documents: List[Tuple[str, str]]) -> None:
        """
        Index a batch of documents.

        Args:
            documents: List of (doc_id, text) tuples
        """
        for doc_id, text in documents:
            self._documents[doc_id] = text

        # Rebuild sparse encoder
        self.sparse_encoder.index_documents(documents)
        self._indexed = True

    async def search(
        self,
        query: str,
        dense_results: List[Tuple[str, float]],
        dense_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse results.

        Args:
            query: Search query string
            dense_results: Dense search results [(doc_id, score), ...]
            dense_payloads: Optional payload data for each doc_id
            limit: Maximum results to return

        Returns:
            List of SearchResult with combined scores
        """
        mode = self.config.mode.lower()

        if mode == "dense":
            return self._dense_only(dense_results, dense_payloads, limit)
        elif mode == "sparse":
            return await self._sparse_only(query, limit, dense_payloads)
        elif mode == "hybrid":
            return await self._hybrid_search(
                query, dense_results, dense_payloads, limit
            )
        else:
            logger.warning(f"Unknown search mode: {mode}, falling back to dense")
            return self._dense_only(dense_results, dense_payloads, limit)

    def _dense_only(
        self,
        dense_results: List[Tuple[str, float]],
        payloads: Optional[Dict[str, Dict[str, Any]]],
        limit: int,
    ) -> List[SearchResult]:
        """Return only dense search results."""
        results = []
        for doc_id, score in dense_results[:limit]:
            if score < self.config.min_dense_score:
                continue
            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    payload=payloads.get(doc_id, {}) if payloads else {},
                    dense_score=score,
                )
            )
        return results

    async def _sparse_only(
        self,
        query: str,
        limit: int,
        payloads: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[SearchResult]:
        """Return only sparse (BM25) search results."""
        if not self._indexed:
            logger.warning("Sparse encoder not indexed, falling back to empty results")
            return []

        sparse_scores = self.sparse_encoder.compute_scores(query, self._documents)

        # Sort by score (descending)
        ranked = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:limit]:
            if score < self.config.min_sparse_score:
                continue
            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    payload=payloads.get(doc_id, {}) if payloads else {},
                    sparse_score=score,
                )
            )

        return results

    async def _hybrid_search(
        self,
        query: str,
        dense_results: List[Tuple[str, float]],
        payloads: Optional[Dict[str, Dict[str, Any]]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Combine dense and sparse search using alpha-weighted scoring.

        Formula: hybrid_score = alpha * dense_norm + (1 - alpha) * sparse_norm

        Both scores are normalized to [0, 1] range before combination.
        """
        if not self._indexed or not dense_results:
            # Fall back to whatever is available
            if dense_results:
                return self._dense_only(dense_results, payloads, limit)
            return await self._sparse_only(query, limit, payloads)

        # Get sparse scores
        sparse_scores = self.sparse_encoder.compute_scores(query, self._documents)

        # Normalize dense scores to [0, 1]
        dense_dict = dict(dense_results)
        if dense_dict:
            max_dense = max(dense_dict.values()) if dense_dict else 1.0
            min_dense = min(dense_dict.values()) if dense_dict else 0.0
            dense_range = max_dense - min_dense if max_dense > min_dense else 1.0
        else:
            dense_range = 1.0
            min_dense = 0.0

        # Normalize sparse scores to [0, 1]
        if sparse_scores:
            max_sparse = max(sparse_scores.values()) if sparse_scores else 1.0
            min_sparse = min(sparse_scores.values()) if sparse_scores else 0.0
            sparse_range = max_sparse - min_sparse if max_sparse > min_sparse else 1.0
        else:
            sparse_range = 1.0
            min_sparse = 0.0

        # Combine scores using alpha weighting
        combined_scores: Dict[str, Tuple[float, float, float]] = {}

        # Process dense results
        for doc_id, dense_score in dense_results:
            norm_dense = (dense_score - min_dense) / dense_range if dense_range > 0 else 0.0
            combined_scores[doc_id] = (norm_dense, 0.0, 0.0)

        # Process sparse results
        for doc_id, sparse_score in sparse_scores.items():
            norm_sparse = (sparse_score - min_sparse) / sparse_range if sparse_range > 0 else 0.0

            if doc_id in combined_scores:
                norm_dense, _, _ = combined_scores[doc_id]
            else:
                norm_dense = 0.0

            # Alpha blending
            hybrid_score = (
                self.config.hybrid_alpha * norm_dense +
                (1 - self.config.hybrid_alpha) * norm_sparse
            )
            combined_scores[doc_id] = (hybrid_score, norm_dense, norm_sparse)

        # Sort by hybrid score
        ranked = sorted(
            [(doc_id, scores[0], scores[1], scores[2])
             for doc_id, scores in combined_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Build results
        results = []
        for doc_id, hybrid_score, dense_norm, sparse_norm in ranked[:limit]:
            # Reconstruct original scores for reporting
            orig_dense = next(
                (s for did, s in dense_results if did == doc_id),
                None
            )
            orig_sparse = sparse_scores.get(doc_id, 0.0)

            results.append(
                SearchResult(
                    id=doc_id,
                    score=hybrid_score,
                    payload=payloads.get(doc_id, {}) if payloads else {},
                    dense_score=orig_dense,
                    sparse_score=orig_sparse,
                )
            )

        return results

    async def search_rrf(
        self,
        query: str,
        dense_results: List[Tuple[str, float]],
        payloads: Optional[Dict[str, Dict[str, Any]]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).

        RRF is more robust to score scale differences and often
        performs better than simple weighted averaging.

        Formula: score(d) = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        if not self._indexed or not dense_results:
            if dense_results:
                return self._dense_only(dense_results, payloads, limit)
            return await self._sparse_only(query, limit, payloads)

        # Get sparse scores
        sparse_scores = self.sparse_encoder.compute_scores(query, self._documents)

        # Create ranked lists (sorted by score descending)
        dense_ranked = sorted(dense_results, key=lambda x: x[1], reverse=True)
        sparse_ranked = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply RRF
        fused = self.rrf.fuse([dense_ranked, sparse_ranked], limit=limit)

        # Build results
        results = []
        for doc_id, fused_score in fused:
            orig_dense = next((s for did, s in dense_results if did == doc_id), None)
            orig_sparse = sparse_scores.get(doc_id, 0.0)

            results.append(
                SearchResult(
                    id=doc_id,
                    score=fused_score,
                    payload=payloads.get(doc_id, {}) if payloads else {},
                    dense_score=orig_dense,
                    sparse_score=orig_sparse,
                )
            )

        return results

    def expand_query(self, query: str, top_terms: int = 3) -> str:
        """
        Simple query expansion using vocabulary statistics.

        Args:
            query: Original query string
            top_terms: Number of expansion terms to add

        Returns:
            Expanded query string
        """
        if not self.config.enable_query_expansion or not self._indexed:
            return query

        query_tokens = set(self.sparse_encoder._tokenize(query))

        # Find related terms based on co-occurrence (simplified)
        expansion_terms = []
        for token in query_tokens:
            if token in self.sparse_encoder.vocabulary:
                expansion_terms.append(token)

        # For a simple implementation, just return the original query
        # A full implementation would use term co-occurrence or word embeddings
        return query

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            "mode": self.config.mode,
            "hybrid_alpha": self.config.hybrid_alpha,
            "indexed": self._indexed,
            "total_documents": len(self._documents),
            "vocabulary_size": len(self.sparse_encoder.vocabulary),
        }
