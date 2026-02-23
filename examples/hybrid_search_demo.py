"""
Hybrid Search Demo
==================
Demonstrates the Phase 4.6 hybrid search capabilities combining
dense (semantic) and sparse (lexical) retrieval.

Usage:
    python -m examples.hybrid_search_demo
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnemocore.core.hybrid_search import (
    HybridSearchEngine,
    HybridSearchConfig,
    SearchResult,
)
from mnemocore.core.config import load_config


def print_results(title: str, results: list[SearchResult]):
    """Pretty print search results."""
    print(f"\n{title}")
    print("-" * len(title))
    for i, r in enumerate(results, 1):
        dense_s = f"{r.dense_score:.3f}" if r.dense_score else "N/A"
        sparse_s = f"{r.sparse_score:.3f}" if r.sparse_score else "N/A"
        print(f"{i}. [{r.id}] score={r.score:.3f} (dense={dense_s}, sparse={sparse_s})")


async def demo_hybrid_search():
    """Demonstrate hybrid search functionality."""
    
    # Load config (includes new search section)
    config = load_config()
    print(f"Search mode: {config.search.mode}")
    print(f"Hybrid alpha: {config.search.hybrid_alpha}")
    
    # Create engine with custom config
    engine_config = HybridSearchConfig(
        mode="hybrid",
        hybrid_alpha=0.7,  # 70% dense, 30% sparse
        rrf_k=60,
    )
    engine = HybridSearchEngine(engine_config)
    
    # Index sample documents
    documents = [
        ("doc1", "Machine learning models learn patterns from data"),
        ("doc2", "Deep neural networks are used for computer vision"),
        ("doc3", "Natural language processing uses transformers"),
        ("doc4", "The quick brown fox jumps over lazy dogs"),
        ("doc5", "Vector databases enable semantic search"),
        ("doc6", "Embeddings convert text to numeric vectors"),
    ]
    
    engine.index_batch(documents)
    print(f"\nIndexed {len(documents)} documents")
    print(f"Vocabulary size: {len(engine.sparse_encoder.vocabulary)}")
    
    # Test 1: Pure semantic (dense) search
    # Simulating dense vector similarity scores
    dense_results = [
        ("doc1", 0.92),  # High semantic similarity
        ("doc5", 0.88),
        ("doc6", 0.85),
        ("doc3", 0.75),
        ("doc4", 0.05),  # Low semantic similarity
    ]
    
    dense_only = engine._dense_only(dense_results, {}, limit=5)
    print_results("Dense-only results (semantic similarity)", dense_only)
    
    # Test 2: Pure lexical (sparse) search
    sparse_only = await engine._sparse_only("machine learning vectors", limit=5)
    print_results("Sparse-only results (keyword matching)", sparse_only)
    
    # Test 3: Hybrid search with alpha blending
    hybrid_results = await engine._hybrid_search(
        "machine learning vectors",
        dense_results,
        {},
        limit=5
    )
    print_results("Hybrid results (alpha-blended)", hybrid_results)
    
    # Test 4: Hybrid search with RRF
    rrf_results = await engine.search_rrf(
        "machine learning vectors",
        dense_results,
        {},
        limit=5
    )
    print_results("Hybrid results (RRF fusion)", rrf_results)
    
    # Test 5: Different search modes
    print("\n" + "=" * 60)
    print("Testing different search modes")
    print("=" * 60)
    
    for mode in ["dense", "sparse", "hybrid"]:
        engine.config = HybridSearchConfig(mode=mode)
        results = await engine.search(
            "machine learning",
            dense_results,
            {},
            limit=3
        )
        print(f"\nMode '{mode}': {[r.id for r in results]}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_hybrid_search())
