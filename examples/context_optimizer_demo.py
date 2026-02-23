#!/usr/bin/env python
"""
Context Window Prioritizer Demo
================================
Demonstration of Phase 6: Context Window Prioritization features.

This script shows:
1. Token counting with tiktoken (or heuristic fallback)
2. Semantic chunk splitting for long memories
3. Memory ranking based on relevance, recency, importance, and token efficiency
4. Budget-aware memory selection
5. Context building for RAG applications
"""

from datetime import datetime, timezone
from mnemocore.cognitive.context_optimizer import (
    ContextWindowPrioritizer,
    TokenCounter,
    SemanticChunker,
    ContextBuilder,
    ChunkConfig,
    ScoringWeights,
    MODEL_LIMITS,
    create_prioritizer,
    rank_memories,
)


def demo_token_counter():
    """Demonstrate token counting capabilities."""
    print("\n" + "=" * 60)
    print("TOKEN COUNTER DEMO")
    print("=" * 60)

    # Test different providers
    for model in ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"]:
        counter = TokenCounter.for_model(model)
        text = "Hello, this is a test of the token counting system."
        tokens = counter.count_tokens(text)
        print(f"{model}: {tokens} tokens for '{text[:40]}...'")

    # Batch counting
    counter = TokenCounter.for_model("gpt-4o")
    texts = [
        "Short text.",
        "This is a medium length text that has more tokens.",
        "Here is a much longer text that should contain significantly more tokens than the previous examples provided.",
    ]
    batch_counts = counter.count_tokens_batch(texts)
    print(f"\nBatch counting: {batch_counts}")


def demo_chunking():
    """Demonstrate semantic chunk splitting."""
    print("\n" + "=" * 60)
    print("SEMANTIC CHUNKING DEMO")
    print("=" * 60)

    counter = TokenCounter.for_model("gpt-4o")
    config = ChunkConfig(
        max_chunk_tokens=100,
        min_chunk_tokens=20,
        chunk_overlap_tokens=20,
    )
    chunker = SemanticChunker(config, counter)

    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans or animals. Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

    Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".

    AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, generative or creative tools, automated decision-making, and competing at the highest level in strategic game systems.

    The field was founded on the assumption that human intelligence "can be so precisely described that a machine can be made to simulate it". This raised philosophical arguments about the nature of the mind and the ethics of creating artificial beings endowed with human-like intelligence. These issues have been explored by myth, fiction and philosophy since antiquity.
    """

    chunks = chunker.split(long_text)
    print(f"Original text: {len(long_text)} chars, ~{counter.count_tokens(long_text)} tokens")
    print(f"Split into {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks, 1):
        tokens = counter.count_tokens(chunk)
        print(f"Chunk {i} ({tokens} tokens):")
        print(f"  {chunk[:100]}...")


def demo_ranking():
    """Demonstrate memory ranking."""
    print("\n" + "=" * 60)
    print("MEMORY RANKING DEMO")
    print("=" * 60)

    # Create sample memories as dicts (works with MemoryNode, WorkingMemoryItem, etc.)
    memories = [
        {
            "id": "mem_001",
            "content": "Python is a high-level programming language known for its simplicity.",
            "created_at": "2024-01-15T10:00:00Z",
            "importance": 0.9,
            "access_count": 50,
            "ltp_strength": 0.85,
        },
        {
            "id": "mem_002",
            "content": "JavaScript is primarily used for web development and runs in browsers.",
            "created_at": "2024-02-01T14:30:00Z",
            "importance": 0.7,
            "access_count": 30,
            "ltp_strength": 0.70,
        },
        {
            "id": "mem_003",
            "content": "Rust is a systems programming language focused on safety and performance.",
            "created_at": "2024-02-15T09:00:00Z",
            "importance": 0.6,
            "access_count": 15,
            "ltp_strength": 0.55,
        },
        {
            "id": "mem_004",
            "content": "Go is designed for simplicity and efficiency in concurrent applications.",
            "created_at": "2024-03-01T11:00:00Z",
            "importance": 0.5,
            "access_count": 10,
            "ltp_strength": 0.50,
        },
    ]

    # Create prioritizer with custom weights
    prioritizer = create_prioritizer(
        model_name="gpt-4o",
        token_budget=200,
        weights=ScoringWeights(
            relevance=1.0,
            recency=0.3,
            importance=0.5,
            token_efficiency=0.2,
        ),
    )

    # Rank memories by query
    query = "programming language for web development"
    print(f"Query: '{query}'\n")

    ranked = prioritizer.rank(memories, query=query)
    print("Ranking results:")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. [{r.memory_id}] Score: {r.score:.3f}")
        print(f"       Content: {r.content[:60]}...")
        print(f"       Relevance: {r.relevance:.2f}, Recency: {r.recency_weight:.2f}, "
              f"Importance: {r.importance:.2f}, Tokens: {r.token_count}")


def demo_budget_selection():
    """Demonstrate budget-aware memory selection."""
    print("\n" + "=" * 60)
    print("BUDGET-AWARE SELECTION DEMO")
    print("=" * 60)

    # Create memories with varying lengths
    memories = [
        {
            "id": f"mem_{i:03d}",
            "content": f"This is memory number {i}. " + ("x" * i * 20),  # Varying lengths
            "created_at": datetime.now(timezone.utc).isoformat(),
            "importance": 0.5 + (i % 5) * 0.1,
            "access_count": i + 1,
            "ltp_strength": 0.5 + (i % 5) * 0.1,
        }
        for i in range(1, 11)
    ]

    prioritizer = create_prioritizer(model_name="gpt-4o", token_budget=100)

    # Select memories with tight budget
    result = prioritizer.select(memories, query="memory")

    print(f"Token budget: {prioritizer.token_budget}")
    print(f"Total memories: {len(memories)}")
    print(f"Selected: {len(result.ranked_memories)} memories")
    print(f"Total tokens used: {result.total_tokens}")
    print(f"Remaining budget: {result.remaining_budget}")
    print(f"\nSelected memories:")
    for i, r in enumerate(result.ranked_memories, 1):
        print(f"  {i}. [{r.memory_id}] Tokens: {r.token_count}, Score: {r.score:.3f}")


def demo_context_building():
    """Demonstrate context building for RAG."""
    print("\n" + "=" * 60)
    print("CONTEXT BUILDER DEMO")
    print("=" * 60)

    memories = [
        {
            "id": "doc_001",
            "content": "MnemoCore is a persistent cognitive memory infrastructure for AI agents.",
            "created_at": "2024-01-01T00:00:00Z",
            "importance": 0.9,
            "ltp_strength": 0.9,
        },
        {
            "id": "doc_002",
            "content": "It uses hyperdimensional vectors for efficient memory storage and retrieval.",
            "created_at": "2024-01-02T00:00:00Z",
            "importance": 0.8,
            "ltp_strength": 0.8,
        },
        {
            "id": "doc_003",
            "content": "The system supports hot/warm/cold storage tiers for optimal performance.",
            "created_at": "2024-01-03T00:00:00Z",
            "importance": 0.7,
            "ltp_strength": 0.7,
        },
    ]

    prioritizer = create_prioritizer(model_name="gpt-4o", token_budget=500)
    builder = ContextBuilder(prioritizer)

    query = "What is MnemoCore and how does it work?"
    context = builder.build_rag_context(memories, query=query, max_tokens=300)

    print("Generated RAG Context:")
    print("-" * 60)
    print(context)


def demo_model_limits():
    """Display context window limits for various models."""
    print("\n" + "=" * 60)
    print("MODEL CONTEXT LIMITS")
    print("=" * 60)

    models_to_show = [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "llama-3.1-405b",
    ]

    print(f"{'Model':<30} {'Max Tokens':>12} {'Input Budget':>12}")
    print("-" * 60)
    for model in models_to_show:
        if model in MODEL_LIMITS:
            limits = MODEL_LIMITS[model]
            print(f"{model:<30} {limits.max_tokens:>12,} {limits.input_budget:>12,}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("CONTEXT WINDOW PRIORITIZER - PHASE 6 DEMO")
    print("=" * 60)

    demo_model_limits()
    demo_token_counter()
    demo_chunking()
    demo_ranking()
    demo_budget_selection()
    demo_context_building()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nQuick usage example:")
    print("""
    from mnemocore.cognitive.context_optimizer import rank_memories

    memories = [{"content": "...", "id": "...", "importance": 0.8}, ...]
    ranked = rank_memories(memories, query="my query", token_budget=4000)
    """)


if __name__ == "__main__":
    main()
