"""
RecursiveSynthesizer – Phase 4.5: Recursive Synthesis Engine
=============================================================
MnemoCore's implementation of the Recursive Language Models (RLM) concept
from MIT CSAIL (arXiv:2512.24601, Zhang, Kraska, Khattab).

The core idea: instead of loading all memories into an LLM's context window
(causing "Context Rot"), we:

1. DECOMPOSE  – Break a complex query into focused sub-queries
2. SEARCH     – Run each sub-query against MnemoCore in PARALLEL
3. RECURSE    – If a cluster is too large/uncertain, spawn a sub-agent call
4. SYNTHESIZE – Merge all sub-results into a final ranked answer

This means AI agents using MnemoCore never need to load all memories into
their context — they just ask the RecursiveSynthesizer to find what's relevant.

Architecture:
    User Query
        │
        ▼
    _decompose()          ← LLM or heuristic
        │
        ├── sub-query 1 ──┐
        ├── sub-query 2 ──┤  asyncio.gather() (parallel)
        ├── sub-query 3 ──┤
        └── sub-query N ──┘
                          │
                    _parallel_sub_search()
                          │
                    (if cluster too large)
                    _recursive_cluster_analysis()  ← sub-agent call
                          │
                    _synthesize_results()
                          │
                    SynthesisResult
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    from .engine import HAIMEngine
    from .node import MemoryNode
    from .ripple_context import RippleContext


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SubQueryResult:
    """Result from a single sub-query search."""

    sub_query: str
    memories: List[Dict[str, Any]]  # [{id, content, score, metadata}]
    depth: int
    elapsed_ms: float
    confidence: float  # best score in this result set


@dataclass
class SynthesisResult:
    """
    Final result from the RecursiveSynthesizer.

    Attributes:
        query:          Original user query.
        sub_queries:    The decomposed sub-questions.
        results:        Ranked list of memory results (deduped, merged).
        synthesis:      LLM-generated synthesis text (if LLM available).
        max_depth_hit:  How deep the recursion went.
        total_elapsed_ms: Wall-clock time for the full synthesis.
        ripple_snippets: Relevant snippets from RippleContext (if provided).
        stats:          Internal stats for debugging.
    """

    query: str
    sub_queries: List[str]
    results: List[Dict[str, Any]]
    synthesis: str
    max_depth_hit: int
    total_elapsed_ms: float
    ripple_snippets: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesizerConfig:
    """
    Configuration for the RecursiveSynthesizer.

    Attributes:
        max_depth:          Maximum recursion depth. Prevents infinite loops.
                            Depth 0 = no recursion (single pass).
        max_sub_queries:    Maximum number of sub-queries to decompose into.
        sub_query_top_k:    Results per sub-query search.
        final_top_k:        Final results to return after synthesis.
        min_confidence:     If best score < this, trigger deeper recursion.
        parallel_limit:     Max concurrent sub-searches (asyncio semaphore).
        cluster_size_threshold: If a sub-result has > this many memories,
                            trigger recursive cluster analysis.
        enable_ripple:      Whether to search RippleContext if provided.
        ripple_top_k:       Snippets to fetch from RippleContext per sub-query.
    """

    max_depth: int = 3
    max_sub_queries: int = 5
    sub_query_top_k: int = 8
    final_top_k: int = 10
    min_confidence: float = 0.35
    parallel_limit: int = 5
    cluster_size_threshold: int = 20
    enable_ripple: bool = True
    ripple_top_k: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic decomposition (no LLM required)
# ─────────────────────────────────────────────────────────────────────────────

_CONJUNCTION_PATTERN = re.compile(
    r"\b(och|and|samt|eller|or|men|but|också|also|dessutom|furthermore|"
    r"relaterat till|related to|kopplat till|connected to)\b",
    re.IGNORECASE,
)

_QUESTION_WORDS = re.compile(
    r"\b(vad|vem|när|var|hur|varför|vilket|vilka|what|who|when|where|how|why|which)\b",
    re.IGNORECASE,
)


def _heuristic_decompose(query: str, max_sub: int = 5) -> List[str]:
    """
    Decompose a query into sub-queries using heuristics (no LLM needed).

    Strategy:
    1. Split on conjunctions (and/och/samt etc.)
    2. Split on question words mid-sentence
    3. If still one chunk, extract key noun phrases as sub-queries
    4. Always include the original query as a sub-query
    """
    # Split on conjunctions
    parts = _CONJUNCTION_PATTERN.split(query)
    # Filter out the conjunction words themselves and whitespace
    conjunction_words = {
        "och",
        "and",
        "samt",
        "eller",
        "or",
        "men",
        "but",
        "också",
        "also",
        "dessutom",
        "furthermore",
        "relaterat till",
        "related to",
        "kopplat till",
        "connected to",
    }
    parts = [p.strip() for p in parts if p.strip().lower() not in conjunction_words]
    parts = [p for p in parts if len(p) > 10]  # filter very short fragments

    # Also try splitting on question words in the middle of a sentence
    if len(parts) <= 1:
        # Try splitting on question words that appear after the first word
        words = query.split()
        split_indices = [
            i for i, w in enumerate(words) if i > 0 and _QUESTION_WORDS.match(w)
        ]
        if split_indices:
            sub_parts = []
            prev = 0
            for idx in split_indices:
                sub_parts.append(" ".join(words[prev:idx]))
                prev = idx
            sub_parts.append(" ".join(words[prev:]))
            parts = [p.strip() for p in sub_parts if len(p.strip()) > 10]

    # Deduplicate and limit
    seen = set()
    unique_parts = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique_parts.append(p)

    # Always include the original query
    if query.lower() not in seen:
        unique_parts.insert(0, query)

    return unique_parts[:max_sub]


# ─────────────────────────────────────────────────────────────────────────────
# Main RecursiveSynthesizer
# ─────────────────────────────────────────────────────────────────────────────


class RecursiveSynthesizer:
    """
    Phase 4.5: Recursive Synthesis Engine for MnemoCore.

    Implements the RLM (Recursive Language Models) paradigm:
    - Decomposes complex queries into focused sub-queries
    - Runs sub-searches in PARALLEL against MnemoCore's tiered storage
    - Recursively analyzes large clusters via sub-agent calls
    - Synthesizes all results into a final ranked answer

    The LLM is optional — without one, heuristic decomposition and
    score-based synthesis are used (still highly effective).

    Example:
        synthesizer = RecursiveSynthesizer(engine=haim_engine)
        result = await synthesizer.synthesize(
            "What do we know about the Lotto patterns and how do they
             relate to the renovation project timeline?"
        )
        print(result.synthesis)
        print(result.sub_queries)
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        config: Optional[SynthesizerConfig] = None,
        llm_call: Optional[Any] = None,  # callable(prompt: str) -> str
    ):
        """
        Args:
            engine:   The HAIMEngine instance (provides memory search).
            config:   SynthesizerConfig. Uses defaults if None.
            llm_call: Optional callable for LLM-powered decomposition and
                      synthesis. Signature: (prompt: str) -> str.
                      If None, heuristic mode is used.
        """
        self.engine = engine
        self.config = config or SynthesizerConfig()
        self.llm_call = llm_call
        self._sem = asyncio.Semaphore(self.config.parallel_limit)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    async def synthesize(
        self,
        query: str,
        ripple_context: Optional["RippleContext"] = None,
        project_id: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Main entry point. Recursively synthesizes an answer to a complex query.

        Args:
            query:          The user's question (can be complex/multi-topic).
            ripple_context: Optional external text corpus (RippleContext).
                            If provided, also searches this for relevant snippets.
            project_id:     Optional project scope for isolation masking.

        Returns:
            SynthesisResult with ranked memories, synthesis text, and trace info.
        """
        t_start = time.monotonic()
        logger.info(f"[Phase 4.5] RecursiveSynthesizer.synthesize: '{query[:80]}...'")

        # 1. Decompose query into sub-queries
        sub_queries = await self._decompose(query)
        logger.info(
            f"[Phase 4.5] Decomposed into {len(sub_queries)} sub-queries: {sub_queries}"
        )

        # 2. Parallel sub-search (with optional recursion)
        sub_results = await self._parallel_sub_search(
            sub_queries=sub_queries,
            depth=0,
            project_id=project_id,
        )

        # 3. Search RippleContext if provided
        ripple_snippets: List[str] = []
        if ripple_context and self.config.enable_ripple:
            ripple_snippets = await self._search_ripple(
                query=query,
                sub_queries=sub_queries,
                ripple_context=ripple_context,
            )

        # 4. Merge and deduplicate all results
        merged = self._merge_results(sub_results)

        # 5. Synthesize final answer
        synthesis_text = await self._synthesize_results(
            query=query,
            merged_results=merged,
            ripple_snippets=ripple_snippets,
        )

        elapsed_ms = (time.monotonic() - t_start) * 1000
        max_depth = max((r.depth for r in sub_results), default=0)

        logger.info(
            f"[Phase 4.5] Synthesis complete: {len(merged)} results, "
            f"depth={max_depth}, elapsed={elapsed_ms:.0f}ms"
        )

        return SynthesisResult(
            query=query,
            sub_queries=sub_queries,
            results=merged[: self.config.final_top_k],
            synthesis=synthesis_text,
            max_depth_hit=max_depth,
            total_elapsed_ms=elapsed_ms,
            ripple_snippets=ripple_snippets,
            stats={
                "sub_query_count": len(sub_queries),
                "raw_result_count": sum(len(r.memories) for r in sub_results),
                "merged_count": len(merged),
                "ripple_snippet_count": len(ripple_snippets),
                "llm_available": self.llm_call is not None,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Decompose
    # ─────────────────────────────────────────────────────────────────────

    async def _decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into focused sub-queries.

        Uses LLM if available, otherwise falls back to heuristic decomposition.
        """
        if self.llm_call is not None:
            try:
                return await self._llm_decompose(query)
            except Exception as e:
                logger.warning(
                    f"[Phase 4.5] LLM decomposition failed ({e}), using heuristic"
                )

        return _heuristic_decompose(query, max_sub=self.config.max_sub_queries)

    async def _llm_decompose(self, query: str) -> List[str]:
        """Use LLM to intelligently decompose the query."""
        prompt = self._build_decomposition_prompt(query)

        # Support both sync and async callables
        if asyncio.iscoroutinefunction(self.llm_call):
            response = await self.llm_call(prompt)
        else:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, self.llm_call, prompt)

        sub_queries = self._parse_sub_queries(response)

        # Fallback if LLM returned nothing useful
        if not sub_queries:
            return _heuristic_decompose(query, max_sub=self.config.max_sub_queries)

        # Always include original query
        if query not in sub_queries:
            sub_queries.insert(0, query)

        return sub_queries[: self.config.max_sub_queries]

    def _build_decomposition_prompt(self, query: str) -> str:
        return f"""You are a memory retrieval assistant. Break down the following complex query into {self.config.max_sub_queries - 1} focused sub-queries that together cover all aspects of the original question.

Original query: "{query}"

Return ONLY the sub-queries, one per line, no numbering, no explanation.
Each sub-query should be a complete, standalone search question.
Sub-queries:"""

    def _parse_sub_queries(self, response: str) -> List[str]:
        """Parse LLM response into a list of sub-queries."""
        lines = [
            line.strip().lstrip("•-*123456789. ")
            for line in response.strip().splitlines()
        ]
        return [line for line in lines if len(line) > 5]

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Parallel Sub-Search
    # ─────────────────────────────────────────────────────────────────────

    async def _parallel_sub_search(
        self,
        sub_queries: List[str],
        depth: int,
        project_id: Optional[str] = None,
    ) -> List[SubQueryResult]:
        """
        Run all sub-queries in PARALLEL against MnemoCore.

        Uses asyncio.gather() with a semaphore to limit concurrency.
        This is the key performance advantage: instead of sequential searches,
        all sub-queries fire simultaneously.
        """
        tasks = [self._single_sub_search(sq, depth, project_id) for sq in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[Phase 4.5] Sub-search failed: {r}")
            else:
                valid_results.append(r)

        return valid_results

    async def _single_sub_search(
        self,
        sub_query: str,
        depth: int,
        project_id: Optional[str] = None,
    ) -> SubQueryResult:
        """
        Execute a single sub-query search with optional recursion.

        Respects the semaphore for concurrency control.
        """
        async with self._sem:
            t_start = time.monotonic()

            # Search MnemoCore
            raw_results = await self.engine.query(
                sub_query,
                top_k=self.config.sub_query_top_k,
                project_id=project_id,
                associative_jump=True,
                track_gaps=False,  # Don't trigger gap detection inside RLM
            )

            # Fetch memory content
            memories = []
            for mem_id, score in raw_results:
                node = await self.engine.tier_manager.get_memory(mem_id)
                if node:
                    memories.append(
                        {
                            "id": mem_id,
                            "content": node.content,
                            "score": float(score),
                            "metadata": node.metadata or {},
                            "tier": getattr(node, "tier", "unknown"),
                        }
                    )

            confidence = max((m["score"] for m in memories), default=0.0)
            elapsed_ms = (time.monotonic() - t_start) * 1000

            result = SubQueryResult(
                sub_query=sub_query,
                memories=memories,
                depth=depth,
                elapsed_ms=elapsed_ms,
                confidence=confidence,
            )

            # Recursive cluster analysis if confidence is low and depth allows
            if (
                depth < self.config.max_depth
                and confidence < self.config.min_confidence
                and len(memories) >= 2
            ):
                logger.debug(
                    f"[Phase 4.5] Low confidence ({confidence:.2f}) at depth {depth}, "
                    f"triggering recursive analysis for: '{sub_query[:50]}'"
                )
                result = await self._recursive_cluster_analysis(
                    parent_result=result,
                    depth=depth + 1,
                    project_id=project_id,
                )

            return result

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Recursive Cluster Analysis (Sub-Agent Call)
    # ─────────────────────────────────────────────────────────────────────

    async def _recursive_cluster_analysis(
        self,
        parent_result: SubQueryResult,
        depth: int,
        project_id: Optional[str] = None,
    ) -> SubQueryResult:
        """
        When a cluster is too large or confidence is low, spawn deeper searches.

        This is the "sub-agent call" from the RLM paper — instead of one big
        search, we break the cluster into focused micro-queries and search again.

        The depth counter prevents infinite recursion.
        """
        if depth > self.config.max_depth:
            logger.debug(
                f"[Phase 4.5] Max depth {self.config.max_depth} reached, stopping recursion"
            )
            return parent_result

        # Extract key terms from the top memories to form micro-queries
        micro_queries = self._extract_micro_queries(
            parent_result.sub_query,
            parent_result.memories,
        )

        if not micro_queries:
            return parent_result

        logger.debug(
            f"[Phase 4.5] Depth {depth}: spawning {len(micro_queries)} micro-queries "
            f"from cluster of {len(parent_result.memories)} memories"
        )

        # Recursively search with micro-queries
        deeper_results = await self._parallel_sub_search(
            sub_queries=micro_queries,
            depth=depth,
            project_id=project_id,
        )

        # Merge deeper results back into parent
        all_memories = list(parent_result.memories)
        seen_ids = {m["id"] for m in all_memories}

        for dr in deeper_results:
            for mem in dr.memories:
                if mem["id"] not in seen_ids:
                    all_memories.append(mem)
                    seen_ids.add(mem["id"])

        # Re-sort by score
        all_memories.sort(key=lambda m: m["score"], reverse=True)
        new_confidence = max((m["score"] for m in all_memories), default=0.0)

        return SubQueryResult(
            sub_query=parent_result.sub_query,
            memories=all_memories,
            depth=depth,
            elapsed_ms=parent_result.elapsed_ms,
            confidence=new_confidence,
        )

    def _extract_micro_queries(
        self,
        original_query: str,
        memories: List[Dict[str, Any]],
        max_micro: int = 3,
    ) -> List[str]:
        """
        Extract focused micro-queries from the top memories' content.

        Takes the most informative terms from top memory snippets and
        forms targeted sub-queries.
        """
        if not memories:
            return []

        # Use top 3 memories
        top_mems = memories[:3]
        micro_queries = []

        for mem in top_mems:
            content = mem.get("content", "")
            if not content:
                continue
            # Take first 100 chars as a focused micro-query
            snippet = content[:100].strip()
            if snippet and snippet != original_query:
                micro_queries.append(snippet)

        return micro_queries[:max_micro]

    # ─────────────────────────────────────────────────────────────────────
    # Step 3b: RippleContext Search
    # ─────────────────────────────────────────────────────────────────────

    async def _search_ripple(
        self,
        query: str,
        sub_queries: List[str],
        ripple_context: "RippleContext",
    ) -> List[str]:
        """
        Search the external RippleContext for relevant snippets.

        Runs all sub-queries against the external corpus and deduplicates.
        """
        all_snippets: List[str] = []
        seen: set = set()

        # Search with original query
        for snippet in ripple_context.search(query, top_k=self.config.ripple_top_k):
            key = snippet[:50]
            if key not in seen:
                seen.add(key)
                all_snippets.append(snippet)

        # Search with each sub-query
        for sq in sub_queries[:3]:  # Limit to avoid too many searches
            for snippet in ripple_context.search(sq, top_k=2):
                key = snippet[:50]
                if key not in seen:
                    seen.add(key)
                    all_snippets.append(snippet)

        logger.debug(
            f"[Phase 4.5] RippleContext returned {len(all_snippets)} unique snippets"
        )
        return all_snippets

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Merge & Synthesize
    # ─────────────────────────────────────────────────────────────────────

    def _merge_results(self, sub_results: List[SubQueryResult]) -> List[Dict[str, Any]]:
        """
        Merge results from all sub-queries, deduplicate, and re-rank.

        Memories appearing in multiple sub-queries get a score boost
        (similar to how RLM merges sub-agent outputs).
        """
        score_map: Dict[str, float] = {}
        content_map: Dict[str, Dict[str, Any]] = {}
        hit_count: Dict[str, int] = {}

        for sub_result in sub_results:
            for mem in sub_result.memories:
                mem_id = mem["id"]
                score = mem["score"]

                if mem_id not in score_map:
                    score_map[mem_id] = score
                    content_map[mem_id] = mem
                    hit_count[mem_id] = 1
                else:
                    # Boost for appearing in multiple sub-queries
                    score_map[mem_id] = max(score_map[mem_id], score) * 1.1
                    hit_count[mem_id] += 1

        # Build final list with hit_count boost
        merged = []
        for mem_id, base_score in score_map.items():
            mem = dict(content_map[mem_id])
            hits = hit_count[mem_id]
            # Multi-hit boost: log scale so it doesn't dominate
            import math

            mem["score"] = base_score * (1.0 + 0.15 * math.log1p(hits - 1))
            mem["sub_query_hits"] = hits
            merged.append(mem)

        merged.sort(key=lambda m: m["score"], reverse=True)
        return merged

    async def _synthesize_results(
        self,
        query: str,
        merged_results: List[Dict[str, Any]],
        ripple_snippets: List[str],
    ) -> str:
        """
        Generate a synthesis text from the merged results.

        Uses LLM if available, otherwise generates a structured summary.
        """
        if not merged_results and not ripple_snippets:
            return "No relevant memories found for this query."

        if self.llm_call is not None:
            try:
                return await self._llm_synthesize(
                    query, merged_results, ripple_snippets
                )
            except Exception as e:
                logger.warning(
                    f"[Phase 4.5] LLM synthesis failed ({e}), using heuristic"
                )

        return self._heuristic_synthesis(query, merged_results, ripple_snippets)

    async def _llm_synthesize(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ripple_snippets: List[str],
    ) -> str:
        """Use LLM to synthesize a coherent answer from all sub-results."""
        prompt = self._build_synthesis_prompt(query, results, ripple_snippets)

        if asyncio.iscoroutinefunction(self.llm_call):
            return await self.llm_call(prompt)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.llm_call, prompt)

    def _build_synthesis_prompt(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ripple_snippets: List[str],
    ) -> str:
        prompt = f"""You are a memory synthesis assistant. Based on the retrieved memory fragments below, provide a coherent, comprehensive answer to the query.

Query: "{query}"

Retrieved Memory Fragments (ranked by relevance):
"""
        for i, mem in enumerate(results[:8], 1):
            prompt += f"\n[{i}] (score: {mem['score']:.3f}, hits: {mem.get('sub_query_hits', 1)})\n{mem['content'][:300]}\n"

        if ripple_snippets:
            prompt += "\n\nAdditional Context (from external corpus):\n"
            for i, snippet in enumerate(ripple_snippets[:3], 1):
                prompt += f"\n[Context {i}]\n{snippet[:300]}\n"

        prompt += """
\nSynthesis (combine all fragments into a coherent answer, note any gaps or contradictions):"""
        return prompt

    def _heuristic_synthesis(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ripple_snippets: List[str],
    ) -> str:
        """Generate a structured synthesis without LLM."""
        lines = [f"Synthesis for: '{query}'", "=" * 60]

        if results:
            lines.append(f"\nTop {min(5, len(results))} relevant memories:\n")
            for i, mem in enumerate(results[:5], 1):
                hits = mem.get("sub_query_hits", 1)
                hit_str = f" [matched {hits} sub-queries]" if hits > 1 else ""
                lines.append(
                    f"{i}. [score: {mem['score']:.3f}{hit_str}]\n"
                    f"   {mem['content'][:200]}"
                )

        if ripple_snippets:
            lines.append(f"\nExternal context ({len(ripple_snippets)} snippets found):")
            for snippet in ripple_snippets[:2]:
                lines.append(f"  • {snippet[:150]}")

        if not results and not ripple_snippets:
            lines.append("No relevant memories found.")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        """Return synthesizer configuration stats."""
        return {
            "max_depth": self.config.max_depth,
            "max_sub_queries": self.config.max_sub_queries,
            "parallel_limit": self.config.parallel_limit,
            "min_confidence": self.config.min_confidence,
            "llm_available": self.llm_call is not None,
        }
