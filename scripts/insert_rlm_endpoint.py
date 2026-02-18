"""Helper script to add /rlm/query endpoint to main.py"""

rlm_endpoint = '''

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4.5: Recursive Synthesis Engine Endpoint
# ─────────────────────────────────────────────────────────────────────────────

class RLMQueryRequest(BaseModel):
    """Request model for Phase 4.5 recursive memory query."""
    query: str = Field(..., min_length=1, max_length=4096, description="The query to synthesize (can be complex/multi-topic)")
    context_text: Optional[str] = Field(None, max_length=500000, description="Optional large external text (Ripple environment)")
    project_id: Optional[str] = Field(None, max_length=128, description="Optional project scope for isolation masking")
    max_depth: Optional[int] = Field(None, ge=0, le=5, description="Max recursion depth (0-5, default 3)")
    max_sub_queries: Optional[int] = Field(None, ge=1, le=10, description="Max sub-queries to decompose into (1-10, default 5)")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Final results to return (default 10)")


class RLMQueryResponse(BaseModel):
    """Response model for Phase 4.5 recursive memory query."""
    ok: bool
    query: str
    sub_queries: List[str]
    results: List[Dict[str, Any]]
    synthesis: str
    max_depth_hit: int
    elapsed_ms: float
    ripple_snippets: List[str]
    stats: Dict[str, Any]


@app.post(
    "/rlm/query",
    response_model=RLMQueryResponse,
    dependencies=[Depends(get_api_key), Depends(QueryRateLimiter())],
    tags=["Phase 4.5"],
    summary="Recursive Synthesis Query",
    description=(
        "Phase 4.5: Recursive Language Model (RLM) query. "
        "Decomposes complex queries into sub-questions, searches MnemoCore in parallel, "
        "recursively analyzes low-confidence clusters, and synthesizes a final answer. "
        "Implements the MIT CSAIL RLM paradigm to eliminate Context Rot."
    ),
)
@track_async_latency(API_REQUEST_LATENCY, {"method": "POST", "endpoint": "/rlm/query"})
async def rlm_query(
    req: RLMQueryRequest,
    engine: HAIMEngine = Depends(get_engine),
):
    """
    Phase 4.5 Recursive Synthesis Engine.

    Instead of a single flat search, this endpoint:
    1. Decomposes your query into focused sub-questions
    2. Searches MnemoCore in PARALLEL for each sub-question
    3. Recursively drills into low-confidence clusters
    4. Synthesizes all results into a coherent answer

    Rate limit: 500/minute (shared with /query).
    """
    API_REQUEST_COUNT.labels(method="POST", endpoint="/rlm/query", status="200").inc()

    from src.core.recursive_synthesizer import RecursiveSynthesizer, SynthesizerConfig
    from src.core.ripple_context import RippleContext

    # Build config from request overrides
    synth_config = SynthesizerConfig(
        max_depth=req.max_depth if req.max_depth is not None else 3,
        max_sub_queries=req.max_sub_queries if req.max_sub_queries is not None else 5,
        final_top_k=req.top_k if req.top_k is not None else 10,
    )

    # Build RippleContext if external text provided
    ripple_ctx = None
    if req.context_text and req.context_text.strip():
        ripple_ctx = RippleContext(text=req.context_text, source_label="api_context")

    # Run recursive synthesis (no LLM wired at API level — use heuristic mode)
    # To enable LLM synthesis, configure via RLMIntegrator in your application code
    synthesizer = RecursiveSynthesizer(engine=engine, config=synth_config)
    result = await synthesizer.synthesize(
        query=req.query,
        ripple_context=ripple_ctx,
        project_id=req.project_id,
    )

    return {
        "ok": True,
        "query": result.query,
        "sub_queries": result.sub_queries,
        "results": result.results,
        "synthesis": result.synthesis,
        "max_depth_hit": result.max_depth_hit,
        "elapsed_ms": result.total_elapsed_ms,
        "ripple_snippets": result.ripple_snippets,
        "stats": result.stats,
    }

'''

path = "src/api/main.py"
content = open(path, "r", encoding="utf-8").read()

# Insert before the if __name__ == "__main__" block
marker = '\nif __name__ == "__main__":'
idx = content.find(marker)
if idx == -1:
    print("ERROR: marker not found")
else:
    new_content = content[:idx] + rlm_endpoint + content[idx:]
    open(path, "w", encoding="utf-8").write(new_content)
    print(f"OK: /rlm/query endpoint inserted at position {idx}")
    print(f"New length: {len(new_content)}")
