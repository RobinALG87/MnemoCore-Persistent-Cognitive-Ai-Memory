"""Helper script to insert RLMIntegrator into llm_integration.py"""
import os

rlm_class = '''

class RLMIntegrator:
    """
    Phase 4.5: RLM (Recursive Language Models) Integrator.

    Bridges HAIMLLMIntegrator with the RecursiveSynthesizer to provide
    LLM-powered recursive memory queries.

    Usage::

        integrator = RLMIntegrator(llm_integrator)
        result = await integrator.rlm_query(
            "What do we know about X and how does it relate to Y?"
        )
        print(result["synthesis"])

    Without an LLM configured, falls back to heuristic decomposition
    and score-based synthesis.
    """

    def __init__(self, llm_integrator, config=None):
        from mnemocore.core.recursive_synthesizer import RecursiveSynthesizer, SynthesizerConfig
        self.llm_integrator = llm_integrator
        self.haim = llm_integrator.haim
        llm_call = None
        if llm_integrator.llm_client is not None:
            llm_call = llm_integrator._call_llm
        synth_config = config or SynthesizerConfig()
        self.synthesizer = RecursiveSynthesizer(
            engine=self.haim,
            config=synth_config,
            llm_call=llm_call,
        )

    async def rlm_query(self, query, context_text=None, project_id=None):
        """
        Execute a Phase 4.5 recursive memory query.

        Args:
            query:        The user question (can be complex/multi-topic).
            context_text: Optional large external text (Ripple environment).
            project_id:   Optional project scope for isolation masking.

        Returns:
            Dict: query, sub_queries, results, synthesis,
                  max_depth_hit, elapsed_ms, ripple_snippets, stats
        """
        from mnemocore.core.ripple_context import RippleContext
        ripple_ctx = None
        if context_text and context_text.strip():
            ripple_ctx = RippleContext(text=context_text, source_label="api_context")
        result = await self.synthesizer.synthesize(
            query=query,
            ripple_context=ripple_ctx,
            project_id=project_id,
        )
        return {
            "query": result.query,
            "sub_queries": result.sub_queries,
            "results": result.results,
            "synthesis": result.synthesis,
            "max_depth_hit": result.max_depth_hit,
            "elapsed_ms": result.total_elapsed_ms,
            "ripple_snippets": result.ripple_snippets,
            "stats": result.stats,
        }

    @classmethod
    def from_config(cls, haim_engine, llm_config, synth_config=None):
        """Create an RLMIntegrator directly from an LLMConfig."""
        llm_integrator = HAIMLLMIntegrator.from_config(haim_engine, llm_config)
        return cls(llm_integrator=llm_integrator, config=synth_config)

'''

path = "src/llm_integration.py"
content = open(path, "r", encoding="utf-8").read()
marker = "\ndef create_demo():"
idx = content.find(marker)
if idx == -1:
    print("ERROR: marker not found")
else:
    new_content = content[:idx] + rlm_class + content[idx:]
    open(path, "w", encoding="utf-8").write(new_content)
    print(f"OK: RLMIntegrator inserted at position {idx}")
    print(f"New length: {len(new_content)}")
