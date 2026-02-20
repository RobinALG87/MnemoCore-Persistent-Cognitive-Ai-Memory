"""
Subconscious AI Worker – Phase 4.4 (BETA)
==========================================
A small LLM (Phi 3.5, Llama 7B) that pulses in the background,
performing memory sorting, enhanced dreaming, and micro self-improvement.

This is an OPT-IN BETA feature that must be explicitly enabled in config.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    SUBCONSCIOUS AI WORKER                    │
    │                                                             │
    │    Pulse Loop (configurable interval)                       │
    │         │                                                   │
    │         ├──► Resource Guard (CPU, rate limits)              │
    │         │                                                   │
    │         ├──► Memory Sorting (categorize & tag)              │
    │         │                                                   │
    │         ├──► Enhanced Dreaming (LLM-assisted consolidation) │
    │         │                                                   │
    │         └──► Micro Self-Improvement (pattern analysis)      │
    │                                                             │
    │    Pluggable Model: Ollama | LM Studio | API                │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from .config import SubconsciousAIConfig
    from .engine import HAIMEngine
    from .node import MemoryNode


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SubconsciousCycleResult:
    """Result from a single subconscious AI cycle."""

    timestamp: str
    operation: str  # "sorting" | "dreaming" | "improvement"
    input_count: int
    output: Dict[str, Any]
    elapsed_ms: float
    model_used: str
    dry_run: bool
    error: Optional[str] = None


@dataclass
class Suggestion:
    """A suggestion from micro self-improvement."""

    suggestion_id: str
    category: str  # "config" | "metadata" | "consolidation" | "query"
    confidence: float
    rationale: str
    proposed_change: Dict[str, Any]
    applied: bool = False
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Model Clients
# ─────────────────────────────────────────────────────────────────────────────


class ModelClient:
    """Base class for LLM model clients."""

    def __init__(self, model_name: str, model_url: str, **kwargs):
        self.model_name = model_name
        self.model_url = model_url

    async def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7
    ) -> str:
        raise NotImplementedError


class OllamaClient(ModelClient):
    """Client for Ollama local models."""

    def __init__(self, model_name: str, model_url: str, timeout: int = 30):
        super().__init__(model_name, model_url)
        self.timeout = timeout
        self._generate_url = f"{model_url.rstrip('/')}/api/generate"

    async def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7
    ) -> str:
        """Generate text using Ollama API."""
        try:
            import aiohttp

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._generate_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "").strip()
                    else:
                        error_text = await resp.text()
                        logger.error(f"Ollama error {resp.status}: {error_text}")
                        return ""

        except asyncio.TimeoutError:
            logger.warning(f"Ollama request timed out after {self.timeout}s")
            return ""
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return ""


class LMStudioClient(ModelClient):
    """Client for LM Studio (OpenAI-compatible API)."""

    def __init__(self, model_name: str, model_url: str, timeout: int = 30):
        super().__init__(model_name, model_url)
        self.timeout = timeout
        self._chat_url = f"{model_url.rstrip('/')}/v1/chat/completions"

    async def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7
    ) -> str:
        """Generate text using LM Studio's OpenAI-compatible API."""
        try:
            import aiohttp

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._chat_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            return (
                                choices[0].get("message", {}).get("content", "").strip()
                            )
                        return ""
                    else:
                        error_text = await resp.text()
                        logger.error(f"LM Studio error {resp.status}: {error_text}")
                        return ""

        except asyncio.TimeoutError:
            logger.warning(f"LM Studio request timed out after {self.timeout}s")
            return ""
        except Exception as e:
            logger.error(f"LM Studio request failed: {e}")
            return ""


class APIClient(ModelClient):
    """Client for external API providers (OpenAI, Anthropic, etc.)."""

    def __init__(
        self,
        model_name: str,
        model_url: str,
        api_key: Optional[str] = None,
        provider: str = "openai",
        timeout: int = 30,
    ):
        super().__init__(model_name, model_url)
        self.api_key = api_key
        self.provider = provider
        self.timeout = timeout

    async def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7
    ) -> str:
        """Generate text using external API."""
        try:
            import aiohttp

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            if self.provider in ("openai", "openai_api"):
                endpoint = f"{self.model_url.rstrip('/')}/v1/chat/completions"
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            elif self.provider in ("anthropic", "anthropic_api"):
                endpoint = f"{self.model_url.rstrip('/')}/v1/messages"
                headers["x-api-key"] = self.api_key or ""
                headers["anthropic-version"] = "2023-06-01"
                payload = {
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
            else:
                logger.error(f"Unknown API provider: {self.provider}")
                return ""

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Handle both OpenAI and Anthropic response formats
                        if "choices" in data:
                            return data["choices"][0]["message"]["content"].strip()
                        elif "content" in data:
                            return data["content"][0]["text"].strip()
                        return ""
                    else:
                        error_text = await resp.text()
                        logger.error(f"API error {resp.status}: {error_text}")
                        return ""

        except asyncio.TimeoutError:
            logger.warning(f"API request timed out after {self.timeout}s")
            return ""
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Resource Guard
# ─────────────────────────────────────────────────────────────────────────────


class ResourceGuard:
    """Monitor and throttle resource usage."""

    def __init__(self, max_cpu_percent: float, rate_limit_per_hour: int):
        self.max_cpu_percent = max_cpu_percent
        self.rate_limit_per_hour = rate_limit_per_hour
        self._call_history: deque = deque(maxlen=1000)
        self._consecutive_errors = 0

    def check_cpu(self) -> bool:
        """Check if CPU usage is below threshold."""
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.1)
            if cpu > self.max_cpu_percent:
                logger.debug(f"CPU {cpu:.1f}% > threshold {self.max_cpu_percent}%")
                return False
            return True
        except ImportError:
            # psutil not available, allow
            return True

    def check_rate_limit(self) -> bool:
        """Check if we're under the hourly rate limit."""
        now = time.time()
        cutoff = now - 3600
        # Remove calls older than 1 hour using O(1) popleft()
        while self._call_history and self._call_history[0] < cutoff:
            self._call_history.popleft()
        if len(self._call_history) >= self.rate_limit_per_hour:
            logger.debug(
                f"Rate limit reached: {len(self._call_history)}/{self.rate_limit_per_hour}"
            )
            return False
        return True

    def record_call(self):
        """Record a call for rate limiting."""
        self._call_history.append(time.time())

    def record_error(self):
        """Record an error for backoff calculation."""
        self._consecutive_errors += 1

    def record_success(self):
        """Reset error counter on success."""
        self._consecutive_errors = 0

    def get_backoff_seconds(self, base_interval: int, max_backoff: int) -> int:
        """Calculate backoff interval based on consecutive errors."""
        if self._consecutive_errors <= 0:
            return base_interval
        # Exponential backoff
        backoff = min(base_interval * (2**self._consecutive_errors), max_backoff)
        return backoff

    @property
    def consecutive_errors(self) -> int:
        """Expose consecutive errors count for stats reporting."""
        return self._consecutive_errors


# ─────────────────────────────────────────────────────────────────────────────
# Main Worker
# ─────────────────────────────────────────────────────────────────────────────


class SubconsciousAIWorker:
    """
    Phase 4.4: Subconscious AI Worker (BETA)

    A small LLM that pulses in the background, performing:
    - Memory sorting and categorization
    - Enhanced dreaming (LLM-assisted consolidation)
    - Micro self-improvement (pattern analysis)

    This is an OPT-IN feature. Set `subconscious_ai.enabled: true` in config.
    """

    def __init__(self, engine: "HAIMEngine", config: "SubconsciousAIConfig"):
        self.engine = engine
        self.cfg = config

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._current_interval = config.pulse_interval_seconds

        # Components
        self._model_client = self._init_model_client()
        self._resource_guard = ResourceGuard(
            max_cpu_percent=config.max_cpu_percent,
            rate_limit_per_hour=config.rate_limit_per_hour,
        )

        # Audit trail
        self._audit_file: Optional[Path] = None
        if config.audit_trail_path:
            self._audit_file = Path(config.audit_trail_path)
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)

        # Stats
        self._total_cycles = 0
        self._successful_cycles = 0
        self._failed_cycles = 0
        self._suggestions_generated = 0
        self._suggestions_applied = 0

        logger.info(
            f"SubconsciousAIWorker created (provider={config.model_provider}, "
            f"model={config.model_name}, enabled={config.enabled})"
        )

    def _init_model_client(self) -> ModelClient:
        """Factory for model clients."""
        provider = self.cfg.model_provider.lower()

        if provider == "ollama":
            return OllamaClient(
                model_name=self.cfg.model_name,
                model_url=self.cfg.model_url,
                timeout=self.cfg.cycle_timeout_seconds,
            )
        elif provider == "lm_studio":
            return LMStudioClient(
                model_name=self.cfg.model_name,
                model_url=self.cfg.model_url or "http://localhost:1234",
                timeout=self.cfg.cycle_timeout_seconds,
            )
        elif provider in ("openai_api", "anthropic_api"):
            return APIClient(
                model_name=self.cfg.model_name,
                model_url=self.cfg.api_base_url or "https://api.openai.com",
                api_key=self.cfg.api_key,
                provider=provider,
                timeout=self.cfg.cycle_timeout_seconds,
            )
        else:
            logger.warning(f"Unknown provider '{provider}', defaulting to Ollama")
            return OllamaClient(
                model_name=self.cfg.model_name,
                model_url=self.cfg.model_url,
                timeout=self.cfg.cycle_timeout_seconds,
            )

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background pulse loop."""
        if not self.cfg.enabled:
            logger.info("SubconsciousAI is disabled, not starting")
            return

        if self._running:
            logger.warning("SubconsciousAI already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._pulse_loop())
        logger.info(
            f"[Phase 4.4 BETA] SubconsciousAI started "
            f"(interval={self.cfg.pulse_interval_seconds}s, dry_run={self.cfg.dry_run})"
        )

    async def stop(self) -> None:
        """Stop the background pulse loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[Phase 4.4] SubconsciousAI stopped")

    # ─────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────

    async def _pulse_loop(self) -> None:
        """Main pulse loop that runs in the background."""
        while self._running:
            try:
                await self._run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SubconsciousAI cycle error: {e}")
                self._failed_cycles += 1
                self._resource_guard.record_error()

            # Calculate sleep interval (with optional backoff)
            if self.cfg.pulse_backoff_enabled:
                sleep_interval = self._resource_guard.get_backoff_seconds(
                    self.cfg.pulse_interval_seconds,
                    self.cfg.pulse_backoff_max_seconds,
                )
            else:
                sleep_interval = self.cfg.pulse_interval_seconds

            await asyncio.sleep(sleep_interval)

    async def _run_cycle(self) -> None:
        """Run a single cycle of subconscious operations."""
        # Resource checks
        if not self._resource_guard.check_cpu():
            logger.debug("Skipping cycle: CPU threshold exceeded")
            return

        if not self._resource_guard.check_rate_limit():
            logger.debug("Skipping cycle: rate limit reached")
            return

        self._total_cycles += 1
        t_start = time.monotonic()

        # Determine which operation to run (round-robin)
        operations = []
        if self.cfg.memory_sorting_enabled:
            operations.append("sorting")
        if self.cfg.enhanced_dreaming_enabled:
            operations.append("dreaming")
        if self.cfg.micro_self_improvement_enabled:
            operations.append("improvement")

        if not operations:
            logger.debug("No operations enabled")
            return

        operation = operations[self._total_cycles % len(operations)]

        try:
            if operation == "sorting":
                result = await self._memory_sorting_cycle()
            elif operation == "dreaming":
                result = await self._enhanced_dreaming_cycle()
            else:
                result = await self._micro_improvement_cycle()

            self._successful_cycles += 1
            self._resource_guard.record_success()
            self._resource_guard.record_call()

            # Log to audit trail
            if self.cfg.log_all_decisions:
                await self._log_cycle_result(result)

            elapsed_ms = (time.monotonic() - t_start) * 1000
            logger.debug(
                f"[SubconsciousAI] Cycle {self._total_cycles} ({operation}) "
                f"completed in {elapsed_ms:.0f}ms"
            )

        except Exception as e:
            self._failed_cycles += 1
            self._resource_guard.record_error()
            logger.error(f"[SubconsciousAI] Cycle {self._total_cycles} failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # Operations
    # ─────────────────────────────────────────────────────────────

    async def _memory_sorting_cycle(self) -> SubconsciousCycleResult:
        """
        Sort and categorize recent memories using LLM.

        Analyzes untagged memories and suggests categories/tags.
        """
        t_start = time.monotonic()

        # Get recent memories without tags
        recent = await self.engine.tier_manager.get_hot_recent(
            self.cfg.max_memories_per_cycle
        )
        unsorted = [m for m in recent if not m.metadata.get("category")]

        if not unsorted:
            return SubconsciousCycleResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                operation="sorting",
                input_count=0,
                output={"message": "No unsorted memories"},
                elapsed_ms=(time.monotonic() - t_start) * 1000,
                model_used=self.cfg.model_name,
                dry_run=self.cfg.dry_run,
            )

        # Build prompt for categorization
        memories_text = "\n".join(
            [f"[{i + 1}] {m.content[:200]}" for i, m in enumerate(unsorted[:5])]
        )

        prompt = f"""Categorize these memories into 2-3 broad categories and suggest tags for each.

Memories:
{memories_text}

Return JSON format:
{{"categories": ["cat1", "cat2"], "memory_tags": {{"1": ["tag1"], "2": ["tag2"]}}}}"""

        response = await self._model_client.generate(prompt, max_tokens=512)
        output = {"raw_response": response}

        # Parse response
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                output["parsed"] = parsed

                # Apply tags if not dry run
                if not self.cfg.dry_run and "memory_tags" in parsed:
                    updated_nodes: List["MemoryNode"] = []
                    for idx_str, tags in parsed["memory_tags"].items():
                        idx = int(idx_str) - 1
                        if 0 <= idx < len(unsorted):
                            mem = unsorted[idx]
                            mem.metadata["tags"] = tags
                            mem.metadata["category"] = parsed.get(
                                "categories", ["unknown"]
                            )[0]
                            updated_nodes.append(mem)
                            output["applied"] = output.get("applied", 0) + 1

                    if updated_nodes:
                        persist_tasks = [
                            self.engine.persist_memory_snapshot(mem)
                            for mem in updated_nodes
                        ]
                        persist_results = await asyncio.gather(
                            *persist_tasks,
                            return_exceptions=True,
                        )
                        persisted = sum(
                            1 for r in persist_results if not isinstance(r, Exception)
                        )
                        output["persisted"] = persisted
                        for r in persist_results:
                            if isinstance(r, Exception):
                                logger.warning(
                                    f"Failed to persist sorting metadata update: {r}"
                                )
        except json.JSONDecodeError:
            output["parse_error"] = "Could not parse JSON response"

        return SubconsciousCycleResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation="sorting",
            input_count=len(unsorted),
            output=output,
            elapsed_ms=(time.monotonic() - t_start) * 1000,
            model_used=self.cfg.model_name,
            dry_run=self.cfg.dry_run,
        )

    async def _enhanced_dreaming_cycle(self) -> SubconsciousCycleResult:
        """
        LLM-assisted memory consolidation.

        Identifies weak or isolated memories and suggests semantic bridges.
        """
        t_start = time.monotonic()

        # Find memories with low LTP (weak connections)
        recent = await self.engine.tier_manager.get_hot_recent(20)
        weak_memories = [
            m
            for m in recent
            if m.ltp_strength < 0.5 and not m.metadata.get("dream_analyzed")
        ][: self.cfg.max_memories_per_cycle]

        if not weak_memories:
            return SubconsciousCycleResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                operation="dreaming",
                input_count=0,
                output={"message": "No weak memories to analyze"},
                elapsed_ms=(time.monotonic() - t_start) * 1000,
                model_used=self.cfg.model_name,
                dry_run=self.cfg.dry_run,
            )

        # Build prompt for semantic bridging
        memories_text = "\n".join(
            [
                f"[{i + 1}] {m.content[:150]} (LTP: {m.ltp_strength:.2f})"
                for i, m in enumerate(weak_memories[:5])
            ]
        )

        prompt = f"""Analyze these memories and suggest semantic connections or bridging concepts.

Memories:
{memories_text}

For each memory, suggest 2-3 keywords or concepts that could connect it to related memories.
Return JSON: {{"bridges": {{"1": ["concept1", "concept2"], "2": ["concept3"]}}}}"""

        response = await self._model_client.generate(prompt, max_tokens=512)
        output = {"raw_response": response}

        # Parse and potentially create associations
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                output["parsed"] = parsed

                # Create synaptic bridges based on suggested concepts
                if not self.cfg.dry_run and "bridges" in parsed:
                    bindings_created = 0
                    for idx_str, concepts in parsed["bridges"].items():
                        try:
                            idx = int(idx_str) - 1
                            if 0 <= idx < len(weak_memories):
                                weak_mem = weak_memories[idx]
                                for concept in concepts[
                                    :2
                                ]:  # Limit to 2 concepts per memory
                                    # Encode the bridging concept and search for related memories
                                    concept_vec = self.engine.binary_encoder.encode(
                                        concept
                                    )
                                    hits = await self.engine.tier_manager.search(
                                        concept_vec, top_k=3
                                    )
                            for hit_id, score in hits:
                                if hit_id != weak_mem.id and score > 0.2:
                                    await self.engine.bind_memories(
                                        weak_mem.id, hit_id, success=True
                                    )
                                    bindings_created += 1
                                    logger.info(
                                        f"[Subconscious Dreaming] Bridge created: "
                                        f"'{weak_mem.content[:30]}...' <-> '{concept}' <-> {hit_id[:8]}"
                                    )
                        except (ValueError, IndexError) as e:
                            logger.debug(
                                f"Skipping invalid bridge index {idx_str}: {e}"
                            )

                    if bindings_created == 0:
                        logger.warning(
                            f"[Subconscious Dreaming] Generated {len(parsed.get('bridges', {}))} bridges "
                            "but no valid connections were found in memory."
                        )
                    output["bindings_created"] = bindings_created

                # Mark as analyzed to avoid re-processing
                for mem in weak_memories:
                    mem.metadata["dream_analyzed"] = True

                if not self.cfg.dry_run:
                    persist_tasks = [
                        self.engine.persist_memory_snapshot(mem)
                        for mem in weak_memories
                    ]
                    persist_results = await asyncio.gather(
                        *persist_tasks,
                        return_exceptions=True,
                    )
                    persisted = sum(
                        1 for r in persist_results if not isinstance(r, Exception)
                    )
                    output["persisted"] = persisted
                    for r in persist_results:
                        if isinstance(r, Exception):
                            logger.warning(
                                f"Failed to persist dreaming metadata update: {r}"
                            )

                output["bridges_found"] = len(parsed.get("bridges", {}))
        except json.JSONDecodeError:
            output["parse_error"] = "Could not parse JSON response"

        return SubconsciousCycleResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation="dreaming",
            input_count=len(weak_memories),
            output=output,
            elapsed_ms=(time.monotonic() - t_start) * 1000,
            model_used=self.cfg.model_name,
            dry_run=self.cfg.dry_run,
        )

    async def _micro_improvement_cycle(self) -> SubconsciousCycleResult:
        """
        Analyze patterns and generate improvement suggestions.

        Identifies:
        - Recurring low-confidence queries (knowledge gaps)
        - Metadata improvement opportunities
        - Configuration optimization suggestions
        """
        t_start = time.monotonic()

        # Get engine stats for analysis
        stats = await self.engine.get_stats()
        gap_stats = stats.get("gap_detector", {})

        suggestions: List[Suggestion] = []

        # Check for knowledge gaps
        if gap_stats.get("total_gaps", 0) > 5:
            suggestions.append(
                Suggestion(
                    suggestion_id=f"gap_{int(time.time())}",
                    category="query",
                    confidence=0.7,
                    rationale=f"High knowledge gap count: {gap_stats['total_gaps']}",
                    proposed_change={
                        "action": "review_gaps",
                        "count": gap_stats["total_gaps"],
                    },
                )
            )

        # Check tier balance
        tiers = stats.get("tiers", {})
        hot_count = tiers.get("hot_count", 0)
        max_hot = self.engine.config.tiers_hot.max_memories

        if hot_count > max_hot * 0.9:
            suggestions.append(
                Suggestion(
                    suggestion_id=f"tier_{int(time.time())}",
                    category="config",
                    confidence=0.6,
                    rationale=f"HOT tier near capacity: {hot_count}/{max_hot}",
                    proposed_change={
                        "action": "consider_tier_expansion",
                        "utilization": hot_count / max_hot,
                    },
                )
            )

        self._suggestions_generated += len(suggestions)
        output = {
            "suggestions": [asdict(s) for s in suggestions],
            "stats_snapshot": {
                "gaps": gap_stats.get("total_gaps", 0),
                "hot_utilization": hot_count / max_hot if max_hot else 0,
            },
        }

        return SubconsciousCycleResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation="improvement",
            input_count=1,  # Stats object
            output=output,
            elapsed_ms=(time.monotonic() - t_start) * 1000,
            model_used=self.cfg.model_name,
            dry_run=self.cfg.dry_run,
        )

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────

    async def _log_cycle_result(self, result: SubconsciousCycleResult) -> None:
        """Log cycle result to audit trail."""
        if not self._audit_file:
            return

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_audit, asdict(result))
        except Exception as e:
            logger.warning(f"Failed to write audit trail: {e}")

    def _write_audit(self, record: Dict[str, Any]) -> None:
        """Write a record to the audit file (sync)."""
        if not self._audit_file:
            return
        with open(self._audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @property
    def stats(self) -> Dict[str, Any]:
        """Return worker statistics."""
        return {
            "enabled": self.cfg.enabled,
            "running": self._running,
            "beta_mode": self.cfg.beta_mode,
            "dry_run": self.cfg.dry_run,
            "model_provider": self.cfg.model_provider,
            "model_name": self.cfg.model_name,
            "pulse_interval_seconds": self.cfg.pulse_interval_seconds,
            "total_cycles": self._total_cycles,
            "successful_cycles": self._successful_cycles,
            "failed_cycles": self._failed_cycles,
            "consecutive_errors": self._resource_guard.consecutive_errors,
            "suggestions_generated": self._suggestions_generated,
            "suggestions_applied": self._suggestions_applied,
            "operations": {
                "memory_sorting": self.cfg.memory_sorting_enabled,
                "enhanced_dreaming": self.cfg.enhanced_dreaming_enabled,
                "micro_self_improvement": self.cfg.micro_self_improvement_enabled,
            },
        }
