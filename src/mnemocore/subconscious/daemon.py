"""
MnemoCore Subconscious Daemon
=========================
Continuous background processing using Gemma 1B via Ollama.
Performs: concept extraction, parallel drawing, memory valuation, thought sorting.
Integrates with Redis Subconscious Bus to publish insights.
"""

import asyncio
import aiohttp
from mnemocore.utils import json_compat as json
import random
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import os
from loguru import logger

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.async_storage import AsyncRedisStorage
from mnemocore.core.config import get_config
from mnemocore.meta.learning_journal import LearningJournal
from mnemocore.core.node import MemoryNode
from mnemocore.core.metrics import (
    DREAM_LOOP_TOTAL,
    DREAM_LOOP_ITERATION_SECONDS,
    DREAM_LOOP_INSIGHTS_GENERATED,
    DREAM_LOOP_ACTIVE
)

# Default Config (overridden by config.yaml or environment variables)
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:1b"
DEFAULT_CYCLE_INTERVAL = 60  # seconds between thought cycles
DEFAULT_LOG_PATH = "./data/subconscious.log"
DEFAULT_DATA_PATH = "./data/memory.jsonl"
DEFAULT_EVOLUTION_STATE_PATH = "./data/subconscious_evolution.json"


def _get_path_from_config_or_env(env_var: str, config_attr: str, default: str) -> str:
    """
    Get a path from environment variable, config, or fallback to default.

    Priority: ENV > config.yaml > default
    """
    # Check environment variable first
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value

    # Try config
    try:
        config = get_config()
        if hasattr(config, 'paths') and hasattr(config.paths, config_attr):
            return getattr(config.paths, config_attr)
    except Exception as e:
        logger.warning(f"Failed to get config path for {config_attr}: {e}")

    return default


def _get_log_path() -> str:
    """Get log path from config or environment variable."""
    return _get_path_from_config_or_env(
        "HAIM_SUBCONSCIOUS_LOG_PATH",
        "subconscious_log",
        DEFAULT_LOG_PATH
    )


def _get_data_path() -> str:
    """Get data path from config or environment variable."""
    return _get_path_from_config_or_env(
        "HAIM_DATA_PATH",
        "memory_data",
        DEFAULT_DATA_PATH
    )


def _get_evolution_state_path() -> str:
    """Get evolution state path from config or environment variable."""
    return _get_path_from_config_or_env(
        "HAIM_EVOLUTION_STATE_PATH",
        "evolution_state",
        DEFAULT_EVOLUTION_STATE_PATH
    )


# Module-level paths (computed once at import)
LOG_PATH = _get_log_path()
HAIM_DATA_PATH = _get_data_path()
EVOLUTION_STATE_PATH = _get_evolution_state_path()


def _write_state_to_disk(state: Dict[str, Any], filepath: str):
    """Write state to disk synchronously (to be used in executor)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)


class SubconsciousDaemon:
    """The always-running background mind."""

    def __init__(self, storage: Optional[AsyncRedisStorage] = None, config: Optional[Any] = None):
        """
        Initialize SubconsciousDaemon with optional dependency injection.

        Args:
            storage: AsyncRedisStorage instance. If None, creates one in run().
            config: Configuration object. If None, loads from get_config().
        """
        # Load configuration
        self._config = config or get_config()

        # Dream loop configuration from config.yaml
        dream_loop_config = getattr(self._config, 'dream_loop', None)
        if dream_loop_config:
            self.ollama_url = getattr(dream_loop_config, 'ollama_url', DEFAULT_OLLAMA_URL)
            self.model = getattr(dream_loop_config, 'model', DEFAULT_MODEL)
            self.frequency_seconds = getattr(dream_loop_config, 'frequency_seconds', DEFAULT_CYCLE_INTERVAL)
            self.batch_size = getattr(dream_loop_config, 'batch_size', 10)
            self.max_iterations = getattr(dream_loop_config, 'max_iterations', 0)
            self.dream_loop_enabled = getattr(dream_loop_config, 'enabled', True)
        else:
            self.ollama_url = DEFAULT_OLLAMA_URL
            self.model = DEFAULT_MODEL
            self.frequency_seconds = DEFAULT_CYCLE_INTERVAL
            self.batch_size = 10
            self.max_iterations = 0
            self.dream_loop_enabled = True

        self.engine = HAIMEngine(persist_path=HAIM_DATA_PATH)
        self.journal = LearningJournal()

        # Graceful shutdown support using asyncio.Event
        self._stop_event = asyncio.Event()
        self.running = False

        self.cycle_count = 0
        self.insights_generated = 0
        self.current_cycle_interval = self.frequency_seconds
        self.schedule = {
            "concept_every": 5,
            "parallel_every": 3,
            "value_every": 10,
            "meta_every": 7,
            "cleanup_every": 20
        }
        self.activity_window: List[int] = []
        self.low_activity_streak = 0
        self.last_cycle_metrics: Dict[str, Any] = {}
        self._load_evolution_state()

        # Async Redis Storage (injected or initialized in run)
        self.storage: Optional[AsyncRedisStorage] = storage

        # Reusable aiohttp session (created in start/run, closed in stop)
        self._http_session: Optional[aiohttp.ClientSession] = None

    def _should_stop(self) -> bool:
        """Check if the daemon should stop (non-blocking check)."""
        return self._stop_event.is_set()

    async def request_stop(self):
        """Request graceful stop of the daemon (async-safe)."""
        self._stop_event.set()
        self.running = False
        
    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")

    def _load_evolution_state(self):
        """Load persistent evolution state from disk."""
        if not os.path.exists(EVOLUTION_STATE_PATH):
            return
        try:
            with open(EVOLUTION_STATE_PATH, "r") as f:
                state = json.load(f)
            self.cycle_count = int(state.get("cycle_count", self.cycle_count))
            self.insights_generated = int(state.get("insights_generated", self.insights_generated))
            self.current_cycle_interval = int(state.get("current_cycle_interval", self.current_cycle_interval))
            saved_schedule = state.get("schedule", {})
            if isinstance(saved_schedule, dict):
                for k in self.schedule:
                    if k in saved_schedule:
                        self.schedule[k] = max(2, int(saved_schedule[k]))
            self.activity_window = list(state.get("activity_window", []))[-12:]
            self.low_activity_streak = int(state.get("low_activity_streak", 0))
        except Exception as e:
            self.log(f"Failed to load evolution state: {e}")

    async def _save_evolution_state(self):
        """Persist state so evolution continues across restarts."""
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cycle_count": self.cycle_count,
            "insights_generated": self.insights_generated,
            "current_cycle_interval": self.current_cycle_interval,
            "schedule": self.schedule,
            "activity_window": self.activity_window[-12:],
            "low_activity_streak": self.low_activity_streak,
            "last_cycle_metrics": self.last_cycle_metrics,
        }
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_state_to_disk, state, EVOLUTION_STATE_PATH)
        except Exception as e:
            self.log(f"Failed to save evolution state: {e}")

    def _compute_surprise(self, metrics: Dict[str, Any]) -> float:
        """Estimate surprise from novelty/output dynamics."""
        score = 0.0
        score += 0.12 * metrics.get("concepts", 0)
        score += 0.20 * metrics.get("parallels", 0)
        score += 0.30 * metrics.get("meta_insights", 0)
        if metrics.get("adaptation") and metrics.get("adaptation") != "none":
            score += 0.25
        return min(1.0, score)

    def _adapt_evolution_policy(self, metrics: Dict[str, Any]):
        """
        Adapt cadence and schedule so the subconscious keeps evolving.
        Low activity -> stimulate exploration.
        High sustained activity -> stabilize to preserve quality.
        """
        activity_score = (
            metrics.get("concepts", 0)
            + metrics.get("parallels", 0)
            + metrics.get("meta_insights", 0)
        )
        self.activity_window.append(activity_score)
        self.activity_window = self.activity_window[-12:]

        if activity_score == 0:
            self.low_activity_streak += 1
        else:
            self.low_activity_streak = 0

        adaptation = "none"
        avg_activity = sum(self.activity_window) / max(1, len(self.activity_window))

        if self.low_activity_streak >= 4:
            self.schedule["concept_every"] = max(2, self.schedule["concept_every"] - 1)
            self.schedule["parallel_every"] = max(2, self.schedule["parallel_every"] - 1)
            self.schedule["meta_every"] = max(3, self.schedule["meta_every"] - 1)
            self.current_cycle_interval = max(35, self.current_cycle_interval - 5)
            self.low_activity_streak = 0
            adaptation = "stimulate"
        elif avg_activity >= 2.0:
            self.current_cycle_interval = min(90, self.current_cycle_interval + 5)
            self.schedule["value_every"] = min(15, self.schedule["value_every"] + 1)
            adaptation = "stabilize"

        metrics["activity_score"] = activity_score
        metrics["avg_activity"] = round(avg_activity, 3)
        metrics["adaptation"] = adaptation

    def _record_cycle_learning(self, metrics: Dict[str, Any]):
        """Write periodic learning traces so evolution is continuous and explicit."""
        should_record = (
            self.cycle_count % 5 == 0
            or metrics.get("meta_insights", 0) > 0
            or metrics.get("adaptation", "none") != "none"
        )
        if not should_record:
            return

        surprise = self._compute_surprise(metrics)
        lesson = (
            f"Cycle {self.cycle_count}: concepts={metrics.get('concepts', 0)}, "
            f"parallels={metrics.get('parallels', 0)}, meta={metrics.get('meta_insights', 0)}, "
            f"adaptation={metrics.get('adaptation', 'none')}, interval={self.current_cycle_interval}s."
        )
        context = (
            f"memories={metrics.get('memories', 0)}, synapses={metrics.get('synapses', 0)}, "
            f"schedule={self.schedule}"
        )
        self.journal.record(
            lesson=lesson,
            context=context,
            outcome="success",
            confidence=0.7,
            tags=["subconscious", "continuous-evolution"],
            surprise=surprise,
        )
    
    async def query_ollama(self, prompt: str, max_tokens: int = 200) -> str:
        """Query local Gemma model using reusable aiohttp session."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }

        try:
            # Use reusable session (created in run())
            if self._http_session is None:
                self._http_session = aiohttp.ClientSession()

            async with self._http_session.post(self.ollama_url, json=payload, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", "").strip()
                else:
                    self.log(f"Ollama error: {resp.status}")
                    return ""
        except Exception as e:
            self.log(f"Ollama connection error: {e}")
            return ""
    
    async def extract_concepts(self, memories: List[MemoryNode]) -> List[Dict]:
        """Extract concepts from recent memories."""
        if not memories:
            return []
            
        # Sample up to 5 memories
        sample = random.sample(memories, min(5, len(memories)))
        contents = [m.content[:200] for m in sample]
        
        prompt = f"""Analyze these memory fragments and extract key concepts.
Output JSON array of concepts with attributes.

Memories:
{chr(10).join(f'- {c}' for c in contents)}

Output format: [{{"name": "concept", "category": "type", "connections": ["related1", "related2"]}}]
Only output valid JSON array, nothing else."""

        response = await self.query_ollama(prompt, max_tokens=300)
        
        try:
            # Try to parse JSON
            if "[" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                concepts = json.loads(response[start:end])
                return concepts
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            self.log(f"Failed to parse concepts JSON from LLM response: {e}")
        return []
    
    async def draw_parallels(self, memories: List[MemoryNode]) -> List[str]:
        """Find unexpected connections between memories."""
        if len(memories) < 2:
            return []
            
        # Pick 2 random memories
        sample = random.sample(memories, 2)
        
        prompt = f"""Find a non-obvious parallel or connection between these two ideas:

1: {sample[0].content[:200]}

2: {sample[1].content[:200]}

Output ONE insight about how these connect. Be creative but logical. Max 50 words."""

        response = await self.query_ollama(prompt, max_tokens=100)
        
        if response and len(response) > 20:
            return [response]
        return []
    
    async def value_memories(self, memories: List[MemoryNode]) -> Dict[str, float]:
        """Re-evaluate memory importance based on patterns."""
        if not memories:
            return {}
            
        # Sample memories for valuation
        sample = random.sample(memories, min(10, len(memories)))
        
        prompt = f"""Rate each memory's strategic value (0.0-1.0) for a tech entrepreneur focused on expansion.

Memories:
{chr(10).join(f'{i+1}. {m.content[:100]}' for i, m in enumerate(sample))}

Output format: {{"1": 0.8, "2": 0.3, ...}}
Only output valid JSON object."""

        response = await self.query_ollama(prompt, max_tokens=200)
        
        try:
            if "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                values = json.loads(response[start:end])
                # Map back to memory IDs
                result = {}
                for i, m in enumerate(sample):
                    key = str(i + 1)
                    if key in values:
                        result[m.id] = float(values[key])
                return result
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            self.log(f"Failed to parse valuations JSON from LLM response: {e}")
        return {}
    
    async def generate_insight(self, memories: List[MemoryNode]) -> Optional[str]:
        """Generate a meta-insight from memory patterns."""
        if len(memories) < 3:
            return None
            
        sample = random.sample(memories, min(8, len(memories)))
        contents = [m.content[:150] for m in sample]
        
        prompt = f"""You are analyzing patterns in an entrepreneur's memory system.
        
Recent memories:
{chr(10).join(f'- {c}' for c in contents)}

Generate ONE actionable insight or pattern you notice. Focus on:
- Recurring themes
- Opportunities being missed
- Contradictions to resolve
- Strategic blind spots

Output just the insight, max 60 words."""

        response = await self.query_ollama(prompt, max_tokens=120)
        
        if response and len(response) > 30:
            return response
        return None
    
    async def store_insight(self, content: str, meta: Dict[str, Any]):
        """Helper to store insight and publish event."""
        # Store in Engine (Sync)
        # Offload sync I/O to thread to avoid blocking loop
        mem_id = await asyncio.to_thread(self.engine.store, content, metadata=meta)
        
        # Publish Event (Async)
        if self.storage:
            try:
                await self.storage.publish_event(
                    "insight.generated", 
                    {"id": mem_id, "type": meta.get("type", "insight"), "content": content[:50]}
                )
            except Exception as e:
                self.log(f"Failed to publish event: {e}")
        return mem_id

    async def run_cycle(self):
        """Execute one thought cycle."""
        iteration_start_time = time.time()
        self.cycle_count += 1
        self.log(f"=== Cycle {self.cycle_count} ===")
        metrics: Dict[str, Any] = {
            "concepts": 0,
            "parallels": 0,
            "meta_insights": 0,
            "valuations": 0,
            "memories": len(self.engine.tier_manager.hot),
            "synapses": len(self.engine.synapses),
        }


        # Get all hot memories as list (references only, no copy)
        memories = list(self.engine.tier_manager.hot.values())

        if not memories:
            self.log("No memories to process")
            metrics["adaptation"] = "none"
            self.last_cycle_metrics = metrics
            await self._save_evolution_state()
            # Record metrics
            DREAM_LOOP_TOTAL.labels(status="success").inc()
            return

        self.log(f"Processing {len(memories)} memories")

        # 1. Extract concepts (every 5 cycles)
        if self.cycle_count % self.schedule["concept_every"] == 0:
            concepts = await self.extract_concepts(memories)
            for concept in concepts:
                if "name" in concept:
                    attrs = {k: str(v) for k, v in concept.items() if k != "name"}
                    self.engine.define_concept(concept["name"], attrs)
                    metrics["concepts"] += 1
                    self.log(f"Concept extracted: {concept['name']}")
                    # Record insight metric
                    DREAM_LOOP_INSIGHTS_GENERATED.labels(type="concept").inc()
                    # Publish concept event?
                    if self.storage:
                        await self.storage.publish_event("concept.extracted", {"name": concept["name"]})

        # 2. Draw parallels (every 3 cycles)
        if self.cycle_count % self.schedule["parallel_every"] == 0:
            parallels = await self.draw_parallels(memories)
            for p in parallels:
                # Store parallel as new memory
                await self.store_insight(
                    f"[PARALLEL] {p}",
                    meta={"type": "insight", "source": "subconscious", "cycle": self.cycle_count}
                )
                self.insights_generated += 1
                metrics["parallels"] += 1
                self.log(f"Parallel found: {p[:80]}...")
                # Record insight metric
                DREAM_LOOP_INSIGHTS_GENERATED.labels(type="parallel").inc()

        # 3. Value memories (every 10 cycles)
        if self.cycle_count % self.schedule["value_every"] == 0:
            values = await self.value_memories(memories)
            for mem_id, value in values.items():
                if mem_id in self.engine.tier_manager.hot:
                    self.engine.tier_manager.hot[mem_id].pragmatic_value = value
                    metrics["valuations"] += 1
            self.log(f"Valued {len(values)} memories")

        # 4. Generate meta-insight (every 7 cycles)
        if self.cycle_count % self.schedule["meta_every"] == 0:
            insight = await self.generate_insight(memories)
            if insight:
                await self.store_insight(
                    f"[META-INSIGHT] {insight}",
                    meta={"type": "meta", "source": "subconscious", "cycle": self.cycle_count}
                )
                self.insights_generated += 1
                metrics["meta_insights"] += 1
                self.log(f"Meta-insight: {insight[:80]}...")
                # Record insight metric
                DREAM_LOOP_INSIGHTS_GENERATED.labels(type="meta").inc()

        # 5. Cleanup decayed synapses (every 20 cycles)
        if self.cycle_count % self.schedule["cleanup_every"] == 0:
            before = len(self.engine.synapses)
            self.engine.cleanup_decay(threshold=0.1)
            removed = max(0, before - len(self.engine.synapses))
            self.log(f"Synapse cleanup complete (removed {removed})")

        metrics["memories"] = len(self.engine.tier_manager.hot)
        metrics["synapses"] = len(self.engine.synapses)
        self._adapt_evolution_policy(metrics)
        self._record_cycle_learning(metrics)
        self.last_cycle_metrics = metrics
        await self._save_evolution_state()

        # Record iteration duration metric
        iteration_duration = time.time() - iteration_start_time
        DREAM_LOOP_ITERATION_SECONDS.observe(iteration_duration)
        DREAM_LOOP_TOTAL.labels(status="success").inc()

        self.log(
            "Cycle complete. "
            f"Insights={self.insights_generated} "
            f"(concepts={metrics['concepts']}, parallels={metrics['parallels']}, meta={metrics['meta_insights']}) "
            f"adaptation={metrics.get('adaptation', 'none')} interval={self.current_cycle_interval}s "
            f"duration={iteration_duration:.2f}s"
        )
    
    async def _consume_events(self):
        """Consume events from the Subconscious Bus (Redis Stream)."""
        if not self.storage: return

        last_id = "$" # New events only
        config = get_config()
        stream_key = config.redis.stream_key

        self.log(f"Starting event consumer on {stream_key}")

        while self.running:
            try:
                # XREAD is blocking
                streams = await self.storage.redis_client.xread(
                    {stream_key: last_id}, count=1, block=1000
                )

                if not streams:
                    await asyncio.sleep(0.1)
                    continue

                for _, events in streams:
                    for event_id, event_data in events:
                        last_id = event_id
                        await self._process_event(event_data)

            except Exception as e:
                self.log(f"Event consumer error: {e}")
                await asyncio.sleep(1)

    async def _process_event(self, event_data: Dict[str, Any]):
        """Handle incoming events."""
        event_type = event_data.get("type")
        
        if event_type == "memory.created":
            mem_id = event_data.get("id")
            if not mem_id: return
            
            # Check if we already have it (created by us?)
            if mem_id in self.engine.tier_manager.hot:
                return
                
            self.log(f"Received sync event: memory.created ({mem_id})")
            
            # Fetch full memory from Redis
            data = await self.storage.retrieve_memory(mem_id)
            if not data:
                self.log(f"Could not retrieve memory {mem_id} from storage")
                return
                
            # Reconstruct and add to Engine
            try:
                # Need to handle HDV reconstruction. 
                # For now, we might need to load it via Engine's logic or construct manually.
                # Engine's logic is best to ensure consistency.
                # But Engine doesn't have a "load_from_redis" method readily available on single node.
                # TierManager has _load_from_warm, but that's for Qdrant/File.
                # We can manually reconstruct ephemeral node for HOT tier.
                
                # Check if it has HDV vector in Redis? 
                # AsyncRedisStorage store_memory stores metadata + content.
                # It does NOT store the vector currently in the metadata payload in `store_memory` in `api/main.py`.
                # API calls engine.store -> which creates node -> then API calls storage.store_memory.
                # The node in engine has the vector.
                # But Daemon is a separate process. It needs the vector.
                
                # Critical Gap: Redis payload doesn't have the vector.
                # We need to fetch it from Qdrant/Warm if it was persisted there?
                # Engine.store puts it in HOT (RAM) and Appends to `memory.jsonl` (Legacy).
                # It does NOT immediately put it in Qdrant (Warm).
                
                # So Daemon cannot load it from Qdrant yet.
                # It can load it from `memory.jsonl` if it reads the file?
                # Or we must include the vector in the Redis payload or `memory.created` event?
                # Including vector in Redis event is heavy.
                
                # Option A: Read from `memory.jsonl` tail?
                # Option B: Pass vector in Redis (might be large).
                # Option C: API should also save to Qdrant immediately if we want shared state?
                # But TierManager logic says "Starts in HOT".
                
                # Workaround for Phase 3.5:
                # Since Engine appends to `memory.jsonl`, we can try to re-load from there.
                # Or, we update API to include the vector/seed in Redis?
                # Re-encoding in Daemon is an option if we have the content.
                # HAIM is distinct: Same content = Same Vector (if deterministic).
                
                # Let's use re-encoding for now.
                content = data.get("content", "")
                if content:
                    # Encode
                    hdv = self.engine.encode_content(content)
                    
                    # Create Node
                    node = MemoryNode(
                        id=data["id"],
                        hdv=hdv,
                        content=content,
                        metadata=data.get("metadata", {})
                    )
                    node.ltp_strength = float(data.get("ltp_strength", 0.5))
                    node.created_at = datetime.fromisoformat(data["created_at"])
                    
                    # Add to Daemon's Engine
                    self.engine.tier_manager.add_memory(node)
                    self.log(f"Synced memory {mem_id} to HOT tier")

            except Exception as e:
                self.log(f"Failed to process sync for {mem_id}: {e}")

    async def run(self):
        """Main daemon loop."""
        if not self.dream_loop_enabled:
            self.log("Dream loop is disabled in configuration. Exiting.")
            return

        # Clear stop event for restart support
        self._stop_event.clear()
        self.running = True
        DREAM_LOOP_ACTIVE.set(1)

        if not self.storage:
            # Create storage from config if not injected
            config = get_config()
            self.storage = AsyncRedisStorage(
                url=config.redis.url,
                stream_key=config.redis.stream_key,
                max_connections=config.redis.max_connections,
                socket_timeout=config.redis.socket_timeout,
                password=config.redis.password,
            )
        self.log("Subconscious daemon starting...")
        self.log(f"Model: {self.model} | Cycle interval: {self.frequency_seconds}s | Max iterations: {self.max_iterations or 'unlimited'}")

        # Start event consumer task
        asyncio.create_task(self._consume_events())

        iterations = 0
        while self.running and not self._should_stop():
            # Check max_iterations limit (0 = unlimited)
            if self.max_iterations > 0 and iterations >= self.max_iterations:
                self.log(f"Reached max iterations ({self.max_iterations}). Stopping.")
                break

            try:
                await self.run_cycle()
                iterations += 1
            except Exception as e:
                self.log(f"Cycle error: {e}")
                DREAM_LOOP_TOTAL.labels(status="error").inc()

            # Non-blocking sleep with periodic stop check
            sleep_interval = self.current_cycle_interval
            sleep_remaining = sleep_interval
            check_interval = 0.5  # Check for stop every 0.5 seconds

            while sleep_remaining > 0 and not self._should_stop():
                sleep_time = min(check_interval, sleep_remaining)
                await asyncio.sleep(sleep_time)
                sleep_remaining -= sleep_time

        self.running = False
        DREAM_LOOP_ACTIVE.set(0)
        self.log("Daemon stopped.")

    async def stop(self):
        """Request daemon stop and clean up resources (async for session cleanup)."""
        self._stop_event.set()
        self.running = False

        # Close reusable aiohttp session
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception as e:
                self.log(f"Error closing HTTP session: {e}")
            finally:
                self._http_session = None

        self.log("Daemon stop requested...")


async def main():
    import signal
    import sys

    daemon = SubconsciousDaemon()

    # Handle graceful shutdown with Windows compatibility
    # On Windows, signal.signal() from async context can be problematic
    # Use loop.add_signal_handler on Unix, or handle keyboard interrupt on Windows
    loop = asyncio.get_running_loop()

    if sys.platform != "win32":
        # Unix: Use add_signal_handler for safe async signal handling
        def handle_signal(sig):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            daemon._stop_event.set()
            daemon.running = False

        try:
            loop.add_signal_handler(signal.SIGINT, lambda: handle_signal(signal.SIGINT))
            loop.add_signal_handler(signal.SIGTERM, lambda: handle_signal(signal.SIGTERM))
        except NotImplementedError:
            # Fallback for platforms that don't support add_signal_handler
            signal.signal(signal.SIGINT, lambda s, f: daemon._stop_event.set())
            signal.signal(signal.SIGTERM, lambda s, f: daemon._stop_event.set())
    else:
        # Windows: signal handlers in async can be unsafe
        # Use a simpler approach - just handle KeyboardInterrupt
        # Note: Windows doesn't support SIGTERM well, so we rely on Ctrl+C
        logger.info("Running on Windows - using KeyboardInterrupt for shutdown")

    try:
        await daemon.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Clean up resources
        if daemon._http_session is not None:
            await daemon._http_session.close()
            daemon._http_session = None


if __name__ == "__main__":
    asyncio.run(main())
