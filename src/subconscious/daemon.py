"""
Omega Subconscious Daemon
=========================
Continuous background processing using Gemma 1B via Ollama.
Performs: concept extraction, parallel drawing, memory valuation, thought sorting.
Integrates with Redis Subconscious Bus to publish insights.
"""

import asyncio
import aiohttp
import json
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.engine import HAIMEngine
from src.core.async_storage import AsyncRedisStorage
from src.meta.learning_journal import LearningJournal

# Config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:1b"
HAIM_DATA_PATH = "./data/memory.jsonl"
CYCLE_INTERVAL = 60  # seconds between thought cycles
LOG_PATH = "/tmp/subconscious.log"
EVOLUTION_STATE_PATH = "./data/subconscious_evolution.json"


class SubconsciousDaemon:
    """The always-running background mind."""
    
    def __init__(self):
        self.engine = HAIMEngine(persist_path=HAIM_DATA_PATH)
        self.journal = LearningJournal()
        self.running = False
        self.cycle_count = 0
        self.insights_generated = 0
        self.current_cycle_interval = CYCLE_INTERVAL
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
        
        # Async Redis Storage (initialized in run)
        self.storage: Optional[AsyncRedisStorage] = None
        
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

    def _save_evolution_state(self):
        """Persist state so evolution continues across restarts."""
        try:
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
            os.makedirs(os.path.dirname(EVOLUTION_STATE_PATH), exist_ok=True)
            with open(EVOLUTION_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
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
        """Query local Gemma model."""
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OLLAMA_URL, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "").strip()
                    else:
                        self.log(f"Ollama error: {resp.status}")
                        return ""
        except Exception as e:
            self.log(f"Ollama connection error: {e}")
            return ""
    
    async def extract_concepts(self, memories: List[Dict]) -> List[Dict]:
        """Extract concepts from recent memories."""
        if not memories:
            return []
            
        # Sample up to 5 memories
        sample = random.sample(memories, min(5, len(memories)))
        contents = [m.get("content", "")[:200] for m in sample]
        
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
        except:
            pass
        return []
    
    async def draw_parallels(self, memories: List[Dict]) -> List[str]:
        """Find unexpected connections between memories."""
        if len(memories) < 2:
            return []
            
        # Pick 2 random memories
        sample = random.sample(memories, 2)
        
        prompt = f"""Find a non-obvious parallel or connection between these two ideas:

1: {sample[0].get('content', '')[:200]}

2: {sample[1].get('content', '')[:200]}

Output ONE insight about how these connect. Be creative but logical. Max 50 words."""

        response = await self.query_ollama(prompt, max_tokens=100)
        
        if response and len(response) > 20:
            return [response]
        return []
    
    async def value_memories(self, memories: List[Dict]) -> Dict[str, float]:
        """Re-evaluate memory importance based on patterns."""
        if not memories:
            return {}
            
        # Sample memories for valuation
        sample = random.sample(memories, min(10, len(memories)))
        
        prompt = f"""Rate each memory's strategic value (0.0-1.0) for a tech entrepreneur focused on expansion.

Memories:
{chr(10).join(f'{i+1}. {m.get("content", "")[:100]}' for i, m in enumerate(sample))}

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
                        result[m.get("id", f"mem_{i}")] = float(values[key])
                return result
        except:
            pass
        return {}
    
    async def generate_insight(self, memories: List[Dict]) -> Optional[str]:
        """Generate a meta-insight from memory patterns."""
        if len(memories) < 3:
            return None
            
        sample = random.sample(memories, min(8, len(memories)))
        contents = [m.get("content", "")[:150] for m in sample]
        
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
        self.cycle_count += 1
        self.log(f"=== Cycle {self.cycle_count} ===")
        metrics: Dict[str, Any] = {
            "concepts": 0,
            "parallels": 0,
            "meta_insights": 0,
            "valuations": 0,
            "memories": len(self.engine.memory_nodes),
            "synapses": len(self.engine.synapses),
        }

        
        # Get all memories as list
        memories = [
            {"id": nid, "content": node.content, "metadata": node.metadata}
            for nid, node in self.engine.memory_nodes.items()
        ]
        
        if not memories:
            self.log("No memories to process")
            metrics["adaptation"] = "none"
            self.last_cycle_metrics = metrics
            self._save_evolution_state()
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
        
        # 3. Value memories (every 10 cycles)
        if self.cycle_count % self.schedule["value_every"] == 0:
            values = await self.value_memories(memories)
            for mem_id, value in values.items():
                if mem_id in self.engine.memory_nodes:
                    self.engine.memory_nodes[mem_id].pragmatic_value = value
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
        
        # 5. Cleanup decayed synapses (every 20 cycles)
        if self.cycle_count % self.schedule["cleanup_every"] == 0:
            before = len(self.engine.synapses)
            self.engine.cleanup_decay(threshold=0.1)
            removed = max(0, before - len(self.engine.synapses))
            self.log(f"Synapse cleanup complete (removed {removed})")

        metrics["memories"] = len(self.engine.memory_nodes)
        metrics["synapses"] = len(self.engine.synapses)
        self._adapt_evolution_policy(metrics)
        self._record_cycle_learning(metrics)
        self.last_cycle_metrics = metrics
        self._save_evolution_state()
        
        self.log(
            "Cycle complete. "
            f"Insights={self.insights_generated} "
            f"(concepts={metrics['concepts']}, parallels={metrics['parallels']}, meta={metrics['meta_insights']}) "
            f"adaptation={metrics.get('adaptation', 'none')} interval={self.current_cycle_interval}s"
        )
    
    async def run(self):
        """Main daemon loop."""
        self.running = True
        self.storage = AsyncRedisStorage.get_instance() # Initialize singleton
        self.log("Subconscious daemon starting...")
        self.log(f"Model: {MODEL} | Cycle interval: {CYCLE_INTERVAL}s")
        
        while self.running:
            try:
                await self.run_cycle()
            except Exception as e:
                self.log(f"Cycle error: {e}")
            
            await asyncio.sleep(self.current_cycle_interval)
    
    def stop(self):
        self.running = False
        self.log("Daemon stopping...")
        if self.storage:
            pass


async def main():
    daemon = SubconsciousDaemon()
    
    # Handle graceful shutdown
    import signal
    def shutdown(sig, frame):
        daemon.stop()
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    await daemon.run()


if __name__ == "__main__":
    asyncio.run(main())
