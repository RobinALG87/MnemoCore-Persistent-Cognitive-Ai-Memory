"""
MnemoCore 60-Second "Wow" Demo
Showcasing Persistent Cognitive Memory, Analogy Reasoning, and Decay.
Run this directly: `python demo_wow.py`
"""

import asyncio
import tempfile
import os
from mnemocore.core.engine import HAIMEngine

async def main():
    print("[1] Booting MnemoCore HAIM Engine...")
    
    # We use a temporary directory for this quick demo so we don't leave files behind.
    temp_dir = tempfile.mkdtemp()
    demo_db_path = os.path.join(temp_dir, "demo_memory.jsonl")
    
    engine = HAIMEngine(persist_path=demo_db_path)
    
    # Temporarily override the conceptual memory storage to avoid corrupting the real ConceptualMemory
    from mnemocore.core.holographic import ConceptualMemory
    engine.soul = ConceptualMemory(dimension=engine.dimension, storage_dir=temp_dir)
    
    await engine.initialize()
    
    print("\n[2] Teaching the AI facts and procedures...")
    await engine.store("The company's main language is Python.", goal_id="onboarding")
    await engine.store("If the database locks, restart the Redis pod.", metadata={"type": "procedure"})
    
    # Wait for the async worker to process the events
    print("   [Syncing memory tiers...]")
    await asyncio.sleep(0.5)
    
    print("\n[3] Retrieving context associatively...")
    results = await engine.query("If the database locks, what should I do?")
    print(f"   [Debug: raw query results: {results}]")
    if results:
        mem = await engine.get_memory(results[0][0])
        print(f"   -> Top memory retrieved: '{mem.content}' (Score: {results[0][1]:.2f})")
    else:
        print("   -> No memories found.")

    print("\n[4] VSA Analogical Reasoning (Math in 16k-dimensions)...")
    await engine.define_concept("CTO", {"role": "technical_leader", "gender": "neutral"})
    await engine.define_concept("CEO", {"role": "business_leader", "gender": "neutral"})
    await engine.define_concept("code", {"domain": "technical_leader"})
    
    print("   If CTO : code :: CEO : ?")
    analogy = await engine.reason_by_analogy(src="CTO", val="code", tgt="CEO")
    print(f"   -> The engine deduced: {analogy[0][0]} (Confidence: {analogy[0][1]:.2f})")

    print("\n[Success] MnemoCore transforms data storage into cognitive reasoning. Try it in your agents today!")
    await engine.close()
    
    # Clean up temp file
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)
    if os.path.exists(os.path.join(temp_dir, "codebook.json")):
        os.remove(os.path.join(temp_dir, "codebook.json"))
    if os.path.exists(os.path.join(temp_dir, "concepts.json")):
        os.remove(os.path.join(temp_dir, "concepts.json"))
    os.rmdir(temp_dir)

if __name__ == "__main__":
    asyncio.run(main())
