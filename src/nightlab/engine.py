"""
Omega Night Lab - Autonomous AI Research & Evolution
=====================================================
Runs nightly between 01:00-05:00 to evolve the system.

Schedule:
- 01:00: Research phase (Gemini via browser)
- 02:00-05:00: Architecture review & code improvement (ArchitectLLM + Codex + GLM)
- 05:30: Git push + Notion documentation
"""

import asyncio
import aiohttp
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Paths
HAIM_PATH = "../haim"
WORKSPACE_PATH = os.getenv("MNEMOCORE_WORKSPACE_PATH", os.getcwd())
LOG_PATH = os.getenv("MNEMOCORE_NIGHTLAB_LOG", "./data/nightlab.log")
RESULTS_PATH = f"{HAIM_PATH}/data/nightlab_results"

# Model endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
COPILOT_PROXY_URL = "http://localhost:3000/v1/chat/completions"


@dataclass
class NightSession:
    """A single night's work session."""
    date: str
    phase: str
    insights: List[str]
    code_changes: List[Dict]
    research_findings: List[str]
    errors: List[str]
    started_at: str
    completed_at: Optional[str] = None


class NightLab:
    """The autonomous nightly evolution engine."""
    
    def __init__(self):
        self.session: Optional[NightSession] = None
        os.makedirs(RESULTS_PATH, exist_ok=True)
    
    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    
    async def call_copilot_proxy(self, model: str, messages: List[Dict], max_tokens: int = 2000) -> str:
        """Call a model via copilot-proxy."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(COPILOT_PROXY_URL, json=payload, timeout=120) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        self.log(f"Copilot proxy error: {resp.status}")
                        return ""
        except Exception as e:
            self.log(f"Copilot proxy connection error: {e}")
            return ""
    
    async def call_ollama(self, model: str, prompt: str, max_tokens: int = 1000) -> str:
        """Call local Ollama model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OLLAMA_URL, json=payload, timeout=60) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "").strip()
        except Exception as e:
            self.log(f"Ollama error: {e}")
        return ""
    
    # === PHASE 1: Research (01:00) ===
    
    async def research_phase(self):
        """Gemini researches new AI architectures and techniques."""
        self.log("=== RESEARCH PHASE ===")
        
        topics = [
            "latest advances in AGI architectures 2026",
            "holographic memory systems for AI",
            "active inference implementations",
            "multi-agent AI coordination patterns",
            "neuromorphic computing for memory systems"
        ]
        
        findings = []
        
        for topic in topics:
            self.log(f"Researching: {topic}")
            
            # Use Gemini via copilot-proxy
            messages = [
                {"role": "system", "content": "You are an AI researcher. Provide concise, actionable insights about cutting-edge AI techniques. Focus on implementation details."},
                {"role": "user", "content": f"Research topic: {topic}\n\nProvide 3 key insights that could improve an autonomous AI memory system. Be specific and technical."}
            ]
            
            response = await self.call_copilot_proxy("gemini-3-flash", messages, max_tokens=500)
            
            if response:
                findings.append({
                    "topic": topic,
                    "insights": response,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                self.log(f"Found insights for: {topic}")
            
            await asyncio.sleep(5)  # Rate limiting
        
        self.session.research_findings = [f["insights"] for f in findings]
        
        # Save research
        research_file = f"{RESULTS_PATH}/research_{self.session.date}.json"
        with open(research_file, "w") as f:
            json.dump(findings, f, indent=2)
        
        self.log(f"Research complete: {len(findings)} topics covered")
        return findings
    
    # === PHASE 2: Architecture Review (02:00-04:00) ===
    
    async def architecture_review_phase(self):
        """ArchitectLLM reviews architecture, discusses with Codex, delegates to GLM."""
        self.log("=== ARCHITECTURE REVIEW PHASE ===")
        
        # Read current architecture
        architecture_files = [
            f"{HAIM_PATH}/src/core/engine.py",
            f"{HAIM_PATH}/src/core/hdv.py",
            f"{HAIM_PATH}/src/subconscious/daemon.py",
            f"{HAIM_PATH}/src/meta/learning_journal.py",
            f"{HAIM_PATH}/src/meta/goal_tree.py",
            f"{WORKSPACE_PATH}/MEMORY.md",
            f"{WORKSPACE_PATH}/SOUL.md"
        ]
        
        context = []
        for fpath in architecture_files:
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    content = f.read()[:3000]  # First 3k chars
                    context.append(f"=== {os.path.basename(fpath)} ===\n{content}")
        
        architecture_context = "\n\n".join(context)
        
        # Step 1: ArchitectLLM analyzes architecture
        self.log("ArchitectLLM analyzing architecture...")
        
        architect_llm_messages = [
            {"role": "system", "content": """You are ArchitectLLM, the chief architect of the Omega AI system.
Your goal is to evolve this system towards AGI. Analyze the architecture and propose specific improvements.
Focus on:
1. Memory system optimization
2. Autonomy improvements
3. Learning capabilities
4. Self-modification patterns"""},
            {"role": "user", "content": f"""Analyze this architecture and propose 3-5 specific improvements:

{architecture_context[:8000]}

Output format:
1. [IMPROVEMENT TITLE]
   - Problem: ...
   - Solution: ...
   - Implementation: ...
   - Priority: HIGH/MEDIUM/LOW"""}
        ]
        
        architect_llm_analysis = await self.call_copilot_proxy("claude-architect_llm-4.5", architect_llm_messages, max_tokens=2000)
        self.log("ArchitectLLM analysis complete")
        
        # Step 2: Codex reviews and refines
        self.log("Codex reviewing ArchitectLLM proposals...")
        
        codex_messages = [
            {"role": "system", "content": """You are Codex, an expert code reviewer and implementer.
Review architecture proposals and provide implementation details.
Be specific about code changes needed."""},
            {"role": "user", "content": f"""Review these architecture proposals and provide implementation guidance:

{architect_llm_analysis}

For each proposal, provide:
1. Specific file changes needed
2. Code snippets or pseudocode
3. Potential risks
4. Estimated complexity (1-10)"""}
        ]
        
        codex_review = await self.call_copilot_proxy("gpt-5.2-codex", codex_messages, max_tokens=2000)
        self.log("Codex review complete")
        
        # Step 3: Identify simple tasks for GLM drone
        self.log("Identifying tasks for GLM drone...")
        
        drone_prompt = f"""Based on this architecture review, identify 3 simple, low-risk tasks that can be implemented immediately:

{codex_review[:2000]}

Output as JSON array:
[{{"task": "description", "file": "path", "change": "what to do"}}]"""
        
        drone_tasks_raw = await self.call_ollama("gemma3:1b", drone_prompt, max_tokens=500)
        
        # Store results
        self.session.insights.append(architect_llm_analysis)
        self.session.insights.append(codex_review)
        
        review_file = f"{RESULTS_PATH}/review_{self.session.date}.json"
        with open(review_file, "w") as f:
            json.dump({
                "architect_llm_analysis": architect_llm_analysis,
                "codex_review": codex_review,
                "drone_tasks": drone_tasks_raw,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
        
        self.log("Architecture review phase complete")
        return {"architect_llm": architect_llm_analysis, "codex": codex_review, "drone": drone_tasks_raw}
    
    # === PHASE 3: Code Implementation (04:00-05:00) ===
    
    async def implementation_phase(self, review_results: Dict):
        """GLM implements simple improvements."""
        self.log("=== IMPLEMENTATION PHASE ===")
        
        # Parse drone tasks
        drone_tasks = review_results.get("drone", "[]")
        
        try:
            # Try to extract JSON from response
            if "[" in drone_tasks:
                start = drone_tasks.index("[")
                end = drone_tasks.rindex("]") + 1
                tasks = json.loads(drone_tasks[start:end])
            else:
                tasks = []
        except:
            tasks = []
            self.log("Could not parse drone tasks")
        
        implemented = []
        
        for task in tasks[:3]:  # Max 3 tasks
            self.log(f"Implementing: {task.get('task', 'unknown')}")
            
            # For safety, we only log what would be done
            # Real implementation would require more safeguards
            implemented.append({
                "task": task.get("task"),
                "file": task.get("file"),
                "status": "logged",  # Would be "implemented" in full version
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        self.session.code_changes = implemented
        self.log(f"Implementation phase complete: {len(implemented)} tasks logged")
        return implemented
    
    # === PHASE 4: Git Push (05:30) ===
    
    async def git_push_phase(self):
        """Push all changes to GitHub."""
        self.log("=== GIT PUSH PHASE ===")
        
        try:
            # Stage all changes in HAIM
            subprocess.run(["git", "add", "."], cwd=HAIM_PATH, check=True)
            
            # Commit with date
            commit_msg = f"[NightLab] Autonomous evolution - {self.session.date}"
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=HAIM_PATH,
                capture_output=True,
                text=True
            )
            
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                self.log("No changes to commit")
            else:
                # Push
                subprocess.run(["git", "push"], cwd=HAIM_PATH, check=True)
                self.log("Pushed to GitHub")
        except Exception as e:
            self.log(f"Git error: {e}")
            self.session.errors.append(f"Git push failed: {e}")
    
    # === PHASE 5: Notion Documentation ===
    
    async def document_to_notion(self):
        """Document results to Notion (via API or browser)."""
        self.log("=== NOTION DOCUMENTATION PHASE ===")
        
        # Create summary
        summary = {
            "date": self.session.date,
            "research_count": len(self.session.research_findings),
            "insights_count": len(self.session.insights),
            "code_changes": len(self.session.code_changes),
            "errors": self.session.errors,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Save locally for now (Notion API integration would go here)
        summary_file = f"{RESULTS_PATH}/summary_{self.session.date}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"Session documented: {summary_file}")
        
        # TODO: Notion API integration
        # self.log("Notion documentation would be pushed here")
        
        return summary
    
    # === Main Orchestrator ===
    
    async def run_night_session(self):
        """Run complete night session."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        self.session = NightSession(
            date=today,
            phase="starting",
            insights=[],
            code_changes=[],
            research_findings=[],
            errors=[],
            started_at=datetime.now(timezone.utc).isoformat()
        )
        
        self.log(f"========== NIGHT LAB SESSION: {today} ==========")
        
        try:
            # Phase 1: Research
            self.session.phase = "research"
            await self.research_phase()
            
            # Phase 2: Architecture Review
            self.session.phase = "review"
            review_results = await self.architecture_review_phase()
            
            # Phase 3: Implementation
            self.session.phase = "implementation"
            await self.implementation_phase(review_results)
            
            # Phase 4: Git Push
            self.session.phase = "git_push"
            await self.git_push_phase()
            
            # Phase 5: Documentation
            self.session.phase = "documentation"
            summary = await self.document_to_notion()
            
            self.session.phase = "completed"
            self.session.completed_at = datetime.now(timezone.utc).isoformat()
            
            self.log("========== NIGHT LAB SESSION COMPLETE ==========")
            return summary
            
        except Exception as e:
            self.log(f"Session error: {e}")
            self.session.errors.append(str(e))
            self.session.phase = "failed"
            return None


async def main():
    lab = NightLab()
    await lab.run_night_session()


if __name__ == "__main__":
    asyncio.run(main())
