"""
Night Lab Scheduler
===================
Cron-like scheduler for nightly AI evolution sessions.
"""

import asyncio
import schedule
import time
from datetime import datetime
from engine import NightLab

LOG_PATH = "/tmp/nightlab_scheduler.log"


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_research_phase():
    """01:00 - Research phase."""
    log("Triggering research phase...")
    lab = NightLab()
    asyncio.run(lab.research_phase())


def run_full_session():
    """02:00 - Full architecture review session."""
    log("Triggering full night session...")
    lab = NightLab()
    asyncio.run(lab.run_night_session())


def main():
    log("Night Lab Scheduler starting...")
    
    # Schedule jobs
    schedule.every().day.at("01:00").do(run_research_phase)
    schedule.every().day.at("02:00").do(run_full_session)
    
    log("Scheduled:")
    log("  - 01:00: Research phase (Gemini)")
    log("  - 02:00: Full session (ArchitectLLM + Codex + GLM)")
    
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
