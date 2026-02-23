"""
CLI Output Formatters

Provides formatted output for CLI commands including tables,
progress bars, and colored output.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tabulate import tabulate


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def green(text: str) -> str:
        return f"{Colors.OKGREEN}{text}{Colors.ENDC}"

    @staticmethod
    def red(text: str) -> str:
        return f"{Colors.FAIL}{text}{Colors.ENDC}"

    @staticmethod
    def yellow(text: str) -> str:
        return f"{Colors.WARNING}{text}{Colors.ENDC}"

    @staticmethod
    def blue(text: str) -> str:
        return f"{Colors.OKBLUE}{text}{Colors.ENDC}"

    @staticmethod
    def bold(text: str) -> str:
        return f"{Colors.BOLD}{text}{Colors.ENDC}"


@dataclass
class MemoryResult:
    """Formatted memory result for CLI output."""
    id: str
    content: str
    score: float
    tier: str
    created_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_table_row(self) -> List[str]:
        """Convert to table row."""
        content_preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        created = self.created_at.split("T")[0] if self.created_at else "N/A"
        tags_str = ", ".join(self.tags[:3]) if self.tags else ""
        return [
            self.id[:12] + "...",
            content_preview,
            f"{self.score:.2f}",
            self.tier.upper(),
            created,
            tags_str,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "tier": self.tier,
            "created_at": self.created_at,
            "tags": self.tags,
            "metadata": self.metadata,
        }


def format_memory_table(memories: List[MemoryResult], show_headers: bool = True) -> str:
    """
    Format memories as a table.

    Args:
        memories: List of memory results
        show_headers: Whether to show table headers

    Returns:
        Formatted table string
    """
    if not memories:
        return "No memories found."

    headers = ["ID", "Content", "Score", "Tier", "Created", "Tags"]
    rows = [m.to_table_row() for m in memories]

    return tabulate(rows, headers=headers if show_headers else [], tablefmt="grid")


def format_stats(stats: Dict[str, Any]) -> str:
    """
    Format engine statistics for display.

    Args:
        stats: Statistics dictionary from engine.get_stats()

    Returns:
        Formatted statistics string
    """
    lines = []
    lines.append("=" * 50)
    lines.append(Colors.bold("MnemoCore System Statistics"))
    lines.append("=" * 50)
    lines.append("")

    # Basic info
    lines.append(f"{Colors.bold('Engine Version:')} {stats.get('engine_version', 'unknown')}")
    lines.append(f"{Colors.bold('Dimension:')} {stats.get('dimension', 'unknown')}")
    lines.append(f"{Colors.bold('Encoding:')} {stats.get('encoding', 'unknown')}")
    lines.append("")

    # Tier stats
    tiers = stats.get("tiers", {})
    if tiers:
        lines.append(Colors.bold("Memory Tiers:"))
        total = 0
        for tier_name in ["hot", "warm", "cold"]:
            if tier_name in tiers:
                tier_data = tiers[tier_name]
                count = tier_data.get("count", 0)
                total += count
                max_mem = tier_data.get("max", "unlimited")
                color = Colors.green if tier_name == "hot" else Colors.blue if tier_name == "warm" else Colors.yellow
                lines.append(f"  {color(tier_name.upper()):<6} {count:>6} memories (max: {max_mem})")
        lines.append(f"  {'TOTAL':<6} {total:>6} memories")
        lines.append("")

    # Concepts and synapses
    lines.append(f"{Colors.bold('Concepts:')}    {stats.get('concepts_count', 0):>6}")
    lines.append(f"{Colors.bold('Symbols:')}     {stats.get('symbols_count', 0):>6}")
    lines.append(f"{Colors.bold('Synapses:')}    {stats.get('synapses_count', 0):>6}")
    lines.append("")

    # Background workers
    workers = stats.get("background_workers", {})
    if workers:
        lines.append(Colors.bold("Background Workers:"))
        for name, worker in workers.items():
            status = worker.get("running", False)
            status_text = Colors.green("running") if status else Colors.red("stopped")
            lines.append(f"  {name}: {status_text}")
        lines.append("")

    lines.append("=" * 50)

    return "\n".join(lines)


def format_health(health: Dict[str, Any]) -> str:
    """
    Format health check results for display.

    Args:
        health: Health check dictionary

    Returns:
        Formatted health string
    """
    lines = []
    lines.append("=" * 40)
    lines.append(Colors.bold("MnemoCore Health Status"))
    lines.append("=" * 40)
    lines.append("")

    # Overall status
    status = health.get("status", "unknown")
    if status == "healthy":
        status_text = Colors.green(status.upper())
    elif status == "degraded":
        status_text = Colors.yellow(status.upper())
    else:
        status_text = Colors.red(status.upper())

    lines.append(f"{Colors.bold('Status:')} {status_text}")
    lines.append(f"{Colors.bold('Initialized:')} {health.get('initialized', False)}")
    lines.append(f"{Colors.bold('Timestamp:')} {health.get('timestamp', 'N/A')}")
    lines.append("")

    # Tier stats
    tiers = health.get("tiers", {})
    if tiers:
        lines.append(Colors.bold("Tiers:"))
        total = 0
        for tier_name, tier_stats in tiers.items():
            if isinstance(tier_stats, dict):
                count = tier_stats.get("count", 0)
                total += count
                lines.append(f"  {tier_name.upper()}: {count} memories")
        lines.append(f"  Total: {total} memories")
        lines.append("")

    # Background workers
    workers = health.get("background_workers", {})
    if workers:
        lines.append(Colors.bold("Background Workers:"))
        for name, worker in workers.items():
            running = worker.get("running", False)
            status_str = Colors.green("Running") if running else Colors.red("Stopped")
            lines.append(f"  {name}: {status_str}")
        lines.append("")

    # Qdrant
    qdrant = health.get("qdrant")
    if qdrant:
        if "error" in qdrant:
            lines.append(f"{Colors.red('Qdrant:')} Error - {qdrant['error']}")
        else:
            lines.append(f"{Colors.green('Qdrant:')} Connected")

    lines.append("=" * 40)

    return "\n".join(lines)


def format_dream_report(report: Dict[str, Any]) -> str:
    """
    Format dream report for display.

    Args:
        report: Dream report dictionary

    Returns:
        Formatted dream report string
    """
    lines = []
    lines.append("=" * 50)
    lines.append(Colors.bold("Dream Session Report"))
    lines.append("=" * 50)
    lines.append("")

    # Summary
    summary = report.get("summary", {})
    lines.append(Colors.bold("Summary:"))
    lines.append(f"  Duration: {report.get('session_duration_seconds', 0):.1f} seconds")
    lines.append(f"  Episodic clusters: {summary.get('episodic_clusters_found', 0)}")
    lines.append(f"  Patterns discovered: {summary.get('patterns_discovered', 0)}")
    lines.append(f"  Synthesis insights: {summary.get('synthesis_insights', 0)}")
    lines.append(f"  Contradictions found: {summary.get('contradictions_found', 0)}")
    lines.append(f"  Contradictions resolved: {summary.get('contradictions_resolved', 0)}")
    lines.append(f"  Memories promoted: {summary.get('memories_promoted', 0)}")

    # Health score
    health = summary.get("overall_health", {})
    if health:
        score = health.get("score", 0)
        status = health.get("status", "unknown")
        if status == "healthy":
            status_color = Colors.green
        elif status == "attention_needed":
            status_color = Colors.yellow
        else:
            status_color = Colors.red
        lines.append(f"  Health Score: {status_color(str(score))}/100 ({status})")
    lines.append("")

    # Top patterns
    pattern_analysis = report.get("pattern_discoveries", {})
    top_patterns = pattern_analysis.get("top_patterns", [])
    if top_patterns:
        lines.append(Colors.bold("Top Patterns:"))
        for i, pattern in enumerate(top_patterns[:5], 1):
            ptype = pattern.get("type", "unknown")
            value = pattern.get("value", "N/A")
            freq = pattern.get("frequency", 0)
            lines.append(f"  {i}. [{ptype}] {value} (freq: {freq})")
        lines.append("")

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.append(Colors.bold("Recommendations:"))
        for rec in recommendations:
            lines.append(f"  - {rec}")
        lines.append("")

    lines.append("=" * 50)

    return "\n".join(lines)


def format_export_summary(result: Dict[str, Any]) -> str:
    """
    Format export result summary.

    Args:
        result: Export result dictionary

    Returns:
        Formatted export summary string
    """
    lines = []
    lines.append("=" * 40)
    lines.append(Colors.bold("Export Summary"))
    lines.append("=" * 40)
    lines.append("")

    success = result.get("success", False)
    if success:
        lines.append(Colors.green("Export completed successfully"))
        lines.append(f"  Records: {result.get('records_exported', 0)}")
        lines.append(f"  Size: {result.get('size_bytes', 0) / 1024 / 1024:.2f} MB")
        lines.append(f"  Duration: {result.get('duration_seconds', 0):.1f} seconds")
        lines.append(f"  Output: {result.get('output_path', 'N/A')}")
    else:
        lines.append(Colors.red("Export failed"))
        lines.append(f"  Error: {result.get('error_message', 'Unknown error')}")

    lines.append("")
    lines.append("=" * 40)

    return "\n".join(lines)


def truncate(text: str, max_length: int = 80, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_timestamp(ts: Optional[str], format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format ISO timestamp for display.

    Args:
        ts: ISO timestamp string
        format: Output format string

    Returns:
        Formatted timestamp string
    """
    if not ts:
        return "N/A"

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime(format)
    except (ValueError, AttributeError):
        return ts


def get_terminal_width(default: int = 80) -> int:
    """
    Get terminal width.

    Args:
        default: Default width if detection fails

    Returns:
        Terminal width in characters
    """
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except Exception:
        return default


def print_box(text: str, padding: int = 1) -> None:
    """
    Print text in a box.

    Args:
        text: Text to print
        padding: Padding inside the box
    """
    width = get_terminal_width()
    lines = text.split("\n")

    # Calculate max line length
    max_len = max(len(line) for line in lines)
    box_width = min(max_len + 2 * padding + 2, width)

    # Print top border
    print("=" * box_width)

    # Print lines with padding
    for line in lines:
        padded = " " * padding + line + " " * padding
        print(padded[:box_width])

    # Print bottom border
    print("=" * box_width)


def format_progress(current: int, total: int, width: int = 40) -> str:
    """
    Format a progress bar.

    Args:
        current: Current progress
        total: Total items
        width: Width of progress bar

    Returns:
        Formatted progress bar string
    """
    if total == 0:
        return "[" + " " * width + "]"

    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    percentage = 100 * current / total

    return f"[{bar}] {percentage:.0f}%"
