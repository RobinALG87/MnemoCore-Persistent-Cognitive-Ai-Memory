"""
MnemoCore Meta Module
=====================
Meta-cognitive components for goal tracking and learning introspection.

This module provides higher-order cognitive capabilities:

Components:
    - GoalTree: Hierarchical task and goal management
    - LearningJournal: Persistent log of learning events and insights

The meta layer enables the system to reason about its own state,
track progress toward goals, and maintain a record of learning
milestones and discoveries.

Usage:
    from mnemocore.meta import GoalTree, LearningJournal

    goals = GoalTree()
    goals.add_goal("Improve memory recall accuracy")
    journal = LearningJournal()
    journal.record_insight("Discovered pattern in user queries")
"""

