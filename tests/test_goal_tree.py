"""
Tests for Goal Tree Module
==========================
Comprehensive tests for hierarchical goal decomposition with CRUD operations,
persistence, and cascading operations.

Tests cover:
- CRUD: add, complete, block, decompose
- ID uniqueness
- Persistence: save to file, load from file
- Corrupt file handling
- stats() returns correct counts
- Decompose cascading
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mnemocore.meta.goal_tree import GoalTree, Goal, GoalStatus


@pytest.fixture
def temp_goals_path():
    """Create a temporary file path for goals storage."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def empty_goal_tree(temp_goals_path):
    """Create an empty GoalTree with temporary storage."""
    return GoalTree(path=temp_goals_path)


@pytest.fixture
def populated_goal_tree(empty_goal_tree):
    """Create a GoalTree with sample goals."""
    gt = empty_goal_tree

    # Add root goals
    root1 = gt.add(
        title="Complete Project",
        description="Finish the MnemoCore project",
        priority=0.9,
        tags=["work", "priority"]
    )
    root2 = gt.add(
        title="Learn New Skill",
        description="Master Python async programming",
        priority=0.7,
        tags=["learning"]
    )

    # Decompose root1 into sub-goals
    gt.decompose(root1, [
        {"title": "Design Architecture", "description": "Create system design", "priority": 0.8},
        {"title": "Implement Core", "description": "Build core features", "priority": 0.9},
        {"title": "Write Tests", "description": "Add test coverage", "priority": 0.7}
    ])

    return gt


class TestGoalDataclass:
    """Tests for the Goal dataclass."""

    def test_goal_creation(self):
        """Test basic Goal creation with default values."""
        goal = Goal(id="test_goal", title="Test", description="A test goal")
        assert goal.id == "test_goal"
        assert goal.title == "Test"
        assert goal.description == "A test goal"
        assert goal.parent_id is None
        assert goal.status == "active"
        assert goal.priority == 0.5
        assert goal.progress == 0.0
        assert goal.tags == []
        assert goal.blockers == []

    def test_goal_is_leaf_no_children(self):
        """Test is_leaf returns True when goal has no children."""
        goal = Goal(id="leaf_goal", title="Leaf", description="Leaf goal")
        all_goals = {"leaf_goal": goal}
        assert goal.is_leaf(all_goals) is True

    def test_goal_is_leaf_with_children(self):
        """Test is_leaf returns False when goal has children."""
        parent = Goal(id="parent", title="Parent", description="Parent goal")
        child = Goal(id="child", title="Child", description="Child goal", parent_id="parent")
        all_goals = {"parent": parent, "child": child}
        assert parent.is_leaf(all_goals) is False
        assert child.is_leaf(all_goals) is True


class TestGoalTreeCRUD:
    """Tests for GoalTree CRUD operations."""

    def test_add_goal(self, empty_goal_tree):
        """Test adding a new goal returns a unique ID."""
        gt = empty_goal_tree

        goal_id = gt.add(
            title="New Goal",
            description="A new goal to achieve",
            priority=0.8
        )

        assert goal_id is not None
        assert goal_id.startswith("goal_")
        assert len(goal_id) > 5  # Should have UUID suffix
        assert goal_id in gt.goals

    def test_add_goal_with_all_parameters(self, empty_goal_tree):
        """Test adding a goal with all optional parameters."""
        gt = empty_goal_tree
        deadline = "2025-12-31T23:59:59Z"

        goal_id = gt.add(
            title="Full Goal",
            description="Goal with all fields",
            parent_id="parent_123",
            priority=0.95,
            deadline=deadline,
            tags=["urgent", "quarterly"]
        )

        goal = gt.goals[goal_id]
        assert goal.title == "Full Goal"
        assert goal.description == "Goal with all fields"
        assert goal.parent_id == "parent_123"
        assert goal.priority == 0.95
        assert goal.deadline == deadline
        assert "urgent" in goal.tags
        assert "quarterly" in goal.tags

    def test_complete_goal(self, populated_goal_tree):
        """Test marking a goal as completed."""
        gt = populated_goal_tree

        # Get a leaf goal
        leaf_goals = [g for g in gt.goals.values() if g.is_leaf(gt.goals)]
        leaf_goal = leaf_goals[0]

        gt.complete(leaf_goal.id)

        assert gt.goals[leaf_goal.id].status == GoalStatus.COMPLETED.value
        assert gt.goals[leaf_goal.id].progress == 1.0

    def test_complete_updates_parent_progress(self, populated_goal_tree):
        """Test completing a child goal updates parent progress."""
        gt = populated_goal_tree

        # Find a parent goal
        parent_goals = [g for g in gt.goals.values() if g.parent_id is None]
        parent = parent_goals[0]

        # Get children of this parent
        children = [g for g in gt.goals.values() if g.parent_id == parent.id]
        assert len(children) > 0, "Parent should have children"

        # Complete all children
        for child in children:
            gt.complete(child.id)

        # Parent progress should be 1.0 (average of all 1.0s)
        assert gt.goals[parent.id].progress == 1.0

    def test_complete_nonexistent_goal(self, empty_goal_tree):
        """Test completing a non-existent goal does not raise."""
        gt = empty_goal_tree
        # Should not raise
        gt.complete("nonexistent_goal")

    def test_block_goal(self, populated_goal_tree):
        """Test marking a goal as blocked."""
        gt = populated_goal_tree

        goal_id = list(gt.goals.keys())[0]
        reason = "Waiting for external dependency"

        gt.block(goal_id, reason)

        goal = gt.goals[goal_id]
        assert goal.status == GoalStatus.BLOCKED.value
        assert reason in goal.blockers

    def test_block_adds_multiple_reasons(self, populated_goal_tree):
        """Test blocking a goal multiple times adds multiple blockers."""
        gt = populated_goal_tree

        goal_id = list(gt.goals.keys())[0]
        gt.block(goal_id, "First blocker")
        gt.block(goal_id, "Second blocker")

        goal = gt.goals[goal_id]
        assert len(goal.blockers) == 2

    def test_decompose_goal(self, populated_goal_tree):
        """Test decomposing a goal into sub-goals."""
        gt = populated_goal_tree

        # Add a new goal to decompose
        parent_id = gt.add(
            title="Complex Task",
            description="A complex task needing decomposition"
        )

        sub_goals = [
            {"title": "Step 1", "description": "First step", "priority": 0.8},
            {"title": "Step 2", "description": "Second step", "priority": 0.6},
            {"title": "Step 3", "description": "Third step", "priority": 0.9}
        ]

        child_ids = gt.decompose(parent_id, sub_goals)

        assert len(child_ids) == 3
        for child_id in child_ids:
            assert child_id in gt.goals
            assert gt.goals[child_id].parent_id == parent_id

    def test_decompose_nonexistent_goal(self, empty_goal_tree):
        """Test decomposing a non-existent goal returns empty list."""
        gt = empty_goal_tree
        result = gt.decompose("nonexistent", [{"title": "Test"}])
        assert result == []

    def test_get_active(self, populated_goal_tree):
        """Test retrieving all active goals."""
        gt = populated_goal_tree

        active = gt.get_active()
        assert len(active) > 0
        for goal in active:
            assert goal.status == GoalStatus.ACTIVE.value

    def test_get_next_actions(self, populated_goal_tree):
        """Test retrieving next actionable leaf goals."""
        gt = populated_goal_tree

        actions = gt.get_next_actions(limit=3)

        # Should return leaf goals sorted by priority
        assert len(actions) <= 3
        for action in actions:
            assert action.is_leaf(gt.goals)
            assert action.status == GoalStatus.ACTIVE.value

        # Verify sorted by priority descending
        if len(actions) > 1:
            for i in range(len(actions) - 1):
                assert actions[i].priority >= actions[i + 1].priority


class TestGoalTreeIDUniqueness:
    """Tests for goal ID uniqueness."""

    def test_unique_ids_generated(self, empty_goal_tree):
        """Test that multiple goals get unique IDs."""
        gt = empty_goal_tree

        ids = set()
        for i in range(100):
            goal_id = gt.add(title=f"Goal {i}", description=f"Description {i}")
            assert goal_id not in ids, f"Duplicate ID generated: {goal_id}"
            ids.add(goal_id)

        assert len(ids) == 100

    def test_id_format(self, empty_goal_tree):
        """Test that goal IDs follow expected format."""
        gt = empty_goal_tree

        goal_id = gt.add(title="Test", description="Test")

        assert goal_id.startswith("goal_")
        # Should have 12 hex characters after prefix
        suffix = goal_id.replace("goal_", "")
        assert len(suffix) == 12
        assert all(c in "0123456789abcdef" for c in suffix)


class TestGoalTreePersistence:
    """Tests for GoalTree persistence."""

    def test_save_to_file(self, temp_goals_path):
        """Test that goals are saved to file."""
        gt = GoalTree(path=temp_goals_path)

        gt.add(title="Persisted Goal", description="This should persist")

        # Read file directly
        with open(temp_goals_path, 'r') as f:
            data = json.load(f)

        assert len(data) == 1
        saved_goal = list(data.values())[0]
        assert saved_goal["title"] == "Persisted Goal"

    def test_load_from_file(self, temp_goals_path):
        """Test that goals are loaded from file on initialization."""
        # Create initial tree and add goals
        gt1 = GoalTree(path=temp_goals_path)
        id1 = gt1.add(title="Goal 1", description="First")
        id2 = gt1.add(title="Goal 2", description="Second")

        # Create new tree with same path - should load existing goals
        gt2 = GoalTree(path=temp_goals_path)

        assert id1 in gt2.goals
        assert id2 in gt2.goals
        assert gt2.goals[id1].title == "Goal 1"
        assert gt2.goals[id2].title == "Goal 2"

    def test_persistence_roundtrip(self, temp_goals_path):
        """Test that goals persist correctly through save/load cycle."""
        gt1 = GoalTree(path=temp_goals_path)

        # Add complex goal structure
        root = gt1.add(
            title="Root Goal",
            description="Root",
            priority=0.9,
            deadline="2025-06-01T00:00:00Z",
            tags=["important"]
        )
        gt1.decompose(root, [
            {"title": "Sub 1", "description": "Sub goal 1", "priority": 0.8},
            {"title": "Sub 2", "description": "Sub goal 2", "priority": 0.7}
        ])

        # Load into new instance
        gt2 = GoalTree(path=temp_goals_path)

        assert len(gt2.goals) == 3
        root_goal = gt2.goals[root]
        assert root_goal.title == "Root Goal"
        assert root_goal.priority == 0.9
        assert "important" in root_goal.tags

        # Check children
        children = [g for g in gt2.goals.values() if g.parent_id == root]
        assert len(children) == 2

    def test_corrupt_file_handling(self, temp_goals_path):
        """Test that corrupt JSON file is handled gracefully."""
        # Write invalid JSON
        with open(temp_goals_path, 'w') as f:
            f.write("{ invalid json }")

        # Should not raise, should have empty goals
        gt = GoalTree(path=temp_goals_path)
        assert gt.goals == {}

    def test_empty_file_handling(self, temp_goals_path):
        """Test that empty file is handled gracefully."""
        # Write empty file
        with open(temp_goals_path, 'w') as f:
            f.write("")

        # Should not raise
        gt = GoalTree(path=temp_goals_path)
        assert gt.goals == {}

    def test_missing_file_handling(self, temp_goals_path):
        """Test that missing file is handled gracefully."""
        # Delete the file
        os.unlink(temp_goals_path)

        # Should not raise, should have empty goals
        gt = GoalTree(path=temp_goals_path)
        assert gt.goals == {}


class TestGoalTreeStats:
    """Tests for GoalTree.stats() method."""

    def test_stats_empty_tree(self, empty_goal_tree):
        """Test stats on empty tree."""
        gt = empty_goal_tree
        stats = gt.stats()

        assert stats["total_goals"] == 0
        assert stats["active"] == 0
        assert stats["completed"] == 0
        assert stats["blocked"] == 0
        assert stats["avg_progress"] == 0.0

    def test_stats_with_goals(self, populated_goal_tree):
        """Test stats returns correct counts."""
        gt = populated_goal_tree
        stats = gt.stats()

        assert stats["total_goals"] > 0
        assert stats["active"] > 0
        assert stats["completed"] == 0  # None completed yet

    def test_stats_after_completions(self, populated_goal_tree):
        """Test stats updates after completing goals."""
        gt = populated_goal_tree

        # Complete some goals
        goals = list(gt.goals.keys())
        gt.complete(goals[0])
        gt.complete(goals[1])

        stats = gt.stats()
        assert stats["completed"] == 2
        assert stats["active"] < stats["total_goals"]

    def test_stats_after_blocking(self, populated_goal_tree):
        """Test stats updates after blocking goals."""
        gt = populated_goal_tree

        # Block some goals
        goals = list(gt.goals.keys())
        gt.block(goals[0], "Blocked for testing")

        stats = gt.stats()
        assert stats["blocked"] == 1

    def test_avg_progress_calculation(self, populated_goal_tree):
        """Test that avg_progress is calculated correctly."""
        gt = populated_goal_tree

        # Complete one goal fully
        goals = list(gt.goals.keys())
        gt.complete(goals[0])

        stats = gt.stats()
        # Should have some progress
        assert stats["avg_progress"] > 0


class TestGoalTreeDecomposeCascading:
    """Tests for decompose cascading behavior."""

    def test_decompose_cascades_progress_updates(self, empty_goal_tree):
        """Test that completing decomposed goals updates parent progress."""
        gt = empty_goal_tree

        # Create parent
        parent_id = gt.add(title="Parent", description="Parent goal")

        # Decompose into 4 sub-goals
        child_ids = gt.decompose(parent_id, [
            {"title": "Child 1", "description": "C1"},
            {"title": "Child 2", "description": "C2"},
            {"title": "Child 3", "description": "C3"},
            {"title": "Child 4", "description": "C4"}
        ])

        # Complete 2 of 4 children (50%)
        gt.complete(child_ids[0])
        gt.complete(child_ids[1])

        # Parent should have 50% progress
        parent = gt.goals[parent_id]
        assert parent.progress == 0.5

    def test_multi_level_decompose(self, empty_goal_tree):
        """Test multi-level goal decomposition."""
        gt = empty_goal_tree

        # Level 0: Root
        root = gt.add(title="Root", description="Root goal")

        # Level 1: First level children
        level1_ids = gt.decompose(root, [
            {"title": "L1-A", "description": "Level 1 A"},
            {"title": "L1-B", "description": "Level 1 B"}
        ])

        # Level 2: Second level children
        level2_ids = gt.decompose(level1_ids[0], [
            {"title": "L2-A1", "description": "Level 2 A1"},
            {"title": "L2-A2", "description": "Level 2 A2"}
        ])

        # Verify structure
        assert len(gt.goals) == 5  # 1 root + 2 L1 + 2 L2

        # Complete leaf nodes
        for lid in level2_ids:
            gt.complete(lid)
        gt.complete(level1_ids[1])

        # L1-A should be complete (all children complete)
        l1_a = gt.goals[level1_ids[0]]
        assert l1_a.progress == 1.0


class TestGoalTreeGetTree:
    """Tests for GoalTree.get_tree() method."""

    def test_get_tree_root_level(self, populated_goal_tree):
        """Test getting tree structure from root level."""
        gt = populated_goal_tree

        tree = gt.get_tree()

        # Should return root goals
        assert len(tree) >= 1
        for node in tree:
            assert "id" in node
            assert "title" in node
            assert "status" in node
            assert "progress" in node
            assert "children" in node

    def test_get_tree_specific_root(self, populated_goal_tree):
        """Test getting tree from specific root goal."""
        gt = populated_goal_tree

        # Get first root goal
        roots = [g for g in gt.goals.values() if g.parent_id is None]
        root_id = roots[0].id

        tree = gt.get_tree(root_id=root_id)

        assert len(tree) == 1
        assert tree[0]["id"] == root_id

    def test_get_tree_nonexistent_root(self, empty_goal_tree):
        """Test getting tree with non-existent root returns empty."""
        gt = empty_goal_tree

        tree = gt.get_tree(root_id="nonexistent")
        assert tree == []

    def test_get_tree_includes_children(self, populated_goal_tree):
        """Test that tree includes nested children."""
        gt = populated_goal_tree

        tree = gt.get_tree()

        # Find a node with children
        def find_node_with_children(nodes):
            for node in nodes:
                if node["children"]:
                    return node
            return None

        node_with_children = find_node_with_children(tree)
        if node_with_children:
            assert len(node_with_children["children"]) > 0
            for child in node_with_children["children"]:
                assert "id" in child
                assert "title" in child


class TestGoalStatus:
    """Tests for GoalStatus enum."""

    def test_status_values(self):
        """Test GoalStatus has expected values."""
        assert GoalStatus.ACTIVE.value == "active"
        assert GoalStatus.COMPLETED.value == "completed"
        assert GoalStatus.BLOCKED.value == "blocked"
        assert GoalStatus.ABANDONED.value == "abandoned"

    def test_status_string_conversion(self):
        """Test GoalStatus can be converted to string."""
        assert str(GoalStatus.ACTIVE) == "active"
        assert GoalStatus.ACTIVE.value == "active"
