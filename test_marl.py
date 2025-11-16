#!/usr/bin/env python3
"""Quick test script for Orchestry MARL system.

Runs a minimal test to verify all components work together.
"""

import os
import sys

from dotenv import load_dotenv

# Load environment
load_dotenv()


def test_imports() -> bool:
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from orchestry.marl import (
            APIGroupRelativePolicyOptimizer,
            CentralizedValueEstimator,
            MARLTrainer,
            MultiTurnTrajectory,
        )
        from orchestry.tasks import BaseTask, CodeReviewTask

        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_api_key() -> bool:
    """Test that API key is set."""
    print("\nTesting API key...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("✗ ANTHROPIC_API_KEY not set properly")
        print("  Please set it in .env file")
        return False

    print("✓ API key found")
    return True


def test_trajectory() -> bool:
    """Test trajectory creation and manipulation."""
    print("\nTesting trajectory system...")

    from orchestry.marl import MultiTurnTrajectory

    try:
        traj = MultiTurnTrajectory(max_turns=10, task_description="Test task")

        traj.add_turn(
            agent_id=0,
            agent_role="Agent 0",
            observation="Test observation",
            action="Test action",
        )

        context = traj.get_context_for_agent(1)

        assert len(traj) == 1
        assert "Test action" in context

        print("✓ Trajectory system working")
        return True
    except Exception as e:
        print(f"✗ Trajectory test failed: {e}")
        return False


def test_task() -> bool:
    """Test task creation."""
    print("\nTesting task system...")

    from orchestry.tasks import CodeReviewTask, TaskConfig

    try:
        config = TaskConfig(max_turns=15, min_turns=3, task_type="code_review")

        task = CodeReviewTask(config)
        initial_obs = task.reset()

        assert "task_description" in initial_obs
        assert "Code Review Task" in initial_obs["task_description"]

        print("✓ Task system working")
        return True
    except Exception as e:
        print(f"✗ Task test failed: {e}")
        return False


def test_agent() -> bool:
    """Test agent creation."""
    print("\nTesting agent system...")

    from orchestry.marl.api_grpo import Agent

    try:
        agent = Agent(
            agent_id=0,
            role="Test Agent",
            goal="Test goal",
            system_prompt="Test prompt",
            learned_behaviors=[],
        )

        assert agent.role == "Test Agent"
        assert agent.agent_id == 0

        print("✓ Agent system working")
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("=" * 60)
    print("Orchestry MARL - Quick Test")
    print("=" * 60)

    tests = [
        test_imports,
        test_api_key,
        test_trajectory,
        test_task,
        test_agent,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed! System is ready.")
        print("\nRun a dry-run to test the full training loop:")
        print("  python main_marl.py --dry-run --verbose")
        return 0
    print("\n✗ Some tests failed. Please fix issues above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
