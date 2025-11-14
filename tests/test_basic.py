"""
Basic tests for Orchestry components.

Run with: python -m pytest tests/test_basic.py
Or simply: python tests/test_basic.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import LLMAgent, Message, AgentMemory
from src.environment import CollaborativeStoryEnvironment, State, EpisodeStatus
from src.rewards import RewardCalculator
from src.utils import load_config, calculate_improvement_rate


def test_agent_memory():
    """Test agent memory functionality."""
    print("Testing agent memory...")

    memory = AgentMemory()

    # Add messages
    msg1 = Message(role="assistant", content="Test content", turn=1, agent_role="Writer")
    memory.add_message(msg1)

    assert len(memory.messages) == 1
    assert memory.get_recent_context(1)[0] == msg1

    # Test learned behaviors
    memory.learned_behaviors.append("Build on ideas")
    assert len(memory.learned_behaviors) == 1

    # Test clear (keeps learned behaviors)
    memory.clear()
    assert len(memory.messages) == 0
    assert len(memory.learned_behaviors) == 1

    print("✓ Agent memory tests passed")


def test_state_creation():
    """Test environment state."""
    print("Testing environment state...")

    state = State(
        conversation_history=[],
        current_turn=0,
        task_description="Test task",
        story_theme="Test theme",
        is_terminal=False,
        status=EpisodeStatus.IN_PROGRESS
    )

    assert state.current_turn == 0
    assert not state.is_terminal
    assert state.status == EpisodeStatus.IN_PROGRESS

    state_dict = state.to_dict()
    assert state_dict["current_turn"] == 0
    assert state_dict["task_description"] == "Test task"

    print("✓ State tests passed")


def test_improvement_calculation():
    """Test improvement rate calculation."""
    print("Testing improvement calculation...")

    # No improvement
    rewards_flat = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    improvement = calculate_improvement_rate(rewards_flat, window=5)
    assert improvement == 0.0

    # Clear improvement
    rewards_improving = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
    improvement = calculate_improvement_rate(rewards_improving, window=5)
    assert improvement > 0

    # Declining
    rewards_declining = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5]
    improvement = calculate_improvement_rate(rewards_declining, window=5)
    assert improvement < 0

    print("✓ Improvement calculation tests passed")


def test_config_structure():
    """Test config file structure."""
    print("Testing config loading...")

    try:
        config = load_config("config.yaml")

        # Check required sections
        assert "api" in config
        assert "agents" in config
        assert "environment" in config
        assert "rewards" in config
        assert "training" in config

        # Check agents
        assert len(config["agents"]) >= 3
        for agent in config["agents"]:
            assert "role" in agent
            assert "goal" in agent

        # Check weights sum to 1
        weights = (
            config["rewards"]["story_quality_weight"] +
            config["rewards"]["collaboration_weight"] +
            config["rewards"]["efficiency_weight"]
        )
        assert abs(weights - 1.0) < 0.01  # Allow small floating point error

        print("✓ Config tests passed")

    except FileNotFoundError:
        print("⚠ Config file not found - skipping config tests")


def test_episode_tracking():
    """Test episode data structure."""
    print("Testing episode tracking...")

    from src.environment import Episode

    episode = Episode(
        episode_id=1,
        state_history=[],
        conversation=[
            {"turn": 1, "agent": "Writer", "content": "Once upon a time..."},
            {"turn": 2, "agent": "Editor", "content": "Let me refine that..."}
        ],
        rewards={"total": 7.5, "story_quality": 8.0},
        total_reward=7.5,
        metadata={"theme": "Test"}
    )

    episode_dict = episode.to_dict()
    assert episode_dict["episode_id"] == 1
    assert episode_dict["num_turns"] == 2
    assert episode_dict["total_reward"] == 7.5

    print("✓ Episode tracking tests passed")


def test_reward_weights():
    """Test reward weight configuration."""
    print("Testing reward weights...")

    config = load_config("config.yaml")

    total_weight = (
        config["rewards"]["story_quality_weight"] +
        config["rewards"]["collaboration_weight"] +
        config["rewards"]["efficiency_weight"]
    )

    assert 0.99 <= total_weight <= 1.01, f"Weights sum to {total_weight}, should be 1.0"

    print("✓ Reward weights tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Orchestry Basic Tests")
    print("=" * 60 + "\n")

    tests = [
        test_agent_memory,
        test_state_creation,
        test_improvement_calculation,
        test_config_structure,
        test_episode_tracking,
        test_reward_weights
    ]

    failed = []

    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed.append(test_func.__name__)
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed.append(test_func.__name__)

    print("\n" + "=" * 60)
    if not failed:
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
    print("=" * 60 + "\n")

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
