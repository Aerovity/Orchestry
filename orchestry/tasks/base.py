"""Abstract base class for all multi-agent tasks.

Defines the interface that all tasks must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for a task."""

    max_turns: int = 15
    min_turns: int = 3
    task_type: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    """Abstract base class for multi-agent tasks.

    All tasks must implement:
    - reset(): Initialize new episode
    - step(): Execute one agent action
    - evaluate(): Calculate rewards
    - is_done(): Check if episode is complete
    """

    def __init__(self, config: TaskConfig) -> None:
        """Initialize task.

        Args:
            config: Task configuration

        """
        self.config = config
        self.current_turn = 0
        self.history: list[str] = []
        self.task_description = ""

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset environment for new episode.

        Returns:
            Dictionary with initial observation:
            {
                'task_description': str,
                'initial_context': str,
                'metadata': dict
            }

        """

    @abstractmethod
    def step(self, agent_id: int, agent_role: str, action: str) -> tuple[dict[str, Any], bool]:
        """Execute one agent action.

        Args:
            agent_id: ID of acting agent
            agent_role: Role of acting agent
            action: Agent's action (text response)

        Returns:
            Tuple of (observation, done):
            - observation: Dict with next observation
            - done: Boolean indicating if episode is complete

        """

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Calculate final rewards for completed episode.

        Returns:
            Dictionary of reward components:
            {
                'quality': float (0-10),
                'collaboration': float (0-10),
                'efficiency': float (0-10),
                'total': float (weighted sum)
            }

        """

    @abstractmethod
    def is_done(self) -> bool:
        """Check if episode should terminate.

        Returns:
            True if episode is complete

        """

    def get_current_turn(self) -> int:
        """Get current turn number."""
        return self.current_turn

    def get_history(self) -> list[str]:
        """Get conversation history."""
        return self.history.copy()

    def get_task_description(self) -> str:
        """Get task description."""
        return self.task_description

    def get_metadata(self) -> dict[str, Any]:
        """Get task metadata."""
        return self.config.metadata.copy()


class SimpleTask(BaseTask):
    """Simple implementation of BaseTask for testing.

    Can be used as a template for new tasks.
    """

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self.episode_complete = False

    def reset(self) -> dict[str, Any]:
        """Reset for new episode."""
        self.current_turn = 0
        self.history = []
        self.episode_complete = False

        self.task_description = "Collaborate to solve a simple task."

        return {
            "task_description": self.task_description,
            "initial_context": "Let's work together!",
            "metadata": self.config.metadata,
        }

    def step(self, _agent_id: int, agent_role: str, action: str) -> tuple[dict[str, Any], bool]:
        """Execute one step."""
        self.current_turn += 1
        self.history.append(f"{agent_role}: {action}")

        # Simple termination: after max_turns
        done = self.current_turn >= self.config.max_turns

        observation = {
            "turn": self.current_turn,
            "last_action": action,
            "history": self.get_history(),
        }

        return observation, done

    def evaluate(self) -> dict[str, float]:
        """Simple evaluation."""
        # Dummy scores
        return {"quality": 7.0, "collaboration": 7.0, "efficiency": 7.0, "total": 7.0}

    def is_done(self) -> bool:
        """Check if done."""
        return self.current_turn >= self.config.max_turns
