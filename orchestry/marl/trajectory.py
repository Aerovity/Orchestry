"""Multi-turn trajectory tracking for MARL episodes.

Handles multi-agent conversation history, observations, actions, and rewards.
"""

import copy
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Turn:
    """Represents a single agent turn in an episode."""

    agent_id: int
    agent_role: str
    observation: str  # What the agent sees (task + conversation history)
    action: str  # What the agent says/does
    turn_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MultiTurnTrajectory:
    """Tracks a complete multi-agent trajectory.

    Supports:
    - Adding turns sequentially
    - Cloning for beam search
    - Context extraction for agents
    - Reward tracking
    - Serialization
    """

    def __init__(self, max_turns: int = 15, task_description: str = "") -> None:
        """Initialize empty trajectory.

        Args:
            max_turns: Maximum number of turns before episode ends
            task_description: Description of the task for context

        """
        self.max_turns = max_turns
        self.task_description = task_description
        self.turns: list[Turn] = []
        self.done = False
        self.total_reward = 0.0
        self.reward_components: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}

    def add_turn(
        self,
        agent_id: int,
        agent_role: str,
        observation: str,
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a turn to the trajectory.

        Args:
            agent_id: ID of the agent taking action
            agent_role: Role of the agent (e.g., "Code Writer")
            observation: What the agent observed
            action: What the agent did
            metadata: Optional additional data

        """
        turn = Turn(
            agent_id=agent_id,
            agent_role=agent_role,
            observation=observation,
            action=action,
            turn_number=len(self.turns) + 1,
            metadata=metadata or {},
        )

        self.turns.append(turn)

        # Check termination conditions
        if len(self.turns) >= self.max_turns:
            self.done = True

    def get_context_for_agent(
        self,
        _agent_id: int,
        include_task: bool = True,
        max_history: int = 10,
    ) -> str:
        """Get conversation context for an agent.

        Args:
            _agent_id: ID of the agent requesting context (unused)
            include_task: Whether to include task description
            max_history: Maximum number of previous turns to include

        Returns:
            Formatted context string

        """
        context_parts = []

        # Add task description
        if include_task and self.task_description:
            context_parts.append(f"Task: {self.task_description}\n")

        # Add conversation history
        if self.turns:
            context_parts.append("Conversation so far:")
            recent_turns = (
                self.turns[-max_history:] if len(self.turns) > max_history else self.turns
            )

            for turn in recent_turns:
                context_parts.append(f"Turn {turn.turn_number} | {turn.agent_role}: {turn.action}")
        else:
            context_parts.append("(No conversation yet - you're going first)")

        return "\n".join(context_parts)

    def get_full_conversation(self) -> str:
        """Get the full conversation as a formatted string."""
        if not self.turns:
            return "(Empty trajectory)"

        lines = [f"Task: {self.task_description}\n"]
        for turn in self.turns:
            lines.append(f"Turn {turn.turn_number} | {turn.agent_role}:")
            lines.append(f"{turn.action}\n")

        return "\n".join(lines)

    def clone(self) -> "MultiTurnTrajectory":
        """Create a deep copy of this trajectory.

        Used for beam search - fork trajectory to explore different paths.

        Returns:
            Independent copy of this trajectory

        """
        new_traj = MultiTurnTrajectory(
            max_turns=self.max_turns,
            task_description=self.task_description,
        )
        new_traj.turns = copy.deepcopy(self.turns)
        new_traj.done = self.done
        new_traj.total_reward = self.total_reward
        new_traj.reward_components = copy.deepcopy(self.reward_components)
        new_traj.metadata = copy.deepcopy(self.metadata)

        return new_traj

    def set_rewards(self, total_reward: float, reward_components: dict[str, float]) -> None:
        """Set the final rewards for this trajectory.

        Args:
            total_reward: Total combined reward
            reward_components: Breakdown by component (quality, collaboration, etc.)

        """
        self.total_reward = total_reward
        self.reward_components = reward_components

    def get_hash(self) -> str:
        """Get a hash of this trajectory for caching/comparison.

        Returns:
            MD5 hash of the conversation content

        """
        conversation = self.get_full_conversation()
        return hashlib.md5(conversation.encode()).hexdigest()

    def __len__(self) -> int:
        """Return number of turns in trajectory."""
        return len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        """Convert trajectory to dictionary for serialization.

        Returns:
            Dictionary representation

        """
        return {
            "task_description": self.task_description,
            "max_turns": self.max_turns,
            "turns": [turn.to_dict() for turn in self.turns],
            "num_turns": len(self.turns),
            "done": self.done,
            "total_reward": self.total_reward,
            "reward_components": self.reward_components,
            "metadata": self.metadata,
            "conversation": self.get_full_conversation(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiTurnTrajectory":
        """Reconstruct trajectory from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed trajectory

        """
        traj = cls(max_turns=data["max_turns"], task_description=data["task_description"])

        for turn_data in data["turns"]:
            traj.add_turn(
                agent_id=turn_data["agent_id"],
                agent_role=turn_data["agent_role"],
                observation=turn_data["observation"],
                action=turn_data["action"],
                metadata=turn_data.get("metadata", {}),
            )

        traj.done = data["done"]
        traj.total_reward = data["total_reward"]
        traj.reward_components = data["reward_components"]
        traj.metadata = data["metadata"]

        return traj


class TrajectoryBeam:
    """Manages a beam of trajectories for beam search.

    Maintains top-N trajectories, prunes low-scoring ones.
    """

    def __init__(self, beam_width: int = 10) -> None:
        """Initialize beam.

        Args:
            beam_width: Maximum number of trajectories to keep

        """
        self.beam_width = beam_width
        self.trajectories: list[MultiTurnTrajectory] = []
        self.scores: list[float] = []

    def add(self, trajectory: MultiTurnTrajectory, score: float) -> None:
        """Add trajectory to beam.

        Args:
            trajectory: Trajectory to add
            score: Current score of trajectory

        """
        self.trajectories.append(trajectory)
        self.scores.append(score)

    def prune(self) -> None:
        """Prune beam to keep only top-N trajectories.

        Keeps trajectories with highest scores.
        """
        if len(self.trajectories) <= self.beam_width:
            return

        # Sort by score (descending)
        sorted_pairs = sorted(
            zip(self.scores, self.trajectories, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )

        # Keep top beam_width
        self.scores = [score for score, _ in sorted_pairs[: self.beam_width]]
        self.trajectories = [traj for _, traj in sorted_pairs[: self.beam_width]]

    def get_trajectories(self) -> list[MultiTurnTrajectory]:
        """Get all trajectories in beam."""
        return self.trajectories

    def get_best(self) -> MultiTurnTrajectory | None:
        """Get highest-scoring trajectory."""
        if not self.trajectories:
            return None

        best_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
        return self.trajectories[best_idx]

    def __len__(self) -> int:
        """Return number of trajectories in beam."""
        return len(self.trajectories)

    def is_empty(self) -> bool:
        """Check if beam is empty."""
        return len(self.trajectories) == 0
