"""
RL Environment for multi-agent collaboration.

Defines states, actions, and manages the episode lifecycle.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import random
import logging

from .agent import LLMAgent, Message

logger = logging.getLogger(__name__)


class EpisodeStatus(Enum):
    """Status of an episode."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_TURNS_REACHED = "max_turns_reached"


@dataclass
class State:
    """
    Represents the current state of the environment.

    State includes:
    - Conversation history
    - Current turn number
    - Task progress indicators
    """
    conversation_history: List[Message] = field(default_factory=list)
    current_turn: int = 0
    task_description: str = ""
    story_theme: str = ""
    is_terminal: bool = False
    status: EpisodeStatus = EpisodeStatus.IN_PROGRESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "current_turn": self.current_turn,
            "task_description": self.task_description,
            "story_theme": self.story_theme,
            "num_messages": len(self.conversation_history),
            "is_terminal": self.is_terminal,
            "status": self.status.value
        }


@dataclass
class Episode:
    """Stores complete episode data."""
    episode_id: int
    state_history: List[State] = field(default_factory=list)
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    rewards: Dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "conversation": self.conversation,
            "rewards": self.rewards,
            "total_reward": self.total_reward,
            "num_turns": len(self.conversation),
            "metadata": self.metadata
        }


class CollaborativeStoryEnvironment:
    """
    RL Environment for collaborative story writing.

    Manages agent interactions, state transitions, and episode lifecycle.
    """

    def __init__(
        self,
        agents: List[LLMAgent],
        max_turns: int = 15,
        story_target_length: int = 5,
        themes: Optional[List[str]] = None
    ):
        """
        Initialize the environment.

        Args:
            agents: List of LLM agents
            max_turns: Maximum turns per episode
            story_target_length: Minimum turns before completion
            themes: List of story themes to choose from
        """
        self.agents = agents
        self.max_turns = max_turns
        self.story_target_length = story_target_length
        self.themes = themes or ["A mysterious discovery", "An unexpected friendship"]

        self.current_state: Optional[State] = None
        self.current_episode: Optional[Episode] = None
        self.episode_count = 0

    def reset(self) -> State:
        """
        Reset the environment for a new episode.

        Returns:
            Initial state for the new episode
        """
        self.episode_count += 1

        # Choose a random theme
        theme = random.choice(self.themes)

        # Create task description
        task_description = f"Write a creative short story together about: {theme}"

        # Initialize state
        self.current_state = State(
            conversation_history=[],
            current_turn=0,
            task_description=task_description,
            story_theme=theme,
            is_terminal=False,
            status=EpisodeStatus.IN_PROGRESS
        )

        # Initialize episode
        self.current_episode = Episode(
            episode_id=self.episode_count,
            metadata={
                "theme": theme,
                "task": task_description,
                "num_agents": len(self.agents)
            }
        )

        # Reset agent memories
        for agent in self.agents:
            agent.reset_episode_memory()

        logger.info(f"Episode {self.episode_count} started with theme: {theme}")

        return self.current_state

    def step(self, agent_idx: int) -> Tuple[State, bool]:
        """
        Execute one step: one agent takes an action.

        Args:
            agent_idx: Index of the agent taking action

        Returns:
            Tuple of (new_state, done)
        """
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.current_state.is_terminal:
            logger.warning("Step called on terminal state")
            return self.current_state, True

        agent = self.agents[agent_idx]
        self.current_state.current_turn += 1
        turn = self.current_state.current_turn

        logger.debug(f"Turn {turn}: Agent {agent.role} acting")

        # Agent generates response
        response = agent.act(
            task_description=self.current_state.task_description,
            conversation_history=self.current_state.conversation_history,
            turn=turn,
            rate_limit_delay=1.0
        )

        # Create message
        message = Message(
            role="assistant",
            content=response,
            turn=turn,
            agent_role=agent.role
        )

        # Update state
        self.current_state.conversation_history.append(message)

        # Record in episode
        self.current_episode.conversation.append({
            "turn": turn,
            "agent": agent.role,
            "content": response
        })

        # Check for terminal conditions
        done = self._check_terminal()

        if done:
            self.current_state.is_terminal = True
            if self.current_state.current_turn >= self.max_turns:
                self.current_state.status = EpisodeStatus.MAX_TURNS_REACHED
            else:
                self.current_state.status = EpisodeStatus.COMPLETED

        return self.current_state, done

    def _check_terminal(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if episode is done
        """
        # Max turns reached
        if self.current_state.current_turn >= self.max_turns:
            logger.info(f"Episode ended: max turns ({self.max_turns}) reached")
            return True

        # Minimum length not met - continue
        if self.current_state.current_turn < self.story_target_length:
            return False

        # Check if story seems complete (simple heuristic)
        if len(self.current_state.conversation_history) >= self.story_target_length:
            # Look for ending indicators in last message
            last_message = self.current_state.conversation_history[-1].content.lower()
            ending_indicators = [
                "the end", "finally", "at last", "ever after",
                "concluded", "finished", "walked away", "faded",
                "never forgot", "always remembered"
            ]

            # Natural completion
            if any(indicator in last_message for indicator in ending_indicators):
                logger.info("Episode ended: natural story completion detected")
                return True

        # Continue if neither condition met
        return False

    def run_episode(self) -> Episode:
        """
        Run a complete episode with all agents.

        Returns:
            Completed episode data
        """
        state = self.reset()
        done = False

        # Rotate through agents
        agent_idx = 0

        while not done:
            state, done = self.step(agent_idx)

            # Move to next agent
            agent_idx = (agent_idx + 1) % len(self.agents)

        logger.info(
            f"Episode {self.episode_count} completed: "
            f"{self.current_state.current_turn} turns, "
            f"status: {self.current_state.status.value}"
        )

        return self.current_episode

    def get_conversation_text(self) -> str:
        """Get the full conversation as formatted text."""
        if not self.current_episode:
            return ""

        lines = [f"Theme: {self.current_episode.metadata['theme']}", ""]

        for turn_data in self.current_episode.conversation:
            lines.append(f"[{turn_data['agent']}]: {turn_data['content']}")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            "episode_count": self.episode_count,
            "max_turns": self.max_turns,
            "num_agents": len(self.agents),
            "current_turn": self.current_state.current_turn if self.current_state else 0
        }
