"""
Training loop for multi-agent RL system.

Manages episodes, learning, and progress tracking.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging
import random
from pathlib import Path
from datetime import datetime

from .agent import LLMAgent
from .environment import CollaborativeStoryEnvironment, Episode
from .rewards import RewardCalculator

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Stores training metrics across episodes."""
    episode_rewards: List[float] = field(default_factory=list)
    story_quality_scores: List[float] = field(default_factory=list)
    collaboration_scores: List[float] = field(default_factory=list)
    efficiency_scores: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    def add_episode(self, rewards: Dict[str, float], episode_length: int) -> None:
        """Add metrics from a completed episode."""
        self.episode_rewards.append(rewards["total"])
        self.story_quality_scores.append(rewards["story_quality"])
        self.collaboration_scores.append(rewards["collaboration"])
        self.efficiency_scores.append(rewards["efficiency"])
        self.episode_lengths.append(episode_length)

    def get_recent_average(self, n: int = 5) -> float:
        """Get average reward over last N episodes."""
        if not self.episode_rewards:
            return 0.0
        recent = self.episode_rewards[-n:]
        return sum(recent) / len(recent)

    def is_improving(self, window: int = 5) -> bool:
        """Check if performance is improving."""
        if len(self.episode_rewards) < window * 2:
            return False

        older_avg = sum(self.episode_rewards[-window*2:-window]) / window
        recent_avg = sum(self.episode_rewards[-window:]) / window

        return recent_avg > older_avg

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "episode_rewards": self.episode_rewards,
            "story_quality_scores": self.story_quality_scores,
            "collaboration_scores": self.collaboration_scores,
            "efficiency_scores": self.efficiency_scores,
            "episode_lengths": self.episode_lengths,
            "total_episodes": len(self.episode_rewards),
            "average_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0,
            "best_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "worst_reward": min(self.episode_rewards) if self.episode_rewards else 0
        }


class Trainer:
    """
    Main training loop for multi-agent RL.

    Manages episodes, learning updates, and progress tracking.
    """

    def __init__(
        self,
        agents: List[LLMAgent],
        environment: CollaborativeStoryEnvironment,
        reward_calculator: RewardCalculator,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.2,
        save_dir: str = "runs"
    ):
        """
        Initialize trainer.

        Args:
            agents: List of LLM agents
            environment: RL environment
            reward_calculator: Reward calculator
            learning_rate: Learning rate for prompt updates
            exploration_rate: Probability of exploration
            save_dir: Directory to save results
        """
        self.agents = agents
        self.environment = environment
        self.reward_calculator = reward_calculator
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate

        # Create save directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = Path(save_dir) / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.metrics = TrainingMetrics()

        # Episode storage
        self.episodes: List[Episode] = []

        logger.info(f"Trainer initialized. Save directory: {self.save_dir}")

    def run_episode(self, episode_num: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single training episode.

        Args:
            episode_num: Episode number
            verbose: Whether to log details

        Returns:
            Episode summary dictionary
        """
        logger.info(f"Starting episode {episode_num}")

        # Run episode
        episode = self.environment.run_episode()

        # Calculate rewards
        rewards = self.reward_calculator.calculate_rewards(episode)

        # Update episode with rewards
        episode.rewards = rewards
        episode.total_reward = rewards["total"]

        # Store episode
        self.episodes.append(episode)

        # Update metrics
        self.metrics.add_episode(rewards, len(episode.conversation))

        # Learning update
        if rewards["total"] > 6.0:  # Threshold for learning
            self._update_agents(episode, rewards)

        # Episode summary
        summary = {
            "episode": episode_num,
            "turns": len(episode.conversation),
            "rewards": rewards,
            "theme": episode.metadata.get("theme", "Unknown"),
            "recent_avg": self.metrics.get_recent_average()
        }

        if verbose:
            logger.info(
                f"Episode {episode_num} completed: "
                f"Reward={rewards['total']:.2f}, "
                f"Turns={len(episode.conversation)}, "
                f"Avg(5)={self.metrics.get_recent_average():.2f}"
            )

        return summary

    def train(
        self,
        num_episodes: int,
        save_frequency: int = 5,
        verbose: bool = True
    ) -> TrainingMetrics:
        """
        Run full training loop.

        Args:
            num_episodes: Number of episodes to run
            save_frequency: Save checkpoint every N episodes
            verbose: Whether to log details

        Returns:
            Training metrics
        """
        logger.info(f"Starting training for {num_episodes} episodes")

        for i in range(1, num_episodes + 1):
            # Run episode
            summary = self.run_episode(i, verbose=verbose)

            # Save checkpoint
            if i % save_frequency == 0:
                self._save_checkpoint(i)

        # Final save
        self._save_final_results()

        logger.info("Training completed")

        return self.metrics

    def _update_agents(self, episode: Episode, rewards: Dict[str, float]) -> None:
        """
        Update agent behaviors based on episode performance.

        Args:
            episode: Completed episode
            rewards: Calculated rewards
        """
        # Identify successful behaviors
        successful_behaviors = self.reward_calculator.identify_successful_behaviors(
            episode, rewards
        )

        # Update each agent
        for agent in self.agents:
            # Decide whether to explore or exploit
            if random.random() < self.exploration_rate:
                # Exploration: don't update, allow variation
                logger.debug(f"Agent {agent.role}: exploring (no update)")
                continue

            # Exploitation: update with successful behaviors
            agent.update_from_episode(
                episode_reward=rewards["total"],
                successful_behaviors=successful_behaviors,
                learning_rate=self.learning_rate
            )

            logger.debug(f"Agent {agent.role}: updated with {len(successful_behaviors)} behaviors")

    def _save_checkpoint(self, episode_num: int) -> None:
        """
        Save training checkpoint.

        Args:
            episode_num: Current episode number
        """
        checkpoint_path = self.save_dir / f"checkpoint_ep{episode_num}.json"

        checkpoint = {
            "episode": episode_num,
            "metrics": self.metrics.to_dict(),
            "agent_stats": [agent.get_stats() for agent in self.agents]
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_final_results(self) -> None:
        """Save final training results."""
        # Save all episodes
        episodes_path = self.save_dir / "episodes.json"
        episodes_data = [ep.to_dict() for ep in self.episodes]

        with open(episodes_path, "w") as f:
            json.dump(episodes_data, f, indent=2)

        # Save metrics
        metrics_path = self.save_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        # Save rewards CSV
        rewards_path = self.save_dir / "rewards.csv"
        with open(rewards_path, "w") as f:
            f.write("episode,total,story_quality,collaboration,efficiency,length\n")
            for i, (total, quality, collab, eff, length) in enumerate(zip(
                self.metrics.episode_rewards,
                self.metrics.story_quality_scores,
                self.metrics.collaboration_scores,
                self.metrics.efficiency_scores,
                self.metrics.episode_lengths
            ), 1):
                f.write(f"{i},{total},{quality},{collab},{eff},{length}\n")

        # Save agent stats
        agent_stats_path = self.save_dir / "agent_stats.json"
        agent_stats = [agent.get_stats() for agent in self.agents]

        with open(agent_stats_path, "w") as f:
            json.dump(agent_stats, f, indent=2)

        logger.info(f"Results saved to: {self.save_dir}")
        logger.info(f"  - {len(self.episodes)} episodes")
        logger.info(f"  - Metrics: {metrics_path}")
        logger.info(f"  - Rewards CSV: {rewards_path}")

    def get_best_episode(self) -> Optional[Episode]:
        """Get the episode with highest reward."""
        if not self.episodes:
            return None

        return max(self.episodes, key=lambda ep: ep.total_reward)

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        best_episode = self.get_best_episode()

        return {
            "total_episodes": len(self.episodes),
            "average_reward": self.metrics.get_recent_average(len(self.episodes)),
            "best_reward": best_episode.total_reward if best_episode else 0,
            "improving": self.metrics.is_improving(),
            "save_directory": str(self.save_dir),
            "metrics": self.metrics.to_dict()
        }
