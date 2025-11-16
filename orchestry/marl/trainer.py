"""MARL Trainer with beam search trajectory optimization.

Main training loop that orchestrates:
- Multi-sample response generation
- Beam search through trajectory space
- Centralized value estimation
- Behavior pattern learning
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from orchestry.tasks.base import BaseTask

from .api_grpo import Agent, APIGroupRelativePolicyOptimizer
from .behavior_library import BehaviorLibrary
from .trajectory import MultiTurnTrajectory, TrajectoryBeam
from .value_estimator import CentralizedValueEstimator

logger = logging.getLogger(__name__)


class MARLTrainer:
    """Multi-Agent RL training loop with beam search.

    Implements API-based MARL:
    1. For each turn, generate k response samples per agent
    2. Use beam search to explore top-N trajectories
    3. Score completed trajectories with centralized value estimator
    4. Select best using group-relative advantages
    5. Extract successful behaviors and update agents
    """

    def __init__(
        self,
        task: BaseTask,
        agents: list[Agent],
        api_key: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize MARL trainer.

        Args:
            task: Task instance
            agents: List of agents
            api_key: Anthropic API key
            config: Configuration dictionary

        """
        self.task = task
        self.agents = agents
        self.num_agents = len(agents)
        self.api_key = api_key

        # Default configuration
        self.config: dict[str, Any] = config or {}
        self.beam_width: int = self.config.get("beam_width", 10)
        self.k_samples: int = self.config.get("k_samples", 5)
        self.exploration_rate: float = self.config.get("exploration_rate", 0.1)
        self.learning_frequency: int = self.config.get("learning_frequency", 5)

        # Initialize components
        self.grpo = APIGroupRelativePolicyOptimizer(agents=agents, api_key=api_key, config=config)

        model_name: str = self.config.get("model", "claude-sonnet-4-20250514")
        self.value_estimator = CentralizedValueEstimator(api_key=api_key, model=model_name)

        self.behavior_library = BehaviorLibrary(api_key=api_key, model=model_name)

        # Episode history
        self.episodes: list[MultiTurnTrajectory] = []

        # Create save directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir_str: str = self.config.get("save_dir", "runs")
        self.save_dir = Path(save_dir_str) / f"marl_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized MARLTrainer with {self.num_agents} agents")
        logger.info(f"Beam width: {self.beam_width}, k_samples: {self.k_samples}")
        logger.info(f"Save directory: {self.save_dir}")

    def train(
        self,
        num_episodes: int,
        verbose: bool = True,
        save_frequency: int = 5,
    ) -> dict[str, Any]:
        """Run full training loop.

        Args:
            num_episodes: Number of episodes to run
            verbose: Whether to print detailed progress
            save_frequency: Save checkpoint every N episodes

        Returns:
            Training summary dictionary

        """
        logger.info(f"Starting training for {num_episodes} episodes")

        # Progress bar
        pbar = tqdm(range(num_episodes), desc="Training", disable=not verbose)

        for episode_num in pbar:
            # Run episode with beam search
            trajectory, reward = self.run_episode_with_beam_search(episode_num + 1, verbose=verbose)

            # Store episode
            self.episodes.append(trajectory)

            # Update progress bar
            avg_reward = np.mean([ep.total_reward for ep in self.episodes[-10:]])
            pbar.set_postfix({"reward": f"{reward:.2f}", "avg_10": f"{avg_reward:.2f}"})

            # Periodic learning updates
            if (episode_num + 1) % self.learning_frequency == 0:
                self._update_agent_behaviors(verbose=verbose)

            # Save checkpoint
            if (episode_num + 1) % save_frequency == 0:
                self._save_checkpoint(episode_num + 1)

        # Final save
        self._save_final_results()

        logger.info("Training completed!")

        return self._get_training_summary()

    def run_episode_with_beam_search(
        self,
        episode_num: int,
        verbose: bool = False,
    ) -> tuple[MultiTurnTrajectory, float]:
        """Run one episode using beam search over trajectory space.

        Algorithm:
        1. Start with empty trajectory
        2. For each turn:
           a. For each trajectory in beam, generate k samples
           b. Score new partial trajectories
           c. Prune to keep top beam_width trajectories
        3. When episodes complete, score all final trajectories
        4. Select best using group-relative advantages

        Args:
            episode_num: Episode number
            verbose: Whether to print details

        Returns:
            (best_trajectory, reward)

        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Episode {episode_num}")
            print(f"{'=' * 60}")

        # Reset task
        initial_obs = self.task.reset()
        task_description = initial_obs["task_description"]

        # Initialize beam with single empty trajectory
        beam = TrajectoryBeam(beam_width=self.beam_width)
        initial_traj = MultiTurnTrajectory(
            max_turns=self.task.config.max_turns,
            task_description=task_description,
        )
        beam.add(initial_traj, score=0.0)

        turn = 0
        max_turns = self.task.config.max_turns

        if verbose:
            print(f"\nTask: {initial_obs.get('metadata', {}).get('problem_name', 'Unknown')}")
            print("Starting beam search...")

        # Beam search loop
        while turn < max_turns:
            agent_id = turn % self.num_agents
            agent = self.agents[agent_id]

            if verbose:
                print(f"\nTurn {turn + 1} | Agent: {agent.role}")

            new_beam = TrajectoryBeam(beam_width=self.beam_width)

            # For each trajectory in current beam
            for _traj_idx, traj in enumerate(beam.get_trajectories()):
                # Skip if trajectory is already done
                if traj.done:
                    new_beam.add(traj, score=traj.total_reward)
                    continue

                # Get context for agent
                context = traj.get_context_for_agent(agent.agent_id)

                # Generate k response samples
                samples = self.grpo.generate_response_samples(
                    agent=agent,
                    context=context,
                    k=self.k_samples,
                )

                # Create k new trajectories
                for sample in samples:
                    new_traj = traj.clone()

                    # Add turn to trajectory
                    new_traj.add_turn(
                        agent_id=agent.agent_id,
                        agent_role=agent.role,
                        observation=context,
                        action=sample,
                    )

                    # Update task state
                    _, done = self.task.step(agent.agent_id, agent.role, sample)
                    if done:
                        new_traj.done = True

                    # Score trajectory (simple heuristic for now)
                    # Full evaluation only at episode end
                    score = len(new_traj)  # Placeholder score

                    new_beam.add(new_traj, score=score)

            # Prune beam
            new_beam.prune()
            beam = new_beam

            if verbose:
                print(f"Generated {self.k_samples * len(beam.get_trajectories())} samples")
                print(f"Kept top {len(beam)} trajectories")

            # Check if all trajectories are done
            if all(traj.done for traj in beam.get_trajectories()):
                if verbose:
                    print("\nAll trajectories complete!")
                break

            turn += 1

        # Final evaluation of completed trajectories
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Evaluating {len(beam)} final trajectories...")
            print(f"{'=' * 60}")

        final_trajectories = beam.get_trajectories()
        final_scores = []

        for traj in final_trajectories:
            # Use task's evaluate method
            task_scores = self.task.evaluate()
            total_score = task_scores["total"]

            traj.set_rewards(total_score, task_scores)
            final_scores.append(total_score)

        # Compute group-relative advantages
        advantages = self.grpo.compute_advantages(final_scores)

        # Select best trajectory
        best_idx = self.grpo.select_best_trajectory(
            advantages,
            exploration_rate=self.exploration_rate,
        )

        best_trajectory = final_trajectories[best_idx]
        best_reward = final_scores[best_idx]

        if verbose:
            print(f"\nSelected trajectory {best_idx}")
            print(f"Reward: {best_reward:.2f}")
            print(
                f"Components: Q={best_trajectory.reward_components.get('quality', 0):.1f}, "
                f"C={best_trajectory.reward_components.get('collaboration', 0):.1f}, "
                f"E={best_trajectory.reward_components.get('efficiency', 0):.1f}",
            )
            print(f"\nConversation ({len(best_trajectory)} turns):")
            print(best_trajectory.get_full_conversation())

        return best_trajectory, best_reward

    def _update_agent_behaviors(self, verbose: bool = False) -> None:
        """Extract successful behaviors and update agents.

        Args:
            verbose: Whether to print details

        """
        if len(self.episodes) < 5:
            logger.info("Not enough episodes for behavior extraction")
            return

        if verbose:
            print(f"\n{'=' * 60}")
            print("Updating agent behaviors...")
            print(f"{'=' * 60}")

        # Extract behaviors from recent high-reward episodes
        agent_roles = [agent.role for agent in self.agents]

        behaviors = self.behavior_library.extract_successful_behaviors(
            episodes=self.episodes,
            _num_agents=self.num_agents,
            agent_roles=agent_roles,
            task_type=self.task.config.task_type,
        )

        # Update each agent
        for agent in self.agents:
            if agent.role in behaviors:
                # Flatten all categories into single list
                new_behaviors = []
                for behavior_list in behaviors[agent.role].values():
                    new_behaviors.extend(behavior_list)

                self.grpo.update_agent_behaviors(agent=agent, new_behaviors=new_behaviors)

        if verbose:
            print(f"\nUpdated {len(behaviors)} agents with new behaviors")
            print(self.behavior_library.summary())

    def _save_checkpoint(self, episode_num: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_ep{episode_num}.json"

        checkpoint = {
            "episode": episode_num,
            "num_episodes": len(self.episodes),
            "recent_rewards": [ep.total_reward for ep in self.episodes[-10:]],
            "avg_reward": np.mean([ep.total_reward for ep in self.episodes]),
            "best_reward": max([ep.total_reward for ep in self.episodes]),
            "agent_behaviors": {agent.role: len(agent.learned_behaviors) for agent in self.agents},
            "cache_stats": self.grpo.get_cache_stats(),
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

        # Save rewards CSV
        rewards_path = self.save_dir / "rewards.csv"
        with open(rewards_path, "w") as f:
            f.write("episode,total,quality,collaboration,efficiency,turns\n")
            f.writelines(
                f"{i},{ep.total_reward:.2f},"
                f"{ep.reward_components.get('quality', 0):.2f},"
                f"{ep.reward_components.get('collaboration', 0):.2f},"
                f"{ep.reward_components.get('efficiency', 0):.2f},"
                f"{len(ep)}\n"
                for i, ep in enumerate(self.episodes, 1)
            )

        # Save behavior library
        behaviors_path = self.save_dir / "learned_behaviors.json"
        self.behavior_library.save_to_file(str(behaviors_path))

        # Save summary
        summary_path = self.save_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self._get_training_summary(), f, indent=2)

        logger.info(f"Results saved to: {self.save_dir}")

    def _get_training_summary(self) -> dict[str, Any]:
        """Get training summary statistics."""
        if not self.episodes:
            return {}

        rewards = [ep.total_reward for ep in self.episodes]

        return {
            "total_episodes": len(self.episodes),
            "average_reward": float(np.mean(rewards)),
            "best_reward": float(np.max(rewards)),
            "worst_reward": float(np.min(rewards)),
            "final_10_avg": (
                float(np.mean(rewards[-10:])) if len(rewards) >= 10 else float(np.mean(rewards))
            ),
            "reward_std": float(np.std(rewards)),
            "save_directory": str(self.save_dir),
            "cache_stats": self.grpo.get_cache_stats(),
            "agent_behaviors": {agent.role: len(agent.learned_behaviors) for agent in self.agents},
        }

    def get_best_episode(self) -> MultiTurnTrajectory | None:
        """Get episode with highest reward."""
        if not self.episodes:
            return None

        return max(self.episodes, key=lambda ep: ep.total_reward)
