"""
MAGRPO Trainer - Main training loop for multi-agent LLM collaboration.

Implements the full training pipeline from the paper with:
- Episode collection
- Trajectory sampling
- MAGRPO policy updates
- Checkpointing
- Metrics tracking
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from orchestry.marl.algorithms.magrpo import MAGRPOOptimizer, Trajectory
from orchestry.marl.local_inference import LocalLLMAgent
from orchestry.marl.rewards.budget_tracker import BudgetTracker
from orchestry.tasks.code_collaboration import CodeCollaborationTask

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    episodes: int = 500
    group_size: int = 4  # G in the paper (samples per turn)
    batch_size: int = 8  # Update every N episodes
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 50
    eval_frequency: int = 10

    max_budget: float = 15.0  # USD


class MAGRPOTrainer:
    """
    MAGRPO trainer for multi-agent LLM collaboration.

    Usage:
        trainer = MAGRPOTrainer(agents, task, config)
        trainer.train()
        trainer.save_final_models("models/final")
    """

    def __init__(
        self,
        agents: list[LocalLLMAgent],
        task: CodeCollaborationTask,
        config: TrainingConfig,
        budget_tracker: BudgetTracker | None = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            agents: List of agent models [helper, main]
            task: Code collaboration task
            config: Training configuration
            budget_tracker: Budget tracker (if None, creates new)

        """
        self.agents = agents
        self.task = task
        self.config = config

        # Initialize MAGRPO optimizer
        self.optimizer = MAGRPOOptimizer(
            agents=agents,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            warmup_steps=config.warmup_steps,
        )

        # Budget tracking
        if budget_tracker is None:
            budget_tracker = BudgetTracker(max_budget=config.max_budget)
        self.budget_tracker = budget_tracker

        # Training state
        self.episode_buffer: list[Trajectory] = []
        self.metrics_history: list[dict[str, float]] = []
        self.current_episode = 0

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("MAGRPO Trainer initialized")
        logger.info(f"Episodes: {config.episodes}, Group size: {config.group_size}")
        logger.info(f"Batch size: {config.batch_size}, Learning rate: {config.learning_rate}")

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting MAGRPO training...")

        pbar = tqdm(range(self.config.episodes), desc="Training")

        for episode in pbar:
            self.current_episode = episode

            try:
                # Collect trajectory group
                trajectories = self._collect_trajectories(self.config.group_size)
                self.episode_buffer.extend(trajectories)

                # MAGRPO update every batch_size episodes
                if len(self.episode_buffer) >= self.config.batch_size:
                    update_metrics = self.optimizer.update(self.episode_buffer)
                    self.episode_buffer = []  # Clear buffer
                else:
                    update_metrics = {}

                # Compute episode metrics
                episode_metrics = self._compute_metrics(trajectories, update_metrics)
                self.metrics_history.append(episode_metrics)

                # Update progress bar
                pbar.set_postfix(
                    {
                        "reward": f"{episode_metrics['mean_reward']:.3f}",
                        "coop": f"{episode_metrics['cooperation_rate']:.2f}",
                        "budget": f"${self.budget_tracker.total_spent:.2f}",
                    },
                )

                # Evaluation
                if (episode + 1) % self.config.eval_frequency == 0:
                    self._log_metrics(episode, episode_metrics)

                # Checkpointing
                if (episode + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(episode)

            except Exception as e:
                logger.exception("Error in episode %d: %s", episode, e)
                if "Budget" in str(e):
                    logger.exception("Budget exceeded, stopping training")
                    break
                raise

        logger.info("Training complete!")
        self.budget_tracker.print_summary()

    def _collect_trajectories(self, k: int) -> list[Trajectory]:
        """
        Collect k trajectory samples.

        Args:
            k: Number of trajectories to collect (group size)

        Returns:
            trajectories: List of k trajectories

        """
        trajectories = []

        for _ in range(k):
            # Reset environment
            obs = self.task.reset()
            trajectory = Trajectory(turns=[], total_reward=0.0, reward_components={})

            # Turn 1: Helper agent (agent 0)
            helper_prompt = obs[0]
            helper_response = self.agents[0].generate(
                prompt=helper_prompt,
                temperature=0.8,
            )

            # Extract just the generated part (remove prompt)
            helper_code = helper_response[len(helper_prompt) :].strip()

            # Store turn with log prob computation delayed
            trajectory.turns.append(
                {
                    "agent_id": 0,
                    "observation": helper_prompt,
                    "action": helper_code,
                    "log_prob": None,  # Computed during update
                },
            )

            # Step environment
            result = self.task.step({0: helper_code})

            # Turn 2: Main agent (agent 1)
            main_prompt = result["observations"][1]
            main_response = self.agents[1].generate(
                prompt=main_prompt,
                temperature=0.8,
            )

            # Extract generated part
            main_code = main_response[len(main_prompt) :].strip()

            # Store turn
            trajectory.turns.append(
                {
                    "agent_id": 1,
                    "observation": main_prompt,
                    "action": main_code,
                    "log_prob": None,
                },
            )

            # Final step
            final_result = self.task.step({1: main_code})

            # Set rewards
            trajectory.total_reward = final_result["rewards"]["total"]
            trajectory.reward_components = final_result["rewards"]

            trajectories.append(trajectory)

        return trajectories

    def _compute_metrics(
        self,
        trajectories: list[Trajectory],
        update_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Compute metrics for logging."""
        rewards = [t.total_reward for t in trajectories]

        # Extract reward components
        structure_scores = [t.reward_components.get("structure", 0) for t in trajectories]
        syntax_scores = [t.reward_components.get("syntax", 0) for t in trajectories]
        test_scores = [t.reward_components.get("tests", 0) for t in trajectories]
        coop_scores = [t.reward_components.get("cooperation", 0) for t in trajectories]

        metrics = {
            "episode": self.current_episode,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "structure_rate": float(np.mean(structure_scores)) / 0.25,  # Normalize to 0-1
            "syntax_rate": float(np.mean(syntax_scores)) / 0.25,
            "test_pass_rate": float(np.mean(test_scores)) / 0.25,
            "cooperation_rate": float(np.mean(coop_scores)) / 0.25,
            "budget_spent": self.budget_tracker.total_spent,
        }

        # Add update metrics
        metrics.update(update_metrics)

        return metrics

    def _log_metrics(self, episode: int, metrics: dict[str, float]) -> None:
        """Log metrics."""
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Episode {episode}")
        logger.info(f"{'=' * 50}")
        logger.info(f"Mean Reward: {metrics['mean_reward']:.3f}")
        logger.info(f"Structure:   {metrics['structure_rate']:.2%}")
        logger.info(f"Syntax:      {metrics['syntax_rate']:.2%}")
        logger.info(f"Tests:       {metrics['test_pass_rate']:.2%}")
        logger.info(f"Cooperation: {metrics['cooperation_rate']:.2%}")
        logger.info(f"Budget:      ${metrics['budget_spent']:.2f}")
        logger.info(f"{'=' * 50}\n")

    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"episode_{episode}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save agent weights
        for i, agent in enumerate(self.agents):
            agent.save_lora_weights(str(checkpoint_path / f"agent_{i}"))

        # Save metrics
        with open(checkpoint_path / "metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Save config
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_final_models(self, output_dir: str) -> None:
        """Save final trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, agent in enumerate(self.agents):
            agent.save_lora_weights(str(output_path / f"agent_{i}"))

        # Save all metrics
        with open(output_path / "training_metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"Final models saved: {output_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_path)

        # Load agent weights
        for i, agent in enumerate(self.agents):
            agent.load_lora_weights(str(checkpoint_dir / f"agent_{i}"))

        # Load metrics
        with open(checkpoint_dir / "metrics.json") as f:
            self.metrics_history = json.load(f)

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def evaluate(
        self,
        test_problems: list[Any] | None = None,
        num_samples: int = 10,
    ) -> dict[str, float]:
        """
        Evaluate trained models.

        Args:
            test_problems: Test problems (if None, uses all task problems)
            num_samples: Number of samples per problem

        Returns:
            eval_metrics: Evaluation results

        """
        logger.info("Evaluating models...")

        if test_problems:
            original_problems = self.task.problems
            self.task.set_problems(test_problems)

        # Collect samples
        all_rewards = []
        all_coop_scores = []

        for _ in range(num_samples):
            trajectories = self._collect_trajectories(k=1)
            all_rewards.append(trajectories[0].total_reward)
            all_coop_scores.append(trajectories[0].reward_components.get("cooperation", 0))

        # Restore original problems
        if test_problems:
            self.task.set_problems(original_problems)

        eval_metrics = {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "cooperation_rate": float(np.mean(all_coop_scores)) / 0.25,
        }

        logger.info(
            f"Evaluation: Reward={eval_metrics['mean_reward']:.3f}, "
            f"Cooperation={eval_metrics['cooperation_rate']:.2%}",
        )

        return eval_metrics
