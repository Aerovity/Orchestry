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
        provider: str = "claude",
        gemini_api_key: str | None = None,
    ) -> None:
        """Initialize MARL trainer.

        Args:
            task: Task instance
            agents: List of agents
            api_key: API key (Anthropic for Claude, Google for Gemini)
            config: Configuration dictionary
            provider: "claude" or "gemini"
            gemini_api_key: Separate Gemini API key if needed

        """
        self.task = task
        self.agents = agents
        self.num_agents = len(agents)
        self.api_key = api_key
        self.provider = provider
        self.gemini_api_key = gemini_api_key

        # Default configuration
        self.config: dict[str, Any] = config or {}
        self.beam_width: int = self.config.get("beam_width", 10)
        self.k_samples: int = self.config.get("k_samples", 5)
        self.exploration_rate: float = self.config.get("exploration_rate", 0.1)
        self.learning_frequency: int = self.config.get("meta_learning", {}).get("update_frequency", 10)

        # Initialize components with provider support
        self.grpo = APIGroupRelativePolicyOptimizer(
            agents=agents,
            api_key=api_key,
            config=config,
            provider=provider,
            gemini_api_key=gemini_api_key
        )

        model_name: str = self.config.get("model", "claude-sonnet-4-20250514")
        # Note: Value estimator and behavior library still use Claude for now
        # Can be extended to support Gemini in future
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
            # Get task name from metadata or topic
            task_name = initial_obs.get('metadata', {}).get('problem_name')
            if not task_name:
                task_name = initial_obs.get('topic', task_description[:50] if task_description else 'Unknown')
            print(f"\nTask: {task_name}")
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

                    # Check if done based on trajectory length and research progress
                    # Don't rely on shared task state - each trajectory tracks its own progress
                    done = self._check_trajectory_complete(new_traj)
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
            # Convert trajectory to dict format for evaluation
            trajectory_dict = traj.to_dict()
            turns = trajectory_dict.get("turns", [])

            # Use LLM judge if configured, otherwise use heuristic evaluation
            if self.config.get("rewards", {}).get("use_llm_judge") and self.config.get("rewards", {}).get("llm_judge"):
                llm_judge = self.config["rewards"]["llm_judge"]
                # Extract research components from trajectory
                literature_reviewed = []
                hypotheses = []
                experiments = []
                analyses = []
                paper_draft = ""

                for turn in turns:
                    if turn.get("agent_role") == "literature_synthesizer":
                        literature_reviewed.append(turn.get("action", ""))
                    elif turn.get("agent_role") == "hypothesis_generator":
                        hypotheses.append(turn.get("action", ""))
                    elif turn.get("agent_role") == "experimental_designer":
                        experiments.append({"design": turn.get("action", "")})
                    elif turn.get("agent_role") == "data_analyst":
                        analyses.append(turn.get("action", ""))
                    elif turn.get("agent_role") == "paper_writer":
                        paper_draft += turn.get("action", "") + "\n\n"

                # Use LLM judge for evaluation
                task_scores = llm_judge.evaluate_research(
                    topic=initial_obs.get("topic", "Unknown"),
                    objective=initial_obs.get("objective", "Research objective"),
                    trajectory=turns,
                    literature_reviewed=literature_reviewed,
                    hypotheses=hypotheses,
                    experiments=experiments,
                    analyses=analyses,
                    paper_draft=paper_draft,
                )
            else:
                # Use task's heuristic evaluate method
                # First sync task state from trajectory
                self.task.literature_reviewed = []
                self.task.hypotheses_generated = []
                self.task.experiments_designed = []
                self.task.analyses_completed = []
                self.task.paper_draft = ""

                for turn in turns:
                    if turn.get("agent_role") == "literature_synthesizer":
                        self.task.literature_reviewed.append(turn.get("action", ""))
                    elif turn.get("agent_role") == "hypothesis_generator":
                        self.task.hypotheses_generated.append(turn.get("action", ""))
                    elif turn.get("agent_role") == "experimental_designer":
                        self.task.experiments_designed.append({"design": turn.get("action", "")})
                    elif turn.get("agent_role") == "data_analyst":
                        self.task.analyses_completed.append(turn.get("action", ""))
                    elif turn.get("agent_role") == "paper_writer":
                        self.task.paper_draft += turn.get("action", "") + "\n\n"

                task_scores = self.task.evaluate(turns)

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
            # Show actual research reward components
            comps = best_trajectory.reward_components
            if 'scientific_rigor' in comps:
                # Research lab rewards
                print(
                    f"Components: Rigor={comps.get('scientific_rigor', 0):.1f}, "
                    f"Novel={comps.get('novelty', 0):.1f}, "
                    f"Complete={comps.get('completeness', 0):.1f}, "
                    f"Collab={comps.get('collaboration', 0):.1f}, "
                    f"Feasible={comps.get('feasibility', 0):.1f}",
                )
            else:
                # Legacy code review rewards
                print(
                    f"Components: Q={comps.get('quality', 0):.1f}, "
                    f"C={comps.get('collaboration', 0):.1f}, "
                    f"E={comps.get('efficiency', 0):.1f}",
                )
            print(f"\nConversation ({len(best_trajectory)} turns):")
            print(best_trajectory.get_full_conversation())

        return best_trajectory, best_reward

    def _check_trajectory_complete(self, traj: MultiTurnTrajectory) -> bool:
        """Check if a trajectory is complete based on research phases.

        Args:
            traj: Trajectory to check

        Returns:
            True if trajectory has completed all research phases
        """
        # Must have minimum turns (all 5 agents contributed)
        if len(traj) < 5:
            return False

        # Check if all phases attempted by analyzing agent roles
        agent_roles_present = set()
        paper_length = 0

        for turn in traj.turns:
            agent_roles_present.add(turn.agent_role)
            # Track paper writer output
            if turn.agent_role == "paper_writer":
                paper_length += len(turn.action)

        # All 5 roles must have contributed
        required_roles = {"literature_synthesizer", "hypothesis_generator",
                         "experimental_designer", "data_analyst", "paper_writer"}
        all_phases_attempted = required_roles.issubset(agent_roles_present)

        # Paper writer should have produced substantial content
        substantial_paper = paper_length >= 3000

        # Complete if all phases done with substantial paper
        return all_phases_attempted and substantial_paper

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
            # Check if research or code review task
            if self.episodes and 'scientific_rigor' in self.episodes[0].reward_components:
                # Research lab CSV
                f.write("episode,total,scientific_rigor,novelty,completeness,collaboration,feasibility,turns\n")
                f.writelines(
                    f"{i},{ep.total_reward:.2f},"
                    f"{ep.reward_components.get('scientific_rigor', 0):.2f},"
                    f"{ep.reward_components.get('novelty', 0):.2f},"
                    f"{ep.reward_components.get('completeness', 0):.2f},"
                    f"{ep.reward_components.get('collaboration', 0):.2f},"
                    f"{ep.reward_components.get('feasibility', 0):.2f},"
                    f"{len(ep)}\n"
                    for i, ep in enumerate(self.episodes, 1)
                )
            else:
                # Legacy code review CSV
                f.write("episode,total,quality,collaboration,efficiency,turns\n")
                f.writelines(
                    f"{i},{ep.total_reward:.2f},"
                    f"{ep.reward_components.get('quality', 0):.2f},"
                    f"{ep.reward_components.get('collaboration', 0):.2f},"
                    f"{ep.reward_components.get('efficiency', 0):.2f},"
                    f"{len(ep)}\n"
                    for i, ep in enumerate(self.episodes, 1)
                )

        # Save research papers as markdown files
        if self.config.get("output", {}).get("save_papers", True):
            self._save_research_papers()

        # Save behavior library
        behaviors_path = self.save_dir / "learned_behaviors.json"
        self.behavior_library.save_to_file(str(behaviors_path))

        # Save summary
        summary_path = self.save_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self._get_training_summary(), f, indent=2)

        logger.info(f"Results saved to: {self.save_dir}")

    def _save_research_papers(self) -> None:
        """Save generated research papers as markdown files."""
        papers_dir = self.save_dir / "papers"
        papers_dir.mkdir(exist_ok=True)

        logger.info(f"Saving {len(self.episodes)} research papers to {papers_dir}")

        for i, episode in enumerate(self.episodes, 1):
            # Extract paper content from trajectory
            paper_content = self._extract_paper_from_episode(episode)

            if paper_content:
                # Save as markdown
                paper_path = papers_dir / f"episode_{i:03d}_paper.md"
                with open(paper_path, "w", encoding="utf-8") as f:
                    f.write(paper_content)
                logger.debug(f"Saved paper for episode {i}: {paper_path}")

        logger.info(f"Saved {len(self.episodes)} papers to {papers_dir}")

    def _extract_paper_from_episode(self, episode: MultiTurnTrajectory) -> str:
        """Extract research paper content from episode trajectory.

        Args:
            episode: Episode trajectory

        Returns:
            Formatted research paper as markdown string
        """
        # Extract all agent contributions
        literature = []
        hypotheses = []
        experiments = []
        analyses = []
        paper_drafts = []

        for turn in episode.turns:
            role = turn.agent_role
            action = turn.action

            if role == "literature_synthesizer":
                literature.append(action)
            elif role == "hypothesis_generator":
                hypotheses.append(action)
            elif role == "experimental_designer":
                experiments.append(action)
            elif role == "data_analyst":
                analyses.append(action)
            elif role == "paper_writer":
                paper_drafts.append(action)

        # Get topic from first turn's observation
        topic = "Research Study"
        objective = "Research objective"
        if episode.turns:
            obs = episode.turns[0].observation
            if isinstance(obs, dict):
                topic = obs.get("topic", topic)
                objective = obs.get("objective", objective)

        # Format as research paper
        paper = f"""# {topic}

## Objective
{objective}

## Reward Score: {episode.total_reward:.2f}
- Scientific Rigor: {episode.reward_components.get('scientific_rigor', 0):.1f}/10
- Novelty: {episode.reward_components.get('novelty', 0):.1f}/10
- Completeness: {episode.reward_components.get('completeness', 0):.1f}/10
- Collaboration: {episode.reward_components.get('collaboration', 0):.1f}/10
- Feasibility: {episode.reward_components.get('feasibility', 0):.1f}/10

---

## Literature Review

"""
        for lit in literature:
            paper += f"{lit}\n\n"

        paper += "## Hypotheses\n\n"
        for i, hyp in enumerate(hypotheses, 1):
            paper += f"### Hypothesis {i}\n{hyp}\n\n"

        paper += "## Experimental Design\n\n"
        for i, exp in enumerate(experiments, 1):
            paper += f"### Experiment {i}\n{exp}\n\n"

        paper += "## Data Analysis\n\n"
        for analysis in analyses:
            paper += f"{analysis}\n\n"

        paper += "## Paper Draft\n\n"
        for draft in paper_drafts:
            paper += f"{draft}\n\n"

        paper += f"""---

*Generated by Orchestry Multi-Agent Research Lab*
*Episode completed in {len(episode)} turns*
"""

        return paper

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
