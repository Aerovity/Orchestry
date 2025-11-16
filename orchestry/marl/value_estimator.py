"""Centralized Value Estimator for MARL.

Uses Claude as a judge agent to evaluate multi-agent trajectories.
Implements centralized critic for decentralized actor training.
"""

import json
import logging
import time

import anthropic
from anthropic.types import TextBlock

from .trajectory import MultiTurnTrajectory

logger = logging.getLogger(__name__)


class CentralizedValueEstimator:
    """Centralized judge agent that evaluates multi-agent interactions.

    Uses Claude to score trajectories on multiple dimensions:
    - Quality: How good is the final output?
    - Collaboration: How well did agents work together?
    - Efficiency: Was the goal reached quickly?

    This implements the "centralized critic" in CTDE (Centralized Training,
    Decentralized Execution) MARL paradigm.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize value estimator.

        Args:
            api_key: Anthropic API key
            model: Claude model to use as judge
            weights: Reward component weights (quality, collaboration, efficiency)

        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Default weights
        self.weights = weights or {"quality": 0.4, "collaboration": 0.4, "efficiency": 0.2}

        # Cache for evaluated trajectories
        self._cache: dict[str, dict[str, float]] = {}

        logger.info(f"Initialized CentralizedValueEstimator with model {model}")

    def estimate_value(
        self,
        trajectory: MultiTurnTrajectory,
        task_type: str = "code_review",
    ) -> dict[str, float]:
        """Estimate the value of a trajectory.

        Args:
            trajectory: Completed or partial trajectory
            task_type: Type of task (affects evaluation criteria)

        Returns:
            Dictionary with scores:
            {
                'quality': 0-10,
                'collaboration': 0-10,
                'efficiency': 0-10,
                'total': weighted sum
            }

        """
        # Check cache
        traj_hash = trajectory.get_hash()
        if traj_hash in self._cache:
            logger.debug(f"Cache hit for trajectory {traj_hash[:8]}")
            return self._cache[traj_hash]

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(trajectory, task_type)

        # Call Claude as judge
        try:
            logger.debug(f"Evaluating trajectory with {len(trajectory)} turns")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,  # Deterministic evaluation
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            content_block = response.content[0]
            if isinstance(content_block, TextBlock):
                text_content = content_block.text
            else:
                text_content = str(content_block)
            scores = self._parse_evaluation_response(text_content)

            # Calculate total
            scores["total"] = (
                scores["quality"] * self.weights["quality"]
                + scores["collaboration"] * self.weights["collaboration"]
                + scores["efficiency"] * self.weights["efficiency"]
            )

            # Cache result
            self._cache[traj_hash] = scores

            logger.debug(
                f"Trajectory scored: quality={scores['quality']:.1f}, "
                f"collaboration={scores['collaboration']:.1f}, "
                f"efficiency={scores['efficiency']:.1f}, "
                f"total={scores['total']:.1f}",
            )

            return scores

        except Exception as e:
            logger.exception(f"Error evaluating trajectory: {e}")
            # Return neutral scores on error
            return {"quality": 5.0, "collaboration": 5.0, "efficiency": 5.0, "total": 5.0}

    def _build_evaluation_prompt(self, trajectory: MultiTurnTrajectory, task_type: str) -> str:
        """Build evaluation prompt for Claude judge.

        Args:
            trajectory: Trajectory to evaluate
            task_type: Type of task

        Returns:
            Evaluation prompt string

        """
        conversation = trajectory.get_full_conversation()

        # Task-specific criteria
        criteria_map = {
            "code_review": {
                "quality": "Is the final code correct, readable, and efficient? Does it handle edge cases?",
                "collaboration": "Did agents build on each other's contributions? Did they reference and improve previous work?",
                "efficiency": "Was the solution reached quickly without unnecessary back-and-forth?",
            },
            "documentation": {
                "quality": "Is the documentation clear, complete, and accurate? Are there good examples?",
                "collaboration": "Did agents complement each other's contributions effectively?",
                "efficiency": "Was the documentation completed efficiently?",
            },
            "story_writing": {
                "quality": "Is the story creative, coherent, and engaging? Does it have a clear structure?",
                "collaboration": 'Did agents use "yes, and" thinking? Did they build on each other\'s ideas?',
                "efficiency": "Was the story completed in a reasonable number of turns?",
            },
        }

        criteria = criteria_map.get(task_type, criteria_map["code_review"])

        return f"""You are evaluating a multi-agent collaboration on a {task_type} task.

{conversation}

Please evaluate this multi-agent conversation on the following criteria:

1. **Quality (0-10)**: {criteria["quality"]}

2. **Collaboration (0-10)**: {criteria["collaboration"]}

3. **Efficiency (0-10)**: {criteria["efficiency"]}

Provide your evaluation as a JSON object with exactly this structure:
{{
  "quality": <score 0-10>,
  "collaboration": <score 0-10>,
  "efficiency": <score 0-10>,
  "reasoning": {{
    "quality": "<brief explanation>",
    "collaboration": "<brief explanation>",
    "efficiency": "<brief explanation>"
  }}
}}

Be critical but fair. Use the full 0-10 range. Return ONLY the JSON object, no other text."""

    def _parse_evaluation_response(self, response_text: str) -> dict[str, float]:
        """Parse Claude's evaluation response.

        Args:
            response_text: Raw response from Claude

        Returns:
            Dictionary with numeric scores

        """
        try:
            # Extract JSON from response
            # Handle cases where Claude adds text before/after JSON
            start = response_text.find("{")
            end = response_text.rfind("}") + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_text = response_text[start:end]
            evaluation = json.loads(json_text)

            # Extract scores
            scores = {
                "quality": float(evaluation.get("quality", 5.0)),
                "collaboration": float(evaluation.get("collaboration", 5.0)),
                "efficiency": float(evaluation.get("efficiency", 5.0)),
            }

            # Clamp to 0-10 range
            for key in scores:
                scores[key] = max(0.0, min(10.0, scores[key]))

            return scores

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            logger.debug(f"Response was: {response_text[:200]}")

            # Return neutral scores
            return {"quality": 5.0, "collaboration": 5.0, "efficiency": 5.0}

    def compute_credit_assignment(
        self,
        trajectory: MultiTurnTrajectory,
        total_reward: float,
    ) -> list[float]:
        """Assign credit to individual agents using counterfactual reasoning.

        For each agent, ask: "What would the reward be without this agent's
        contributions?" The difference is that agent's credit.

        Args:
            trajectory: Completed trajectory
            total_reward: Total reward achieved

        Returns:
            List of rewards per agent (same length as number of agents)

        """
        if len(trajectory) == 0:
            return []

        # Get unique agents
        agent_ids = sorted({turn.agent_id for turn in trajectory.turns})
        num_agents = len(agent_ids)

        # Simple credit assignment: divide equally for now
        # TODO: Implement counterfactual reasoning in v2
        credit_per_agent = [total_reward / num_agents] * num_agents

        logger.debug(f"Credit assignment: {credit_per_agent}")

        return credit_per_agent

    def batch_evaluate(
        self,
        trajectories: list[MultiTurnTrajectory],
        task_type: str = "code_review",
        delay: float = 0.5,
    ) -> list[dict[str, float]]:
        """Evaluate multiple trajectories with rate limiting.

        Args:
            trajectories: List of trajectories to evaluate
            task_type: Type of task
            delay: Delay between API calls (seconds)

        Returns:
            List of score dictionaries

        """
        scores = []

        for i, trajectory in enumerate(trajectories):
            # Rate limiting
            if i > 0:
                time.sleep(delay)

            score = self.estimate_value(trajectory, task_type)
            scores.append(score)

        return scores

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
        logger.info("Evaluation cache cleared")

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_hits": sum(1 for _ in self._cache),  # TODO: Track actual hits
        }
