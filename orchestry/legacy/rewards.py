"""Reward calculation system for Orchestry.

Rewards agents based on story quality, collaboration, and efficiency.
"""

import logging
import re

import anthropic

from .environment import Episode

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Calculates rewards for collaborative story writing episodes.

    Reward components:
    1. Story Quality: Overall quality of the final story
    2. Collaboration Quality: How well agents worked together
    3. Efficiency: Completing the task in reasonable time
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        story_quality_weight: float = 0.4,
        collaboration_weight: float = 0.4,
        efficiency_weight: float = 0.2,
        efficiency_threshold_good: int = 10,
        efficiency_threshold_bad: int = 20,
    ) -> None:
        """Initialize reward calculator.

        Args:
            api_key: Anthropic API key for judge agent
            model: Model to use for evaluation
            story_quality_weight: Weight for story quality (0-1)
            collaboration_weight: Weight for collaboration (0-1)
            efficiency_weight: Weight for efficiency (0-1)
            efficiency_threshold_good: Turns for efficiency bonus
            efficiency_threshold_bad: Turns for efficiency penalty

        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        self.story_quality_weight = story_quality_weight
        self.collaboration_weight = collaboration_weight
        self.efficiency_weight = efficiency_weight
        self.efficiency_threshold_good = efficiency_threshold_good
        self.efficiency_threshold_bad = efficiency_threshold_bad

    def calculate_rewards(self, episode: Episode) -> dict[str, float]:
        """Calculate all reward components for an episode.

        Args:
            episode: Completed episode

        Returns:
            Dictionary with reward components and total

        """
        logger.info(f"Calculating rewards for episode {episode.episode_id}")

        # Calculate individual components
        story_quality = self._evaluate_story_quality(episode)
        collaboration_score = self._evaluate_collaboration(episode)
        efficiency_score = self._calculate_efficiency(episode)

        # Weighted total
        total_reward = (
            story_quality * self.story_quality_weight
            + collaboration_score * self.collaboration_weight
            + efficiency_score * self.efficiency_weight
        )

        rewards = {
            "story_quality": round(story_quality, 2),
            "collaboration": round(collaboration_score, 2),
            "efficiency": round(efficiency_score, 2),
            "total": round(total_reward, 2),
        }

        logger.info(f"Rewards: {rewards}")

        return rewards

    def _evaluate_story_quality(self, episode: Episode) -> float:
        """Use a judge LLM to evaluate story quality.

        Args:
            episode: Episode to evaluate

        Returns:
            Score from 0-10

        """
        # Build the full story
        story_parts = [turn["content"] for turn in episode.conversation]
        full_story = "\n\n".join(story_parts)

        # Judge prompt
        judge_prompt = f"""Evaluate this collaboratively written story on a scale of 0-10.

Story Theme: {episode.metadata.get("theme", "Unknown")}

Story:
{full_story}

Rate the story considering:
1. Creativity and originality (is it interesting?)
2. Coherence (does it make sense and flow well?)
3. Completeness (does it feel like a complete story?)
4. Engagement (would readers enjoy it?)

Provide ONLY a numeric score from 0-10 (can use decimals like 7.5).
Format your response as: SCORE: X.X"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0.3,  # Lower temperature for consistent evaluation
                messages=[{"role": "user", "content": judge_prompt}],
            )

            content = response.content[0].text

            # Extract score
            score = self._extract_score(content)

            logger.debug(f"Story quality score: {score}")

            return score

        except Exception as e:
            logger.exception(f"Error evaluating story quality: {e}")
            return 5.0  # Default middle score

    def _evaluate_collaboration(self, episode: Episode) -> float:
        """Evaluate how well agents collaborated.

        Args:
            episode: Episode to evaluate

        Returns:
            Score from 0-10

        """
        # Build conversation with agent labels
        conversation = []
        for turn in episode.conversation:
            conversation.append(f"[{turn['agent']}]: {turn['content']}")

        full_conversation = "\n\n".join(conversation)

        # Judge prompt
        judge_prompt = f"""Evaluate how well these agents collaborated on writing a story together.

Conversation:
{full_conversation}

Rate the collaboration quality (0-10) based on:
1. Building on ideas: Did agents use "yes, and" thinking? Did they expand on each other's contributions?
2. Division of work: Did agents contribute fairly and play to their roles?
3. Coherence: Did they maintain consistency and connect their ideas?
4. Supportiveness: Did they support each other's creative choices?
5. Constructiveness: Were disagreements (if any) handled constructively?

Look for signs of:
✓ Good collaboration: References to previous contributions, expanding on ideas, smooth transitions
✗ Poor collaboration: Ignoring what came before, contradicting unnecessarily, working in isolation

Provide ONLY a numeric score from 0-10 (can use decimals like 8.5).
Format your response as: SCORE: X.X"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0.3,
                messages=[{"role": "user", "content": judge_prompt}],
            )

            content = response.content[0].text
            score = self._extract_score(content)

            logger.debug(f"Collaboration score: {score}")

            # Analyze patterns for learning
            self._extract_collaboration_patterns(episode)

            return score

        except Exception as e:
            logger.exception(f"Error evaluating collaboration: {e}")
            return 5.0

    def _calculate_efficiency(self, episode: Episode) -> float:
        """Calculate efficiency score based on number of turns.

        Args:
            episode: Episode to evaluate

        Returns:
            Score from 0-10

        """
        num_turns = len(episode.conversation)

        if num_turns <= self.efficiency_threshold_good:
            # Very efficient - bonus
            score = 10.0
        elif num_turns >= self.efficiency_threshold_bad:
            # Too long - penalty
            score = 3.0
        else:
            # Linear interpolation between thresholds
            range_size = self.efficiency_threshold_bad - self.efficiency_threshold_good
            turns_over_good = num_turns - self.efficiency_threshold_good
            score = 10.0 - (turns_over_good / range_size) * 7.0

        logger.debug(f"Efficiency score for {num_turns} turns: {score}")

        return max(0.0, min(10.0, score))

    def _extract_score(self, text: str) -> float:
        """Extract numeric score from judge response.

        Args:
            text: Judge response text

        Returns:
            Extracted score (0-10)

        """
        # Look for "SCORE: X.X" pattern
        pattern = r"SCORE:\s*(\d+\.?\d*)"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))

        # Fallback: look for any number between 0-10
        numbers = re.findall(r"\b(\d+\.?\d*)\b", text)
        for num in numbers:
            score = float(num)
            if 0 <= score <= 10:
                return score

        logger.warning(f"Could not extract score from: {text}")
        return 5.0  # Default

    def _extract_collaboration_patterns(self, episode: Episode) -> list[str]:
        """Extract successful collaboration patterns for learning.

        Args:
            episode: Episode to analyze

        Returns:
            List of pattern descriptions

        """
        patterns = []
        conversation = episode.conversation

        # Check for reference to previous ideas
        for i, turn in enumerate(conversation[1:], 1):
            content_lower = turn["content"].lower()

            # Check for building on previous content
            if any(
                word in content_lower for word in ["building on", "expanding", "continuing", "yes,"]
            ):
                patterns.append(f"{turn['agent']} explicitly built on previous ideas")

            # Check for references
            if i > 0:
                prev_content = conversation[i - 1]["content"].lower()
                # Simple heuristic: check if current mentions concepts from previous
                prev_words = set(prev_content.split())
                curr_words = set(content_lower.split())
                overlap = prev_words & curr_words

                if len(overlap) > 5:  # Significant overlap
                    patterns.append(f"{turn['agent']} maintained continuity with previous turn")

        return patterns

    def identify_successful_behaviors(
        self,
        episode: Episode,
        rewards: dict[str, float],
    ) -> list[str]:
        """Identify specific successful behaviors for learning.

        Args:
            episode: Completed episode
            rewards: Calculated rewards

        Returns:
            List of successful behavior descriptions

        """
        behaviors = []

        # High story quality
        if rewards["story_quality"] >= 8.0:
            behaviors.append("Maintain creative and engaging storytelling")
            behaviors.append("Ensure story has clear beginning, middle, and end")

        # High collaboration
        if rewards["collaboration"] >= 8.0:
            behaviors.append("Actively reference and build on teammates' contributions")
            behaviors.append("Use 'yes, and' approach to expand ideas")

        # Good efficiency
        if rewards["efficiency"] >= 8.0:
            behaviors.append("Make each contribution substantial and meaningful")
            behaviors.append("Move the story forward with every turn")

        # Analyze conversation patterns
        if len(episode.conversation) > 0:
            # Check for good transitions
            first_turn = episode.conversation[0]["content"]
            if len(first_turn) > 50:
                behaviors.append("Start with strong, detailed opening")

            # Check ending
            if episode.conversation[-1]["content"]:
                last_turn = episode.conversation[-1]["content"].lower()
                if any(word in last_turn for word in ["end", "finally", "concluded"]):
                    behaviors.append("Provide clear story conclusion")

        return behaviors
