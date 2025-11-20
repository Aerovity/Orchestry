"""Behavior pattern extraction and learning.

Analyzes successful episodes to extract reusable behavioral patterns.
Uses meta-learning to improve agent prompts over time.
"""

import json
import logging
from typing import cast

import anthropic
from anthropic.types import TextBlock
import google.generativeai as genai

from .trajectory import MultiTurnTrajectory

logger = logging.getLogger(__name__)


class BehaviorLibrary:
    """Library of successful behavioral patterns learned from high-reward episodes.

    Uses Claude or Gemini to analyze what worked in successful episodes and
    extract structured behavioral patterns.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "claude",
        gemini_api_key: str | None = None,
    ) -> None:
        """Initialize behavior library.

        Args:
            api_key: API key (Anthropic for Claude, Google for Gemini)
            model: Model name for analysis
            provider: "claude" or "gemini"
            gemini_api_key: Separate Gemini API key if needed

        """
        self.provider = provider.lower()
        self.model = model

        if self.provider == "claude":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "gemini":
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
            elif api_key:
                genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model or "gemini-2.0-flash-thinking-exp")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'gemini'")

        # Storage for learned behaviors by agent role
        self.behaviors_by_role: dict[str, dict[str, list[str]]] = {}

        logger.info(f"Initialized BehaviorLibrary with {provider}")

    def extract_successful_behaviors(
        self,
        episodes: list[MultiTurnTrajectory],
        _num_agents: int,
        agent_roles: list[str],
        task_type: str = "code_review",
        top_percentile: float = 0.2,
    ) -> dict[str, dict[str, list[str]]]:
        """Extract successful behaviors from high-reward episodes.

        Analyzes top-performing episodes to identify what worked.

        Args:
            episodes: List of completed episodes
            _num_agents: Number of agents (unused)
            agent_roles: List of agent role names
            task_type: Type of task
            top_percentile: Consider top X% of episodes (default: 20%)

        Returns:
            Dictionary mapping agent_id to behavior categories:
            {
                "Agent 0": {
                    "collaboration": ["Always reference previous code", ...],
                    "code_quality": ["Use descriptive names", ...],
                    "efficiency": ["Avoid redundant checks", ...]
                },
                ...
            }

        """
        if not episodes:
            logger.warning("No episodes provided for behavior extraction")
            return {}

        # Sort episodes by reward (descending)
        sorted_episodes = sorted(episodes, key=lambda ep: ep.total_reward, reverse=True)

        # Take top percentile
        num_top = max(1, int(len(sorted_episodes) * top_percentile))
        top_episodes = sorted_episodes[:num_top]

        logger.info(
            f"Analyzing top {num_top} episodes (out of {len(episodes)}) for behavior extraction",
        )

        # Build analysis prompt
        prompt = self._build_analysis_prompt(top_episodes, agent_roles, task_type)

        # Ask LLM to analyze
        try:
            if self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.0,  # Deterministic analysis
                    messages=[{"role": "user", "content": prompt}],
                )
                # Parse structured response
                content_block = response.content[0]
                if isinstance(content_block, TextBlock):
                    text_content = content_block.text
                else:
                    text_content = str(content_block)
            elif self.provider == "gemini":
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0, max_output_tokens=4096
                    ),
                )
                text_content = response.text
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            behaviors = self._parse_behavior_response(text_content, agent_roles)

            # Update library
            for role, patterns in behaviors.items():
                if role not in self.behaviors_by_role:
                    self.behaviors_by_role[role] = {}

                for category, pattern_list in patterns.items():
                    if category not in self.behaviors_by_role[role]:
                        self.behaviors_by_role[role][category] = []

                    # Add new unique patterns
                    for pattern in pattern_list:
                        if pattern not in self.behaviors_by_role[role][category]:
                            self.behaviors_by_role[role][category].append(pattern)

            logger.info(f"Extracted behaviors for {len(behaviors)} agent roles")

            return behaviors

        except Exception as e:
            logger.exception(f"Error extracting behaviors: {e}")
            return {}

    def _build_analysis_prompt(
        self,
        episodes: list[MultiTurnTrajectory],
        agent_roles: list[str],
        task_type: str,
    ) -> str:
        """Build prompt for behavior analysis.

        Args:
            episodes: Top-performing episodes
            agent_roles: List of agent roles
            task_type: Type of task

        Returns:
            Analysis prompt

        """
        # Format episodes
        episodes_text = []
        for i, ep in enumerate(episodes[:5]):  # Max 5 episodes to keep prompt manageable
            episodes_text.append(f"--- Episode {i + 1} (Reward: {ep.total_reward:.2f}) ---")
            episodes_text.append(ep.get_full_conversation())
            episodes_text.append("")

        episodes_str = "\n".join(episodes_text)

        # Task-specific categories
        categories_map = {
            "code_review": ["collaboration", "code_quality", "efficiency"],
            "documentation": ["collaboration", "clarity", "completeness"],
            "story_writing": ["collaboration", "creativity", "coherence"],
        }

        categories = categories_map.get(task_type, ["collaboration", "quality", "efficiency"])

        return f"""You are analyzing successful multi-agent {task_type} episodes to extract behavioral patterns.

Here are the top-performing episodes:

{episodes_str}

Your task: Identify what made these episodes successful. Extract specific, actionable behavioral patterns for each agent role.

Agent Roles: {", ".join(agent_roles)}

For each agent role, identify patterns in these categories:
{", ".join(categories)}

Return your analysis as a JSON object with this exact structure:

{{
  "{agent_roles[0]}": {{
    "{categories[0]}": [
      "Specific behavior 1",
      "Specific behavior 2",
      "Specific behavior 3"
    ],
    "{categories[1]}": [...],
    "{categories[2]}": [...]
  }},
  "{agent_roles[1]}": {{
    ...
  }},
  ...
}}

Guidelines:
- Be specific and actionable (not vague like "be better")
- Focus on collaboration patterns (how agents built on each other)
- Include 3-5 behaviors per category
- Behaviors should be reproducible in future episodes

Return ONLY the JSON object, no other text."""

    def _parse_behavior_response(
        self,
        response_text: str,
        agent_roles: list[str],
    ) -> dict[str, dict[str, list[str]]]:
        """Parse behavior extraction response.

        Args:
            response_text: Raw response from Claude
            agent_roles: Expected agent roles

        Returns:
            Structured behaviors dictionary

        """
        try:
            # Extract JSON
            start = response_text.find("{")
            end = response_text.rfind("}") + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_text = response_text[start:end]
            behaviors = cast("dict[str, dict[str, list[str]]]", json.loads(json_text))

            # Validate structure
            for role in agent_roles:
                if role not in behaviors:
                    behaviors[role] = {"collaboration": [], "quality": [], "efficiency": []}

            return behaviors

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse behavior response: {e}")
            logger.debug(f"Response was: {response_text[:500]}")

            # Return empty structure
            return {
                role: {"collaboration": [], "quality": [], "efficiency": []} for role in agent_roles
            }

    def get_behaviors_for_role(
        self,
        role: str,
        category: str | None = None,
        max_behaviors: int = 5,
    ) -> list[str]:
        """Get learned behaviors for a specific agent role.

        Args:
            role: Agent role name
            category: Specific category (optional)
            max_behaviors: Maximum number to return

        Returns:
            List of behavior strings

        """
        if role not in self.behaviors_by_role:
            return []

        if category:
            behaviors = self.behaviors_by_role[role].get(category, [])
        else:
            # Return all behaviors from all categories
            behaviors = []
            for cat_behaviors in self.behaviors_by_role[role].values():
                behaviors.extend(cat_behaviors)

        # Return most recent
        return behaviors[-max_behaviors:] if behaviors else []

    def get_all_behaviors(self) -> dict[str, dict[str, list[str]]]:
        """Get all learned behaviors."""
        return self.behaviors_by_role.copy()

    def save_to_file(self, filepath: str) -> None:
        """Save behavior library to JSON file.

        Args:
            filepath: Path to save file

        """
        import json

        with open(filepath, "w") as f:
            json.dump(self.behaviors_by_role, f, indent=2)

        logger.info(f"Saved behavior library to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load behavior library from JSON file.

        Args:
            filepath: Path to load file

        """
        import json

        with open(filepath) as f:
            self.behaviors_by_role = json.load(f)

        logger.info(f"Loaded behavior library from {filepath}")

    def clear(self) -> None:
        """Clear all learned behaviors."""
        self.behaviors_by_role = {}
        logger.info("Behavior library cleared")

    def summary(self) -> str:
        """Get a summary of the behavior library."""
        lines = ["Behavior Library Summary:"]

        for role, categories in self.behaviors_by_role.items():
            total_behaviors = sum(len(behaviors) for behaviors in categories.values())
            lines.append(f"\n{role}: {total_behaviors} behaviors")

            for category, behaviors in categories.items():
                lines.append(f"  {category}: {len(behaviors)}")

        return "\n".join(lines)
