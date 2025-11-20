"""API-based Group Relative Policy Optimization (GRPO).

Adapted for LLM APIs without fine-tuning:
- Multi-sample response generation
- Group-relative advantage estimation
- Best-response trajectory selection
- Behavior pattern extraction for learning
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import anthropic
import google.generativeai as genai
import numpy as np
from anthropic.types import TextBlock
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Simple agent wrapper for GRPO."""

    agent_id: int
    role: str
    goal: str
    system_prompt: str
    learned_behaviors: list[str]


class ResponseCache:
    """Cache for API responses to reduce costs."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum number of cached responses

        """
        self.cache: dict[str, list[str]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(self, agent_id: int, context: str, temperature: float, k: int) -> str:
        """Create cache key from request parameters."""
        content = f"{agent_id}:{context}:{temperature}:{k}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, agent_id: int, context: str, temperature: float, k: int) -> list[str] | None:
        """Get cached responses if available."""
        key = self._make_key(agent_id, context, temperature, k)

        if key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit! ({self.hits} hits / {self.misses} misses)")
            return self.cache[key]

        self.misses += 1
        return None

    def put(
        self,
        agent_id: int,
        context: str,
        temperature: float,
        k: int,
        responses: list[str],
    ) -> None:
        """Cache responses."""
        key = self._make_key(agent_id, context, temperature, k)

        # Simple LRU: if cache full, remove first item
        if len(self.cache) >= self.max_size:
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[key] = responses

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (
                self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0
            ),
        }


class APIGroupRelativePolicyOptimizer:
    """GRPO adapted for API-based LLMs.

    Core algorithm:
    1. Generate k response samples per agent per turn
    2. Build candidate trajectories using beam search
    3. Score trajectories with centralized value estimator
    4. Compute group-relative advantages
    5. Select best trajectory
    6. Extract successful patterns for learning
    """

    def __init__(
        self,
        agents: list[Agent],
        api_key: str,
        config: dict[str, Any] | None = None,
        provider: str = "claude",
        gemini_api_key: str | None = None,
    ) -> None:
        """Initialize GRPO optimizer.

        Args:
            agents: List of agents
            api_key: API key (Anthropic for Claude, Google for Gemini)
            config: Configuration dictionary
            provider: "claude" or "gemini"
            gemini_api_key: Separate Gemini API key if needed

        """
        self.agents = agents
        self.num_agents = len(agents)
        self.provider = provider.lower()

        # Configuration
        config = config or {}
        self.k_samples = config.get("k_samples", 5)
        self.temperature = config.get("temperature", 0.8)
        self.model = config.get("model")
        self.max_tokens = config.get("max_tokens", 1024)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.5)

        # Initialize client based on provider
        if self.provider == "claude":
            self.client = anthropic.Anthropic(api_key=api_key)
            if not self.model:
                self.model = "claude-3-5-sonnet-20241022"
        elif self.provider == "gemini":
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
            elif api_key:
                genai.configure(api_key=api_key)
            if not self.model:
                self.model = "gemini-2.0-flash-exp"
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'gemini'")

        # Response cache
        self.cache = ResponseCache(max_size=config.get("cache_size", 1000))

        logger.info(
            f"Initialized GRPO with {self.num_agents} agents using {self.provider}, "
            f"k={self.k_samples}, temp={self.temperature}, model={self.model}",
        )

    def generate_response_samples(
        self,
        agent: Agent,
        context: str,
        k: int | None = None,
        use_cache: bool = True,
    ) -> list[str]:
        """Generate k response candidates for an agent.

        Uses temperature sampling for diversity.

        Args:
            agent: Agent to generate responses
            context: Conversation context
            k: Number of samples (defaults to self.k_samples)
            use_cache: Whether to use response cache

        Returns:
            List of k possible responses

        """
        k = k or self.k_samples

        # Check cache
        if use_cache:
            cached = self.cache.get(agent.agent_id, context, self.temperature, k)
            if cached is not None:
                return cached

        # Build system prompt with learned behaviors
        system_prompt = self._build_agent_system_prompt(agent)

        # Generate k samples
        samples = []
        for i in range(k):
            try:
                # Rate limiting
                if i > 0:
                    time.sleep(self.rate_limit_delay)

                logger.debug(f"Generating sample {i + 1}/{k} for agent {agent.role}")

                # Generate based on provider
                if self.provider == "claude":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": context}],
                    )

                    content_block = response.content[0]
                    if isinstance(content_block, TextBlock):
                        sample_text = content_block.text
                    else:
                        sample_text = str(content_block)
                elif self.provider == "gemini":
                    # Combine system prompt and context for Gemini
                    full_prompt = f"{system_prompt}\n\n{context}"
                    response = self.client.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        ),
                    )
                    sample_text = response.text
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

                samples.append(sample_text)

            except Exception as e:
                logger.exception(f"Error generating sample {i + 1} for {agent.role}: {e}")
                # Use fallback response
                samples.append(f"[Error generating response: {e!s}]")

        # Cache the results
        if use_cache:
            self.cache.put(agent.agent_id, context, self.temperature, k, samples)

        logger.debug(f"Generated {len(samples)} samples for agent {agent.role}")

        return samples

    def _build_agent_system_prompt(self, agent: Agent) -> str:
        """Build system prompt including learned behaviors.

        Args:
            agent: Agent to build prompt for

        Returns:
            Complete system prompt

        """
        prompt = agent.system_prompt

        # Add learned behaviors
        if agent.learned_behaviors:
            prompt += "\n\nLearned Successful Behaviors:\n"
            for behavior in agent.learned_behaviors[-5:]:  # Last 5
                prompt += f"- {behavior}\n"

        return prompt

    def compute_advantages(self, rewards: list[float]) -> NDArray[np.floating[Any]]:
        """Compute group-relative advantages.

        Advantage(trajectory_i) = Reward(trajectory_i) - GroupMean(rewards)

        This encourages agents to optimize joint reward, not individual.

        Args:
            rewards: List of rewards for trajectories

        Returns:
            Array of advantages

        """
        rewards_array: NDArray[np.floating[Any]] = np.array(rewards)
        baseline = np.mean(rewards_array)
        advantages: NDArray[np.floating[Any]] = rewards_array - baseline

        logger.debug(
            f"Computed advantages: mean={baseline:.2f}, "
            f"range=[{advantages.min():.2f}, {advantages.max():.2f}]",
        )

        return advantages

    def select_best_trajectory(self, advantages: np.ndarray, exploration_rate: float = 0.1) -> int:
        """Select trajectory index using advantage-based sampling.

        With probability (1-exploration_rate): pick max advantage
        With probability exploration_rate: sample proportional to exp(advantage)

        Args:
            advantages: Advantage values
            exploration_rate: Probability of exploration

        Returns:
            Index of selected trajectory

        """
        if np.random.random() < exploration_rate:
            # Exploration: sample proportional to exp(advantage)
            # Shift advantages to avoid overflow
            shifted_adv = advantages - advantages.max()
            probs = np.exp(shifted_adv)
            probs = probs / probs.sum()

            selected_idx = int(np.random.choice(len(advantages), p=probs))
            logger.debug(
                f"Exploration: selected trajectory {selected_idx} "
                f"(advantage={advantages[selected_idx]:.2f})",
            )
        else:
            # Exploitation: pick best
            selected_idx = int(np.argmax(advantages))
            logger.debug(
                f"Exploitation: selected best trajectory {selected_idx} "
                f"(advantage={advantages[selected_idx]:.2f})",
            )

        return selected_idx

    def update_agent_behaviors(
        self,
        agent: Agent,
        new_behaviors: list[str],
        max_behaviors: int = 10,
    ) -> None:
        """Update agent's learned behaviors.

        Args:
            agent: Agent to update
            new_behaviors: New behaviors to add
            max_behaviors: Maximum number of behaviors to keep

        """
        for behavior in new_behaviors:
            if behavior not in agent.learned_behaviors:
                agent.learned_behaviors.append(behavior)

        # Keep only most recent
        if len(agent.learned_behaviors) > max_behaviors:
            agent.learned_behaviors = agent.learned_behaviors[-max_behaviors:]

        logger.info(
            f"Updated {agent.role}: now has {len(agent.learned_behaviors)} learned behaviors",
        )

    def get_cache_stats(self) -> dict[str, int | float]:
        """Get response cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear response cache."""
        self.cache = ResponseCache()
        logger.info("Response cache cleared")
