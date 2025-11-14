"""
LLM Agent implementation for Orchestry.

Each agent has a role, memory, and learned behaviors that evolve through RL.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import anthropic
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str
    content: str
    turn: int
    agent_role: Optional[str] = None


@dataclass
class AgentMemory:
    """Stores conversation history and context for an agent."""
    messages: List[Message] = field(default_factory=list)
    learned_behaviors: List[str] = field(default_factory=list)
    successful_patterns: List[str] = field(default_factory=list)

    def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        self.messages.append(message)

    def get_recent_context(self, num_messages: int = 5) -> List[Message]:
        """Get the most recent N messages."""
        return self.messages[-num_messages:] if self.messages else []

    def clear(self) -> None:
        """Clear episodic memory (keep learned behaviors)."""
        self.messages.clear()


class LLMAgent:
    """
    An LLM-powered agent that participates in collaborative tasks.

    The agent maintains its role, memory, and learned behaviors that
    improve through reinforcement learning.
    """

    def __init__(
        self,
        role: str,
        goal: str,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        color: str = "white"
    ):
        """
        Initialize an LLM agent.

        Args:
            role: The agent's role (e.g., "Creative Writer")
            goal: The agent's specific goal
            api_key: Anthropic API key
            model: Claude model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            color: Color for CLI output
        """
        self.role = role
        self.goal = goal
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.color = color

        self.memory = AgentMemory()
        self.client = anthropic.Anthropic(api_key=api_key)

        # Base system prompt
        self.base_system_prompt = self._create_base_prompt()

        # Performance metrics
        self.total_responses = 0
        self.avg_response_quality = 0.0

    def _create_base_prompt(self) -> str:
        """Create the base system prompt for the agent."""
        return f"""You are the {self.role} in a collaborative story writing team.

Your specific goal: {self.goal}

Core Collaboration Principles:
1. Build on your teammates' ideas using "yes, and" thinking
2. Be specific, creative, and add meaningful details
3. Keep the story coherent with what came before
4. Work efficiently - make each contribution count
5. Support your teammates' creative choices

Guidelines:
- Read what others have written carefully
- Add to the story in a way that moves it forward
- Stay true to your role while being collaborative
- If you see an opportunity to tie things together, take it
- Keep your contributions focused (2-4 sentences typically)

Remember: This is a team effort. The best stories come from building on each other's ideas, not competing."""

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt including learned behaviors."""
        prompt = self.base_system_prompt

        if self.memory.learned_behaviors:
            prompt += "\n\nLearned Successful Behaviors:\n"
            for behavior in self.memory.learned_behaviors[-5:]:  # Last 5 behaviors
                prompt += f"- {behavior}\n"

        if self.memory.successful_patterns:
            prompt += "\n\nSuccessful Collaboration Patterns:\n"
            for pattern in self.memory.successful_patterns[-3:]:  # Last 3 patterns
                prompt += f"- {pattern}\n"

        return prompt

    def _format_conversation_history(self) -> List[Dict[str, str]]:
        """Format conversation history for API call."""
        messages = []

        # Add recent context
        recent_messages = self.memory.get_recent_context(10)

        for msg in recent_messages:
            role_prefix = f"[{msg.agent_role}] " if msg.agent_role else ""
            messages.append({
                "role": msg.role,
                "content": f"{role_prefix}{msg.content}"
            })

        return messages

    def act(
        self,
        task_description: str,
        conversation_history: List[Message],
        turn: int,
        rate_limit_delay: float = 1.0
    ) -> str:
        """
        Generate a response for the current turn.

        Args:
            task_description: Description of the current task
            conversation_history: Full conversation history
            turn: Current turn number
            rate_limit_delay: Delay between API calls

        Returns:
            The agent's response
        """
        # Update memory with conversation history
        self.memory.messages = conversation_history.copy()

        # Build messages for API
        messages = self._format_conversation_history()

        # Add current task context
        current_prompt = f"""Current Task: {task_description}

Turn {turn}: It's your turn as the {self.role}.

Based on what's been written so far, contribute to the story. Remember to build on your teammates' ideas and move the story forward."""

        messages.append({
            "role": "user",
            "content": current_prompt
        })

        # Rate limiting
        if self.total_responses > 0:
            time.sleep(rate_limit_delay)

        # Call API
        try:
            logger.debug(f"Agent {self.role} calling API (turn {turn})")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self._build_system_prompt(),
                messages=messages
            )

            content = response.content[0].text
            self.total_responses += 1

            logger.debug(f"Agent {self.role} received response: {content[:100]}...")

            return content

        except Exception as e:
            logger.error(f"Error in agent {self.role}: {e}")
            return f"[Error: {str(e)}]"

    def update_from_episode(
        self,
        episode_reward: float,
        successful_behaviors: List[str],
        learning_rate: float = 0.1
    ) -> None:
        """
        Update agent's learned behaviors based on episode outcome.

        Args:
            episode_reward: Total reward from the episode
            successful_behaviors: List of successful behavior descriptions
            learning_rate: How much to weight new learnings
        """
        if episode_reward > 7.0:  # High reward threshold
            # Add successful behaviors
            for behavior in successful_behaviors:
                if behavior not in self.memory.learned_behaviors:
                    self.memory.learned_behaviors.append(behavior)

            logger.info(f"Agent {self.role} learned {len(successful_behaviors)} new behaviors")

        # Limit memory to prevent prompt bloat
        if len(self.memory.learned_behaviors) > 10:
            self.memory.learned_behaviors = self.memory.learned_behaviors[-10:]

    def add_successful_pattern(self, pattern: str) -> None:
        """Add a successful collaboration pattern to memory."""
        if pattern not in self.memory.successful_patterns:
            self.memory.successful_patterns.append(pattern)

        # Limit patterns
        if len(self.memory.successful_patterns) > 5:
            self.memory.successful_patterns = self.memory.successful_patterns[-5:]

    def reset_episode_memory(self) -> None:
        """Reset episodic memory while keeping learned behaviors."""
        self.memory.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "role": self.role,
            "total_responses": self.total_responses,
            "learned_behaviors": len(self.memory.learned_behaviors),
            "successful_patterns": len(self.memory.successful_patterns)
        }
