"""
Code collaboration task for MAGRPO.

Two agents collaborate to solve coding problems:
- Agent 0 (Helper): Generates auxiliary helper function
- Agent 1 (Main): Generates main function using helper
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestry.marl.rewards.code_reward import CodeCollaborationReward, TestCase
from orchestry.tasks.base import BaseTask, TaskConfig


@dataclass
class CodeProblem:
    """Single code collaboration problem."""

    id: str
    description: str
    helper_role: str
    main_role: str
    helper_signature: str
    main_signature: str
    tests: list[TestCase]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeProblem":
        """Create problem from dictionary."""
        tests = [
            TestCase(input=test["input"], expected_output=test["expected"])
            for test in data["tests"]
        ]
        return cls(
            id=data["id"],
            description=data["description"],
            helper_role=data["helper_role"],
            main_role=data["main_role"],
            helper_signature=data["helper_signature"],
            main_signature=data["main_signature"],
            tests=tests,
        )


class CodeCollaborationTask(BaseTask):
    """
    Code collaboration task implementing Dec-POMDP.

    State: Problem description + current code
    Observations: Agent-specific prompts
    Actions: Generated code (helper or main function)
    Rewards: Level-based (structure, syntax, tests, cooperation)
    """

    def __init__(
        self,
        problems_file: str = "datasets/coop_problems.json",
        reward_model: CodeCollaborationReward | None = None,
        use_claude_eval: bool = True,
    ) -> None:
        """
        Initialize code collaboration task.

        Args:
            problems_file: Path to problems JSON file
            reward_model: Reward model (if None, creates default)
            use_claude_eval: Use Claude for cooperation evaluation

        """
        # Load problems
        problems_path = Path(problems_file)
        if not problems_path.exists():
            # Try relative to repo root
            repo_root = Path(__file__).parent.parent.parent
            problems_path = repo_root / problems_file

        with problems_path.open() as f:
            data = json.load(f)

        self.problems = [CodeProblem.from_dict(p) for p in data["problems"]]

        # Create reward model
        if reward_model is None:
            reward_model = CodeCollaborationReward(use_claude=use_claude_eval)
        self.reward_model = reward_model

        # Episode state
        self.current_problem: CodeProblem | None = None
        self.helper_code: str | None = None
        self.main_code: str | None = None
        self.turn_count = 0

        super().__init__(
            TaskConfig(
                max_turns=2,  # Helper turn, then main turn
            ),
        )

    def reset(self) -> dict[int, str]:  # type: ignore[override]
        """
        Start new episode.

        Returns:
            observations: Dict mapping agent_id to observation string

        """
        # Sample a random problem
        self.current_problem = random.choice(self.problems)
        self.helper_code = None
        self.main_code = None
        self.turn_count = 0

        # Return observation for helper agent (agent 0)
        return {0: self._create_helper_prompt()}

    def step(self, actions: dict[int, str]) -> dict[str, Any]:  # type: ignore[override]
        """
        Execute agent actions.

        Args:
            actions: Dict mapping agent_id to action (code string)

        Returns:
            result: Dict with observations, rewards, done flag

        """
        self.turn_count += 1

        if 0 in actions:
            # Helper agent's turn
            self.helper_code = self._extract_code(actions[0])

            return {
                "observations": {1: self._create_main_prompt()},
                "rewards": None,
                "done": False,
                "info": {"turn": "helper"},
            }

        if 1 in actions:
            # Main agent's turn
            self.main_code = self._extract_code(actions[1])

            # Evaluate collaboration
            rewards = self.evaluate()

            return {
                "observations": None,
                "rewards": rewards,
                "done": True,
                "info": {
                    "turn": "main",
                    "problem_id": self.current_problem.id if self.current_problem else "",
                },
            }

        msg = f"Invalid agent_id in actions: {actions.keys()}"
        raise ValueError(msg)

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the collaboration.

        Returns:
            rewards: Dict with reward components

        """
        if self.helper_code is None or self.main_code is None or self.current_problem is None:
            return {
                "structure": 0.0,
                "syntax": 0.0,
                "tests": 0.0,
                "cooperation": 0.0,
                "total": 0.0,
            }

        # Extract function names from signatures
        helper_name = self.current_problem.helper_signature.split("(")[0]
        main_name = self.current_problem.main_signature.split("(")[0]

        # Evaluate with reward model
        return self.reward_model.evaluate(
            helper_code=self.helper_code,
            main_code=self.main_code,
            test_cases=self.current_problem.tests,
            helper_name=helper_name,
            main_name=main_name,
        )

    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.turn_count >= self.config.max_turns

    def _create_helper_prompt(self) -> str:
        """Create prompt for helper agent."""
        if self.current_problem is None:
            return ""
        return f"""You are a helper agent writing auxiliary functions for code problems.

**Problem**: {self.current_problem.description}

**Your Role**: {self.current_problem.helper_role}

**Function Signature**: {self.current_problem.helper_signature}

Write ONLY the helper function implementation in Python. Do not include any tests or examples."""

    def _create_main_prompt(self) -> str:
        """Create prompt for main agent."""
        if self.current_problem is None or self.helper_code is None:
            return ""
        return f"""You are the main agent completing code solutions.

**Problem**: {self.current_problem.description}

**Helper Function** (already implemented):
```python
{self.helper_code}
```

**Your Role**: {self.current_problem.main_role}

**Function Signature**: {self.current_problem.main_signature}

Write ONLY the main function implementation in Python. You should use the helper function above.
Do not include tests or examples."""

    def _extract_code(self, text: str) -> str:
        """
        Extract code from agent response.

        Handles cases where agent includes markdown code blocks or explanations.
        """
        # Remove markdown code blocks if present
        if "```python" in text:
            # Extract code between ```python and ```
            parts = text.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()

        if "```" in text:
            # Generic code block
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()

        # If no code blocks, return as is (might just be the function)
        return text.strip()

    def get_train_test_split(
        self,
        train_ratio: float = 0.75,
    ) -> tuple[list[CodeProblem], list[CodeProblem]]:
        """
        Split problems into train and test sets.

        Args:
            train_ratio: Fraction of problems for training

        Returns:
            (train_problems, test_problems): Train and test sets

        """
        shuffled = self.problems.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_problems = shuffled[:split_idx]
        test_problems = shuffled[split_idx:]

        return train_problems, test_problems

    def set_problems(self, problems: list[CodeProblem]) -> None:
        """Set specific problems to use (for train/test split)."""
        self.problems = problems
