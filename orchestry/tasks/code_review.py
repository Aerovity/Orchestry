"""Code Review Task - Production multi-agent collaboration.

Three agents collaborate to improve code:
- Agent 0 (Code Writer): Writes initial code solution
- Agent 1 (Code Reviewer): Identifies issues and suggests improvements
- Agent 2 (Code Refactorer): Implements the improvements

Reward: Code quality improvement over initial version.
"""

import logging
import random
from typing import Any, TypedDict, cast

from .base import BaseTask, TaskConfig


class ProblemDict(TypedDict):
    """Type for coding problem dictionary."""

    difficulty: str
    name: str
    description: str
    requirements: list[str]
    test_cases: list[str]


logger = logging.getLogger(__name__)


class CodeReviewTask(BaseTask):
    """Code review task for MARL training.

    Episode flow:
    1. Task presents coding problem
    2. Agent 0 (Writer) writes initial solution
    3. Agent 1 (Reviewer) reviews and suggests improvements
    4. Agent 2 (Refactorer) implements improvements
    5. Cycle continues until code is satisfactory or max turns reached
    """

    # Problem bank (easy → medium → hard)
    PROBLEMS: list[ProblemDict] = [
        {
            "difficulty": "easy",
            "name": "Reverse String",
            "description": "Write a function that reverses a string.",
            "requirements": [
                "Function should be called reverse_string(s)",
                "Should handle empty strings",
                "Should preserve Unicode characters",
                "Include docstring and type hints",
            ],
            "test_cases": [
                "reverse_string('hello') == 'olleh'",
                "reverse_string('') == ''",
                "reverse_string('a') == 'a'",
            ],
        },
        {
            "difficulty": "easy",
            "name": "Is Palindrome",
            "description": "Write a function that checks if a string is a palindrome.",
            "requirements": [
                "Function should be called is_palindrome(s)",
                "Should be case-insensitive",
                "Should ignore spaces and punctuation",
                "Include docstring and type hints",
            ],
            "test_cases": [
                "is_palindrome('racecar') == True",
                "is_palindrome('hello') == False",
                "is_palindrome('A man a plan a canal Panama') == True",
            ],
        },
        {
            "difficulty": "easy",
            "name": "FizzBuzz",
            "description": "Write a function that returns FizzBuzz sequence up to n.",
            "requirements": [
                "Function should be called fizzbuzz(n)",
                "Return list of strings",
                'Multiples of 3: "Fizz", multiples of 5: "Buzz", both: "FizzBuzz"',
                "Include docstring and type hints",
            ],
            "test_cases": [
                "fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']",
                "fizzbuzz(15)[-1] == 'FizzBuzz'",
            ],
        },
        {
            "difficulty": "medium",
            "name": "Two Sum",
            "description": "Find two numbers in array that sum to target.",
            "requirements": [
                "Function should be called two_sum(nums, target)",
                "Return indices of the two numbers",
                "Assume exactly one solution exists",
                "Optimize for time complexity",
                "Include docstring and type hints",
            ],
            "test_cases": ["two_sum([2,7,11,15], 9) == [0,1]", "two_sum([3,2,4], 6) == [1,2]"],
        },
        {
            "difficulty": "medium",
            "name": "Binary Search",
            "description": "Implement binary search on sorted array.",
            "requirements": [
                "Function should be called binary_search(arr, target)",
                "Return index if found, -1 if not found",
                "Must use binary search (O(log n))",
                "Handle edge cases (empty array, single element)",
                "Include docstring and type hints",
            ],
            "test_cases": [
                "binary_search([1,2,3,4,5], 3) == 2",
                "binary_search([1,2,3,4,5], 6) == -1",
                "binary_search([], 1) == -1",
            ],
        },
        {
            "difficulty": "medium",
            "name": "Valid Parentheses",
            "description": "Check if string has valid parentheses pairing.",
            "requirements": [
                "Function should be called is_valid_parentheses(s)",
                "Handle (), {}, []",
                "Return True if valid, False otherwise",
                "Use stack-based approach",
                "Include docstring and type hints",
            ],
            "test_cases": [
                "is_valid_parentheses('()') == True",
                "is_valid_parentheses('()[]{}') == True",
                "is_valid_parentheses('(]') == False",
            ],
        },
    ]

    def __init__(self, config: TaskConfig | None = None) -> None:
        """Initialize code review task.

        Args:
            config: Task configuration

        """
        if config is None:
            config = TaskConfig(max_turns=15, min_turns=3, task_type="code_review")

        super().__init__(config)

        self.current_problem: ProblemDict | None = None
        self.initial_code: str = ""
        self.current_code: str = ""
        self.review_feedback: list[str] = []
        self.iteration: int = 0

    def reset(self) -> dict[str, Any]:
        """Reset for new code review episode.

        Returns:
            Initial observation with problem description

        """
        self.current_turn = 0
        self.history = []
        self.iteration = 0
        self.review_feedback = []
        self.initial_code = ""
        self.current_code = ""

        # Select random problem
        self.current_problem = random.choice(self.PROBLEMS)
        problem = self.current_problem  # For shorter access

        # Build task description
        requirements_str = "\n".join(f"  - {req}" for req in problem["requirements"])
        tests_str = "\n".join(f"  - {test}" for test in problem["test_cases"])

        self.task_description = f"""Code Review Task: {problem["name"]}

Problem: {problem["description"]}
Difficulty: {problem["difficulty"]}

Requirements:
{requirements_str}

Test Cases:
{tests_str}

Instructions:
- Agent 0 (Code Writer): Write initial implementation
- Agent 1 (Code Reviewer): Review code, identify issues, suggest improvements
- Agent 2 (Code Refactorer): Implement the improvements
- Continue until code quality is high or max turns reached
"""

        logger.info(f"Starting code review episode: {problem['name']} ({problem['difficulty']})")

        return {
            "task_description": self.task_description,
            "initial_context": "Begin by writing the initial code implementation.",
            "metadata": {"problem_name": problem["name"], "difficulty": problem["difficulty"]},
        }

    def step(self, agent_id: int, agent_role: str, action: str) -> tuple[dict[str, Any], bool]:
        """Execute one agent action in code review.

        Args:
            agent_id: ID of acting agent (0=Writer, 1=Reviewer, 2=Refactorer)
            agent_role: Role name
            action: Agent's response

        Returns:
            (observation, done)

        """
        self.current_turn += 1
        self.history.append(f"Turn {self.current_turn} | {agent_role}: {action}")

        # Track agent actions
        if agent_id == 0:  # Code Writer
            self._handle_writer_action(action)
        elif agent_id == 1:  # Code Reviewer
            self._handle_reviewer_action(action)
        elif agent_id == 2:  # Code Refactorer
            self._handle_refactorer_action(action)

        # Check termination
        done = self.is_done()

        observation = {
            "turn": self.current_turn,
            "last_action": action,
            "last_agent": agent_role,
            "history": self.get_history(),
            "current_code": self.current_code,
        }

        return observation, done

    def _handle_writer_action(self, action: str) -> None:
        """Handle Code Writer's action."""
        # Extract code from action (simple heuristic: look for code blocks)
        if "```python" in action:
            # Extract code between ```python and ```
            start = action.find("```python") + len("```python")
            end = action.find("```", start)
            if end != -1:
                code = action[start:end].strip()
                if not self.initial_code:
                    self.initial_code = code
                self.current_code = code
        else:
            # No code block, treat entire action as code
            self.current_code = action.strip()
            if not self.initial_code:
                self.initial_code = self.current_code

    def _handle_reviewer_action(self, action: str) -> None:
        """Handle Code Reviewer's action."""
        self.review_feedback.append(action)

    def _handle_refactorer_action(self, action: str) -> None:
        """Handle Code Refactorer's action."""
        # Extract improved code
        if "```python" in action:
            start = action.find("```python") + len("```python")
            end = action.find("```", start)
            if end != -1:
                code = action[start:end].strip()
                self.current_code = code
                self.iteration += 1

    def is_done(self) -> bool:
        """Check if episode should terminate.

        Terminates if:
        - Max turns reached
        - Code has been reviewed and refactored at least once
        - Agent signals completion (contains "FINAL CODE" or similar)
        """
        # Max turns
        if self.current_turn >= self.config.max_turns:
            return True

        # Minimum turns (at least one full cycle)
        if self.current_turn < self.config.min_turns:
            return False

        # Check for completion signal in recent history
        if self.history:
            recent = " ".join(self.history[-3:]).lower()
            if any(
                phrase in recent for phrase in ["final code", "looks good", "approved", "complete"]
            ):
                return True

        return False

    def evaluate(self) -> dict[str, float]:
        """Evaluate code review episode.

        This is a placeholder - in production, you'd run actual tests.
        For now, we use heuristics:
        - Quality: Code length, has docstring, has type hints
        - Collaboration: Number of review iterations, feedback incorporation
        - Efficiency: Completed in reasonable turns

        Returns:
            Reward components dictionary

        """
        quality_score = self._evaluate_code_quality()
        collaboration_score = self._evaluate_collaboration()
        efficiency_score = self._evaluate_efficiency()

        # Weighted total
        total = quality_score * 0.4 + collaboration_score * 0.4 + efficiency_score * 0.2

        return {
            "quality": quality_score,
            "collaboration": collaboration_score,
            "efficiency": efficiency_score,
            "total": total,
        }

    def _evaluate_code_quality(self) -> float:
        """Evaluate code quality (heuristic-based).

        Returns:
            Score 0-10

        """
        if not self.current_code:
            return 0.0

        score = 5.0  # Base score

        self.current_code.lower()

        # Check for docstring
        if '"""' in self.current_code or "'''" in self.current_code:
            score += 1.0

        # Check for type hints
        if (
            "->" in self.current_code
            or ": str" in self.current_code
            or ": int" in self.current_code
        ):
            score += 1.0

        # Check for proper function definition
        if self.current_problem is not None:
            func_name = (
                self.current_problem["requirements"][0]
                .split("(")[0]
                .replace("Function should be called ", "")
            )
            if f"def {func_name}" in self.current_code:
                score += 1.0

        # Check for comments (but not too many)
        comment_count = self.current_code.count("#")
        if 1 <= comment_count <= 5:
            score += 0.5

        # Check code length (not too short, not too long)
        code_lines = len([line for line in self.current_code.split("\n") if line.strip()])
        if 5 <= code_lines <= 30:
            score += 1.0

        # Improvement over initial
        if self.initial_code and len(self.current_code) > len(self.initial_code):
            score += 0.5

        return min(10.0, score)

    def _evaluate_collaboration(self) -> float:
        """Evaluate collaboration quality.

        Returns:
            Score 0-10

        """
        score = 5.0

        # Number of review iterations
        score += min(2.0, self.iteration * 0.5)

        # Feedback given
        score += min(2.0, len(self.review_feedback) * 0.5)

        # Check if agents referenced each other
        history_text = " ".join(self.history).lower()
        if any(
            word in history_text
            for word in ["previous", "earlier", "mentioned", "suggested", "building on"]
        ):
            score += 1.0

        return min(10.0, score)

    def _evaluate_efficiency(self) -> float:
        """Evaluate efficiency (turn count).

        Returns:
            Score 0-10

        """
        # Ideal: 6-12 turns
        # Too fast (<6): Probably didn't review properly
        # Too slow (>15): Inefficient

        if self.current_turn < self.config.min_turns:
            return 3.0
        if self.current_turn <= 9:
            return 10.0
        if self.current_turn <= 12:
            return 8.0
        if self.current_turn <= 15:
            return 6.0
        return 4.0

    def get_current_problem(self) -> dict[str, Any] | None:
        """Get current problem details."""
        return cast("dict[str, Any]", self.current_problem) if self.current_problem else None

    def get_current_code(self) -> str:
        """Get current code state."""
        return self.current_code
