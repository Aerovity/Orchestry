"""
One-Round Discussion Baseline

Agents generate once, exchange outputs, then generate again.
Two-way communication, no training.
"""

from orchestry.marl.local_inference import LocalLLMAgent
from orchestry.tasks.code_collaboration import CodeProblem


class OneRoundDiscussionBaseline:
    """One round of discussion between agents."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B") -> None:
        """Initialize two agents."""
        self.agent_helper = LocalLLMAgent(model_name=model_name, load_in_4bit=True)
        self.agent_main = LocalLLMAgent(model_name=model_name, load_in_4bit=True)

    def solve(self, problem: CodeProblem) -> tuple[str, str]:
        """
        Generate solution with one round of discussion.

        Args:
            problem: Code problem to solve

        Returns:
            (helper_code, main_code): Final generated codes

        """
        # Round 1: Both generate independently
        helper_prompt_1 = f"""Write a helper function for this problem:

**Problem**: {problem.description}

**Helper Role**: {problem.helper_role}

**Function Signature**: {problem.helper_signature}

Write only the helper function."""

        main_prompt_1 = f"""Write the main function for this problem:

**Problem**: {problem.description}

**Main Role**: {problem.main_role}

**Function Signature**: {problem.main_signature}

Write only the main function."""

        helper_code_1 = self.agent_helper.generate(helper_prompt_1)
        main_code_1 = self.agent_main.generate(main_prompt_1)

        # Round 2: Each sees the other's output
        helper_prompt_2 = f"""Revise your helper function based on the main function:

**Problem**: {problem.description}

**Your previous helper**:
```python
{helper_code_1}
```

**Main function** (from other agent):
```python
{main_code_1}
```

Write an improved helper function that better supports the main function."""

        main_prompt_2 = f"""Revise your main function based on the helper function:

**Problem**: {problem.description}

**Your previous main**:
```python
{main_code_1}
```

**Helper function** (from other agent):
```python
{helper_code_1}
```

Write an improved main function that better uses the helper function."""

        helper_code_2 = self.agent_helper.generate(helper_prompt_2)
        main_code_2 = self.agent_main.generate(main_prompt_2)

        return helper_code_2, main_code_2

    def evaluate(self, problems: list[CodeProblem]) -> dict[str, float]:
        """Evaluate on multiple problems."""
        from orchestry.marl.rewards import CodeCollaborationReward

        reward_model = CodeCollaborationReward(use_claude=False)
        rewards = []

        for problem in problems:
            helper_code, main_code = self.solve(problem)

            helper_name = problem.helper_signature.split("(")[0]
            main_name = problem.main_signature.split("(")[0]

            result = reward_model.evaluate(
                helper_code=helper_code,
                main_code=main_code,
                test_cases=problem.tests,
                helper_name=helper_name,
                main_name=main_name,
            )
            rewards.append(result["total"])

        return {
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "pass_rate": sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0.0,
        }
