"""
Fixed Model Baseline

Single Qwen2.5-Coder model without any training.
Generates complete solution in one shot.
"""

from orchestry.marl.local_inference import LocalLLMAgent
from orchestry.tasks.code_collaboration import CodeProblem


class FixedModelBaseline:
    """Single model baseline without training."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B") -> None:
        """Initialize fixed model."""
        self.agent = LocalLLMAgent(model_name=model_name, load_in_4bit=True)

    def solve(self, problem: CodeProblem) -> str:
        """
        Generate solution for problem.

        Args:
            problem: Code problem to solve

        Returns:
            code: Generated code

        """
        prompt = f"""Solve this coding problem:

**Problem**: {problem.description}

**Function Signature**: {problem.main_signature}

Write the complete Python function. You may include helper functions if needed."""

        return self.agent.generate(prompt)

    def evaluate(self, problems: list[CodeProblem]) -> dict[str, float]:
        """Evaluate on multiple problems."""
        from orchestry.marl.rewards import CodeCollaborationReward

        reward_model = CodeCollaborationReward(use_claude=False)
        rewards = []

        for problem in problems:
            code = self.solve(problem)

            # Simple evaluation (no helper/main split)
            # Extract main function
            main_name = problem.main_signature.split("(")[0]

            result = reward_model.evaluate(
                helper_code="",  # No helper
                main_code=code,
                test_cases=problem.tests,
                main_name=main_name,
            )
            rewards.append(result["total"])

        return {
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "pass_rate": sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0.0,
        }
