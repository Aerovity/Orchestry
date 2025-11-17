"""
Naive Concatenation Baseline

Two agents generate in parallel without communication.
Outputs are simply concatenated.
"""

from orchestry.marl.local_inference import LocalLLMAgent
from orchestry.tasks.code_collaboration import CodeProblem


class NaiveConcatenationBaseline:
    """Two agents generating in parallel, no communication."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B") -> None:
        """Initialize two independent agents."""
        self.agent_helper = LocalLLMAgent(model_name=model_name, load_in_4bit=True)
        self.agent_main = LocalLLMAgent(model_name=model_name, load_in_4bit=True)

    def solve(self, problem: CodeProblem) -> tuple[str, str]:
        """
        Generate solution with parallel agents.

        Args:
            problem: Code problem to solve

        Returns:
            (helper_code, main_code): Generated codes

        """
        # Helper prompt (no context about main)
        helper_prompt = f"""Write a helper function for this problem:

**Problem**: {problem.description}

**Helper Role**: {problem.helper_role}

**Function Signature**: {problem.helper_signature}

Write only the helper function."""

        # Main prompt (no context about helper)
        main_prompt = f"""Write the main function for this problem:

**Problem**: {problem.description}

**Main Role**: {problem.main_role}

**Function Signature**: {problem.main_signature}

Write only the main function."""

        # Generate in parallel (simulated - actually sequential)
        helper_code = self.agent_helper.generate(helper_prompt)
        main_code = self.agent_main.generate(main_prompt)

        return helper_code, main_code

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
