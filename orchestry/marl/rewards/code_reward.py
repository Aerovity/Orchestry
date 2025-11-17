"""
Code collaboration reward model.

Implements the level-based reward system from the paper:
1. Structural integrity (both functions defined correctly)
2. Syntax correctness (valid Python)
3. Test pass rate (functional correctness)
4. Cooperation quality (helper usage)

Rewards are accumulated only when all previous levels pass.
"""

import ast
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Single test case for code evaluation."""

    input: Any
    expected_output: Any
    description: str | None = None


class CodeCollaborationReward:
    """
    Level-based reward model for code collaboration.

    Reward components (each 0.25):
    - Structure: Both functions defined with correct signatures
    - Syntax: Code is valid Python
    - Tests: Pass rate on unit tests
    - Cooperation: Main function uses helper effectively
    """

    def __init__(
        self,
        use_claude: bool = True,
        anthropic_api_key: str | None = None,
    ) -> None:
        """
        Initialize reward model.

        Args:
            use_claude: Use Claude for nuanced cooperation evaluation
            anthropic_api_key: Anthropic API key (if None, reads from env)

        """
        self.use_claude = use_claude

        if use_claude:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def evaluate(
        self,
        helper_code: str,
        main_code: str,
        test_cases: list[TestCase],
        helper_name: str | None = None,
        main_name: str | None = None,
    ) -> dict[str, float]:
        """
        Evaluate code collaboration with level-based rewards.

        Args:
            helper_code: Helper function code
            main_code: Main function code
            test_cases: List of test cases
            helper_name: Expected helper function name
            main_name: Expected main function name

        Returns:
            rewards: Dict with components and total

        """
        rewards = {
            "structure": 0.0,
            "syntax": 0.0,
            "tests": 0.0,
            "cooperation": 0.0,
            "total": 0.0,
        }

        # Level 1: Structural integrity
        has_structure, helper_fn, main_fn = self._check_structure(
            helper_code,
            main_code,
            helper_name,
            main_name,
        )
        if has_structure:
            rewards["structure"] = 0.25
        else:
            return rewards  # Early exit if no structure

        # Level 2: Syntax correctness
        if self._check_syntax(helper_code) and self._check_syntax(main_code):
            rewards["syntax"] = 0.25
        else:
            rewards["total"] = rewards["structure"]
            return rewards  # Early exit if syntax error

        # Level 3: Test pass rate
        if main_fn is not None:
            pass_rate = self._run_tests(helper_code, main_code, test_cases, main_fn)
            rewards["tests"] = 0.25 * pass_rate

            # Level 4: Cooperation (only if at least some tests pass)
            if pass_rate > 0 and helper_fn is not None:
                coop_score = self._measure_cooperation(helper_code, main_code, helper_fn, main_fn)
                rewards["cooperation"] = 0.25 * coop_score

        rewards["total"] = sum(rewards[k] for k in ["structure", "syntax", "tests", "cooperation"])
        return rewards

    def _check_structure(
        self,
        helper_code: str,
        main_code: str,
        helper_name: str | None,
        main_name: str | None,
    ) -> tuple[bool, str | None, str | None]:
        """
        Check if both functions are defined.

        Returns:
            (has_structure, helper_func_name, main_func_name)

        """
        try:
            # Parse both code snippets
            helper_tree = ast.parse(helper_code)
            main_tree = ast.parse(main_code)

            # Extract function definitions
            helper_funcs = [
                node.name for node in ast.walk(helper_tree) if isinstance(node, ast.FunctionDef)
            ]
            main_funcs = [
                node.name for node in ast.walk(main_tree) if isinstance(node, ast.FunctionDef)
            ]

            if not helper_funcs or not main_funcs:
                return False, None, None

            # Get actual function names
            helper_fn = helper_funcs[0]
            main_fn = main_funcs[0]

            # Check expected names if provided
            if helper_name and helper_fn != helper_name:
                return False, None, None
            if main_name and main_fn != main_name:
                return False, None, None

            return True, helper_fn, main_fn

        except SyntaxError:
            return False, None, None

    def _check_syntax(self, code: str) -> bool:
        """Check if code is valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _run_tests(
        self,
        helper_code: str,
        main_code: str,
        test_cases: list[TestCase],
        main_fn: str,
    ) -> float:
        """
        Run test cases and return pass rate.

        Args:
            helper_code: Helper function code
            main_code: Main function code
            test_cases: List of test cases
            main_fn: Main function name to call

        Returns:
            pass_rate: Fraction of tests passed (0.0 to 1.0)

        """
        if not test_cases:
            return 0.0

        # Combine code
        full_code = f"{helper_code}\n\n{main_code}"

        passed = 0
        for test in test_cases:
            try:
                # Create temporary file with code
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(full_code)
                    f.write("\n\n# Test execution\n")
                    f.write(f"result = {main_fn}({test.input!r})\n")
                    f.write(f"expected = {test.expected_output!r}\n")
                    f.write("print('PASS' if result == expected else 'FAIL')\n")
                    temp_file = f.name

                # Execute code with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if "PASS" in result.stdout:
                    passed += 1

            except (subprocess.TimeoutExpired, Exception) as e:
                logger.debug(f"Test failed: {e}")
            finally:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()

        return passed / len(test_cases)

    def _measure_cooperation(
        self,
        helper_code: str,
        main_code: str,
        helper_fn: str,
        _main_fn: str,
    ) -> float:
        """
        Measure cooperation quality (0.0 to 1.0).

        Args:
            helper_code: Helper function code
            main_code: Main function code
            helper_fn: Helper function name
            main_fn: Main function name

        Returns:
            cooperation_score: 0.0 to 1.0

        """
        # Basic check: is helper called in main?
        if f"{helper_fn}(" not in main_code:
            return 0.0

        # Advanced check with Claude (if enabled)
        if self.use_claude:
            return self._claude_cooperation_eval(helper_code, main_code, helper_fn, _main_fn)

        # Simple heuristic: called = good
        return 1.0

    def _claude_cooperation_eval(
        self,
        helper_code: str,
        main_code: str,
        _helper_fn: str,
        _main_fn: str,
    ) -> float:
        """
        Use Claude Haiku for nuanced cooperation evaluation.

        Returns:
            score: 0.0 to 1.0

        """
        try:
            prompt = f"""Evaluate the cooperation between these two functions:

**Helper Function:**
```python
{helper_code}
```

**Main Function:**
```python
{main_code}
```

Rate the cooperation quality from 0.0 to 1.0:
- 1.0: Perfect cooperation, helper is essential and well-used
- 0.7: Good cooperation, helper is helpful
- 0.3: Weak cooperation, helper barely used or redundant
- 0.0: No meaningful cooperation

Respond with ONLY a number between 0.0 and 1.0."""

            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract score from response
            content = response.content[0]
            if hasattr(content, "text"):
                text = content.text.strip()
                match = re.search(r"0?\.\d+|[01]\.0", text)
                if match:
                    score = float(match.group())
                    return max(0.0, min(1.0, score))
            return 0.5  # Default score if parsing fails

        except Exception as e:
            logger.warning(f"Claude evaluation failed: {e}")
            return 0.5  # Default to neutral score


def extract_function_name(code: str) -> str | None:
    """Extract first function name from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None
    except SyntaxError:
        return None
