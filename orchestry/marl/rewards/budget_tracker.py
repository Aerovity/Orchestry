"""
Budget tracking for API costs during training.

Ensures we don't exceed the experiment budget.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""


class BudgetTracker:
    """
    Track API spending and enforce budget limits.

    Usage:
        tracker = BudgetTracker(max_budget=15.0)

        # Before each API call
        tracker.track_call(cost=0.005)  # Will raise if exceeds budget
    """

    # Pricing (as of 2024, subject to change)
    HAIKU_INPUT_COST = 0.25 / 1_000_000  # $0.25 per MTok
    HAIKU_OUTPUT_COST = 1.25 / 1_000_000  # $1.25 per MTok
    SONNET_INPUT_COST = 3.00 / 1_000_000  # $3.00 per MTok
    SONNET_OUTPUT_COST = 15.00 / 1_000_000  # $15.00 per MTok

    def __init__(self, max_budget: float = 15.0, warn_threshold: float = 0.8) -> None:
        """
        Initialize budget tracker.

        Args:
            max_budget: Maximum budget in USD
            warn_threshold: Warn when this fraction of budget is used

        """
        self.max_budget = max_budget
        self.warn_threshold = warn_threshold
        self.total_spent = 0.0
        self.call_count = 0
        self.start_time = datetime.now()

        logger.info(f"Budget tracker initialized: ${max_budget:.2f} limit")

    def track_call(self, cost: float, description: str = "") -> None:
        """
        Track an API call cost.

        Args:
            cost: Cost in USD
            description: Optional description of the call

        Raises:
            BudgetExceededError: If spending exceeds max_budget

        """
        self.total_spent += cost
        self.call_count += 1

        # Check if budget exceeded
        if self.total_spent > self.max_budget:
            msg = f"Budget exceeded: ${self.total_spent:.2f} > ${self.max_budget:.2f}"
            raise BudgetExceededError(msg)

        # Warn if approaching limit
        if self.total_spent > self.max_budget * self.warn_threshold:
            remaining = self.max_budget - self.total_spent
            logger.warning(
                f"Budget warning: ${self.total_spent:.2f} / ${self.max_budget:.2f} "
                f"({self.total_spent / self.max_budget * 100:.1f}%) - ${remaining:.2f} remaining",
            )

        if description:
            logger.debug(f"API call: {description} - ${cost:.4f}")

    def estimate_claude_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "haiku",
    ) -> float:
        """
        Estimate cost for Claude API call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: "haiku" or "sonnet"

        Returns:
            estimated_cost: Cost in USD

        """
        if model == "haiku":
            return input_tokens * self.HAIKU_INPUT_COST + output_tokens * self.HAIKU_OUTPUT_COST
        if model == "sonnet":
            return input_tokens * self.SONNET_INPUT_COST + output_tokens * self.SONNET_OUTPUT_COST
        msg = f"Unknown model: {model}"
        raise ValueError(msg)

    def get_stats(self) -> dict[str, float]:
        """Get budget statistics."""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_spent": self.total_spent,
            "max_budget": self.max_budget,
            "remaining": self.max_budget - self.total_spent,
            "percent_used": (self.total_spent / self.max_budget) * 100,
            "call_count": self.call_count,
            "avg_cost_per_call": self.total_spent / self.call_count if self.call_count > 0 else 0,
            "elapsed_hours": elapsed_time / 3600,
            "cost_per_hour": (self.total_spent / elapsed_time * 3600) if elapsed_time > 0 else 0,
        }

    def print_summary(self) -> None:
        """Print budget summary."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("BUDGET SUMMARY")
        print("=" * 50)
        print(f"Total Spent:      ${stats['total_spent']:.2f}")
        print(f"Max Budget:       ${stats['max_budget']:.2f}")
        print(f"Remaining:        ${stats['remaining']:.2f}")
        print(f"Percent Used:     {stats['percent_used']:.1f}%")
        print(f"Total API Calls:  {stats['call_count']}")
        print(f"Avg Cost/Call:    ${stats['avg_cost_per_call']:.4f}")
        print(f"Elapsed Time:     {stats['elapsed_hours']:.2f} hours")
        print(f"Cost/Hour:        ${stats['cost_per_hour']:.2f}")
        print("=" * 50 + "\n")

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if we can afford a given cost."""
        return (self.total_spent + estimated_cost) <= self.max_budget

    def reset(self) -> None:
        """Reset budget tracker."""
        self.total_spent = 0.0
        self.call_count = 0
        self.start_time = datetime.now()
        logger.info("Budget tracker reset")
