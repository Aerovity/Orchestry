"""MAGRPO algorithms for multi-agent LLM training."""

from orchestry.marl.algorithms.magrpo import (
    MAGRPOOptimizer,
    compute_advantages,
    compute_policy_loss,
)

__all__ = [
    "MAGRPOOptimizer",
    "compute_advantages",
    "compute_policy_loss",
]
