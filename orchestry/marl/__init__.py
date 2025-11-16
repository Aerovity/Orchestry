"""Orchestry MARL - Multi-Agent Reinforcement Learning for LLMs.

API-based MARL implementation using:
- Group Relative Policy Optimization (GRPO)
- Multi-sample trajectory search
- Centralized value estimation
- Behavior pattern extraction
"""

from .api_grpo import Agent, APIGroupRelativePolicyOptimizer, ResponseCache
from .behavior_library import BehaviorLibrary
from .trainer import MARLTrainer
from .trajectory import MultiTurnTrajectory, TrajectoryBeam, Turn
from .value_estimator import CentralizedValueEstimator

__version__ = "1.0.0-marl"

__all__ = [
    "APIGroupRelativePolicyOptimizer",
    "Agent",
    "BehaviorLibrary",
    "CentralizedValueEstimator",
    "MARLTrainer",
    "MultiTurnTrajectory",
    "ResponseCache",
    "TrajectoryBeam",
    "Turn",
]
