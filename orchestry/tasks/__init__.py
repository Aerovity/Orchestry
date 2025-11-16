"""Task implementations for Orchestry MARL.

Production-ready tasks for multi-agent collaboration.
"""

from .base import BaseTask, TaskConfig
from .code_review import CodeReviewTask

__all__ = [
    "BaseTask",
    "CodeReviewTask",
    "TaskConfig",
]
