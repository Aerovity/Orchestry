"""Task implementations for Orchestry MARL.

Production-ready tasks for multi-agent collaboration.
"""

from .base import BaseTask, TaskConfig
from .code_review import CodeReviewTask
from .research_lab import ResearchLabTask

__all__ = [
    "BaseTask",
    "CodeReviewTask",
    "ResearchLabTask",
    "TaskConfig",
]
