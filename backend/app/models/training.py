from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingConfig(BaseModel):
    task_type: str = "code_review"
    episodes: int = 10
    beam_width: int = 10
    k_samples: int = 5
    temperature: float = 0.8


class TrainingJobCreate(BaseModel):
    config: TrainingConfig
    anthropic_api_key: str | None = None  # User can provide their own key


class TrainingJob(BaseModel):
    id: str
    user_id: str
    config: TrainingConfig
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_episode: int = 0
    total_episodes: int
    average_reward: float | None = None
    error_message: str | None = None


class TrainingResult(BaseModel):
    job_id: str
    episodes: list[dict]
    rewards: list[float]
    learned_behaviors: list[dict]
    summary: dict
