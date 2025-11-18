from datetime import datetime, timezone
import uuid
import json
from supabase import Client

from app.models.training import (
    TrainingJobCreate,
    TrainingJob,
    TrainingResult,
    JobStatus,
)


class TrainingService:
    def __init__(self, db: Client):
        self.db = db

    async def create_job(
        self, user_id: str, job_request: TrainingJobCreate
    ) -> TrainingJob:
        """Create a new training job in the database."""
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        job_data = {
            "id": job_id,
            "user_id": user_id,
            "config": job_request.config.model_dump(),
            "status": JobStatus.PENDING,
            "created_at": now.isoformat(),
            "total_episodes": job_request.config.episodes,
            "current_episode": 0,
        }

        # Store API key separately (encrypted in production)
        if job_request.anthropic_api_key:
            job_data["anthropic_api_key"] = job_request.anthropic_api_key

        self.db.table("training_jobs").insert(job_data).execute()

        return TrainingJob(**job_data)

    async def get_job(self, job_id: str, user_id: str) -> TrainingJob | None:
        """Get a training job by ID."""
        result = (
            self.db.table("training_jobs")
            .select("*")
            .eq("id", job_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not result.data:
            return None

        return TrainingJob(**result.data[0])

    async def list_jobs(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> list[TrainingJob]:
        """List training jobs for a user."""
        result = (
            self.db.table("training_jobs")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return [TrainingJob(**job) for job in result.data]

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        current_episode: int | None = None,
        average_reward: float | None = None,
        error_message: str | None = None,
    ):
        """Update job status."""
        update_data = {"status": status}

        if current_episode is not None:
            update_data["current_episode"] = current_episode
        if average_reward is not None:
            update_data["average_reward"] = average_reward
        if error_message is not None:
            update_data["error_message"] = error_message

        if status == JobStatus.RUNNING:
            update_data["started_at"] = datetime.now(timezone.utc).isoformat()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

        self.db.table("training_jobs").update(update_data).eq("id", job_id).execute()

    async def run_training(self, job_id: str):
        """Run the MARL training for a job."""
        # Get job details
        result = (
            self.db.table("training_jobs").select("*").eq("id", job_id).execute()
        )

        if not result.data:
            return

        job_data = result.data[0]
        config = job_data["config"]

        try:
            await self.update_job_status(job_id, JobStatus.RUNNING)

            # Import MARL trainer
            from orchestry.marl.trainer import MARLTrainer
            from orchestry.tasks.code_review import CodeReviewTask

            # Initialize task
            task = CodeReviewTask()

            # Initialize trainer with config
            trainer = MARLTrainer(
                task=task,
                beam_width=config.get("beam_width", 10),
                k_samples=config.get("k_samples", 5),
                temperature=config.get("temperature", 0.8),
            )

            # Training loop with progress updates
            episodes_data = []
            rewards = []

            for episode in range(config["episodes"]):
                # Run single episode
                episode_result = trainer.run_episode()
                episodes_data.append(episode_result)
                rewards.append(episode_result.get("reward", 0))

                # Update progress
                avg_reward = sum(rewards) / len(rewards)
                await self.update_job_status(
                    job_id,
                    JobStatus.RUNNING,
                    current_episode=episode + 1,
                    average_reward=avg_reward,
                )

            # Save results
            results_data = {
                "job_id": job_id,
                "episodes": episodes_data,
                "rewards": rewards,
                "learned_behaviors": trainer.get_learned_behaviors(),
                "summary": {
                    "total_episodes": len(episodes_data),
                    "average_reward": sum(rewards) / len(rewards) if rewards else 0,
                    "best_reward": max(rewards) if rewards else 0,
                },
            }

            self.db.table("training_results").insert(results_data).execute()

            await self.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                current_episode=config["episodes"],
                average_reward=results_data["summary"]["average_reward"],
            )

        except Exception as e:
            await self.update_job_status(
                job_id, JobStatus.FAILED, error_message=str(e)
            )

    async def get_results(self, job_id: str) -> TrainingResult:
        """Get training results for a job."""
        result = (
            self.db.table("training_results")
            .select("*")
            .eq("job_id", job_id)
            .execute()
        )

        if not result.data:
            raise ValueError(f"No results found for job {job_id}")

        return TrainingResult(**result.data[0])

    async def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel a running job."""
        job = await self.get_job(job_id, user_id)

        if not job or job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
            return False

        await self.update_job_status(job_id, JobStatus.CANCELLED)
        return True
