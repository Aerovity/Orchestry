from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.middleware.auth import get_current_user
from app.models.training import (
    TrainingJobCreate,
    TrainingJob,
    TrainingResult,
    JobStatus,
)
from app.services.training_service import TrainingService
from app.core.database import get_db


router = APIRouter(prefix="/training", tags=["training"])


@router.post("/jobs", response_model=TrainingJob)
async def create_training_job(
    job_request: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """Create a new MARL training job."""
    service = TrainingService(get_db())

    # Create job in database
    job = await service.create_job(user["user_id"], job_request)

    # Queue the training task
    background_tasks.add_task(service.run_training, job.id)

    return job


@router.get("/jobs", response_model=list[TrainingJob])
async def list_training_jobs(
    user: dict = Depends(get_current_user),
    limit: int = 10,
    offset: int = 0,
):
    """List user's training jobs."""
    service = TrainingService(get_db())
    return await service.list_jobs(user["user_id"], limit, offset)


@router.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """Get training job details."""
    service = TrainingService(get_db())
    job = await service.get_job(job_id, user["user_id"])

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.get("/jobs/{job_id}/results", response_model=TrainingResult)
async def get_training_results(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """Get training results for a completed job."""
    service = TrainingService(get_db())
    job = await service.get_job(job_id, user["user_id"])

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    results = await service.get_results(job_id)
    return results


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """Cancel a running training job."""
    service = TrainingService(get_db())
    success = await service.cancel_job(job_id, user["user_id"])

    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job")

    return {"message": "Job cancelled"}
