from fastapi import APIRouter


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "orchestry-api"}


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Orchestry API",
        "version": "1.0.0",
        "docs": "/docs",
    }
