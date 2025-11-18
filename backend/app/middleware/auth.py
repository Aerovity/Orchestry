from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from app.core.config import get_settings


security = HTTPBearer()


async def verify_clerk_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Verify Clerk JWT token and return user data."""
    settings = get_settings()
    token = credentials.credentials

    # Verify with Clerk's API
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.clerk.com/v1/sessions/verify",
            headers={
                "Authorization": f"Bearer {settings.clerk_secret_key}",
                "Content-Type": "application/json",
            },
            params={"token": token},
        )

        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid authentication token")

        session_data = response.json()
        return {
            "user_id": session_data.get("user_id"),
            "session_id": session_data.get("id"),
        }


async def get_current_user(user_data: dict = Depends(verify_clerk_token)) -> dict:
    """Get current authenticated user."""
    return user_data
