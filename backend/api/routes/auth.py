# backend/api/routes/auth.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.config import get_settings
from backend.core.logger import get_logger

from backend.utils.auth import create_token

logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/login")
async def login(payload: LoginRequest):
    """
    Single hardcoded account for demo purposes.
    Credentials live in .env, never in code.
    """
    if payload.email == settings.app_user_email and payload.password == settings.app_user_password:
        logger.info(f"Login successful for {payload.email}")
        token = create_token(payload.email)
        return {"success": True, "token": token}
    raise HTTPException(status_code=401, detail="Invalid email or password.")