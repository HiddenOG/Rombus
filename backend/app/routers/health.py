"""Health, public settings, and notebook recovery."""

from fastapi import APIRouter, Depends

from app.access import owned_session
from app.config import settings
from app.models import HealthResponse
from app.session import DataSession, session_manager

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return HealthResponse(
        status="healthy",
        version="3.0",
        active_sessions=session_manager.active_count,
    )


@router.get("/config")
async def public_config():
    """What this server allows, so the interface only offers what works here."""
    return {
        "custom_code": settings.ALLOW_CUSTOM_CODE,
        "max_upload_mb": settings.MAX_UPLOAD_SIZE_MB,
        "max_rows": settings.MAX_ROWS,
    }


@router.get("/session/{session_id}")
async def get_session(session: DataSession = Depends(owned_session)):
    """Recover a notebook by id (e.g. after a page refresh)."""
    return {
        "session_id": session.session_id,
        "current_stage": session.current_stage,
        "summary": session.get_summary(),
    }
