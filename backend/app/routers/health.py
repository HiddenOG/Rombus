"""Health and session recovery router."""

from fastapi import APIRouter, HTTPException, Depends

from app.auth import verify_api_key
from app.models import HealthResponse
from app.session import session_manager

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return HealthResponse(
        status="healthy",
        version="3.0",
        active_sessions=session_manager.active_count,
    )


@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    _: str = Depends(verify_api_key),
):
    """Recover a session by ID (e.g., after page refresh)."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    return {
        "session_id": session.session_id,
        "current_stage": session.current_stage,
        "summary": session.get_summary(),
    }
