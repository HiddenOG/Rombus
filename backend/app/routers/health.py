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
    from app.sandbox import status
    from app.tools.model import CLASSIFICATION_MODELS, MODEL_HINTS, MODEL_NAMES, REGRESSION_MODELS
    from starlette.concurrency import run_in_threadpool
    code = await run_in_threadpool(status)  # the first call may run the sandbox's self-test
    return {
        "custom_code": code["enabled"],
        "code_for": settings.CODE_FOR,  # "accounts": anonymous visitors are asked to sign in to run code
        "sandbox": code.get("sandbox"),
        "code_off_reason": code.get("reason"),  # e.g. which sandbox protection failed its self-test
        # Readable model names (lr -> logistic regression), and which models fit which kind of problem
        "model_names": MODEL_NAMES,
        "model_hints": MODEL_HINTS,
        "models_for": {"classification": list(CLASSIFICATION_MODELS), "regression": list(REGRESSION_MODELS)},
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
