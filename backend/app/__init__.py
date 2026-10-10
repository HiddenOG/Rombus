import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import upload, chat, action, health, recipe, account
from app.session import session_manager

logger = logging.getLogger(__name__)

CLEANUP_EVERY_SECONDS = 3600


async def _cleanup_loop():
    """Delete old notebooks every hour (only when SESSION_TTL_HOURS is set)."""
    while True:
        try:
            removed = session_manager.cleanup_expired()
            if removed:
                logger.info("Cleaned up %d old notebooks", removed)
        except Exception:
            logger.exception("Notebook cleanup failed")
        await asyncio.sleep(CLEANUP_EVERY_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    session_manager.initialize()
    cleanup = asyncio.create_task(_cleanup_loop()) if settings.SESSION_TTL_HOURS > 0 else None
    yield
    if cleanup:
        cleanup.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup


def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoML Platform — AI Data Science Copilot",
        version="3.0",
        description="AI-powered interactive data science copilot following the ASEMIC workflow.",
        lifespan=lifespan,
    )

    # CORS: only the configured sites may call the API. No cookies are used (owners are identified
    # by the Authorization and X-Visitor-Id headers), so credentials stay off.
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-Visitor-Id", "Authorization"],
        expose_headers=["Content-Disposition"],  # so exports keep their file names
    )

    # Include routers
    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(account.router, prefix="/api", tags=["Account"])
    app.include_router(upload.router, prefix="/api", tags=["Upload"])
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(action.router, prefix="/api", tags=["Action"])
    app.include_router(recipe.router, prefix="/api", tags=["Recipe"])

    return app


app = create_app()
