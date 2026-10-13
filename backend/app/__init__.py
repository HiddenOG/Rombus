import asyncio
import contextlib
import json
import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.routers import upload, chat, action, health, recipe, account
from app.session import session_manager

logger = logging.getLogger(__name__)

CLEANUP_EVERY_SECONDS = 3600


async def _cleanup_loop():
    """Delete old notebooks every hour (only when SESSION_TTL_HOURS is set)."""
    while True:
        try:
            removed = await run_in_threadpool(session_manager.cleanup_expired)  # database work: off the loop
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


def _finite(value):
    """NaN and infinity become None: JSON has no way to write them."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


class SafeJSONResponse(JSONResponse):
    """A safety net: one NaN anywhere in a reply (say, a statistic of an all-blank column) would
    otherwise crash the whole response."""

    def render(self, content) -> bytes:
        try:
            return json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except ValueError:
            return json.dumps(_finite(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


class BodySizeLimit:
    """Refuse oversized requests before reading them: uploads up to MAX_UPLOAD_SIZE_MB,
    everything else (chat, cells, settings) up to 2 MB. Stops one visitor filling memory or storage."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            length = dict(scope["headers"]).get(b"content-length")
            cap = (settings.MAX_UPLOAD_SIZE_MB + 1) * 1024 * 1024 if scope["path"].endswith("/upload") else 2 * 1024 * 1024
            if length and length.isdigit() and int(length) > cap:
                response = JSONResponse({"detail": f"Request too large (limit {cap // (1024 * 1024)} MB)."}, status_code=413)
                return await response(scope, receive, send)
        return await self.app(scope, receive, send)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Rhombus — data science you can trust",
        version="3.0",
        description="Rhombus API: a data science notebook with an AI copilot, following the ASEMIC workflow.",
        lifespan=lifespan,
        default_response_class=SafeJSONResponse,
    )
    app.add_middleware(BodySizeLimit)

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
