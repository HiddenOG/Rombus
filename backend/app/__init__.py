from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import upload, chat, action, health, recipe
from app.session import session_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    session_manager.initialize()
    yield
    session_manager.cleanup_expired()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoML Platform — AI Data Science Copilot",
        version="3.0",
        description="AI-powered interactive data science copilot following the ASEMIC workflow.",
        lifespan=lifespan,
    )

    # CORS
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(upload.router, prefix="/api", tags=["Upload"])
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(action.router, prefix="/api", tags=["Action"])
    app.include_router(recipe.router, prefix="/api", tags=["Recipe"])

    return app


app = create_app()

