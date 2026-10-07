from pydantic_settings import BaseSettings
from pydantic import ConfigDict
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Google Gemini
    GOOGLE_API_KEY: str = ""

    # API Authentication (optional — empty string disables auth)
    API_SECRET_KEY: str = ""

    # CORS
    CORS_ORIGINS: str = "*"

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 50

    # Session management
    SESSION_TTL_HOURS: int = 24

    # Database
    DATABASE_URL: str = "sqlite:///./data/sessions.db"

    # Data storage directory
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


settings = Settings()
