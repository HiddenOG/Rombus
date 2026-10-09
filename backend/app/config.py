from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Google Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    # Tried in order when the main model is busy (comma-separated)
    GEMINI_FALLBACK_MODELS: str = "gemini-2.5-flash-lite"

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
