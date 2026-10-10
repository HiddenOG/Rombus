from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Which AI answers the chat: "gemini" (Google, cloud), "ollama" (local model), or
    # "openai_compatible" (any inference server: vLLM, SGLang, LM Studio, Groq, Together, OpenRouter …)
    AI_PROVIDER: Literal["gemini", "ollama", "openai_compatible"] = "gemini"

    # OpenAI-compatible inference server (only used when AI_PROVIDER=openai_compatible)
    AI_BASE_URL: str = ""
    AI_MODEL: str = ""
    AI_API_KEY: str = ""
    AI_TIMEOUT_SECONDS: float = 120

    # Fallback chain: providers tried in order until one answers, e.g. "groq,gemini,ollama".
    # Empty = use AI_PROVIDER alone.
    AI_CHAIN: str = ""

    # Groq (hosted, fast, free tier). Either name works for the key.
    GROQ_API_KEY: str = Field("", validation_alias=AliasChoices("GROQ_API_KEY", "GROQ_KEY"))
    GROQ_MODEL: str = "qwen/qwen3.8-27b"

    # OpenRouter (hosted, many models incl. free ":free" variants)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "qwen/qwen3.6-27b:free"

    # Local models via Ollama
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3.5:2b"
    OLLAMA_CONTEXT: int = 8192
    OLLAMA_TIMEOUT_SECONDS: float = 180

    # Google Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    # Tried in order when the main model is busy (comma-separated)
    GEMINI_FALLBACK_MODELS: str = "gemini-2.5-flash-lite"

    # Run user-written Python in custom code steps. It runs with full access to this machine,
    # like a Jupyter cell: fine on your own laptop, but MUST be false on any shared/public server.
    ALLOW_CUSTOM_CODE: bool = True

    # API Authentication (optional — empty string disables auth)
    API_SECRET_KEY: str = ""

    # CORS
    CORS_ORIGINS: str = "*"

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 50

    # Session management
    SESSION_TTL_HOURS: int = 24

    # Database. Empty = a local SQLite file in DATA_DIR (private, offline).
    # A PostgreSQL URL (e.g. from Supabase) works without code changes.
    DATABASE_URL: str = ""

    # Data storage directory
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


settings = Settings()
