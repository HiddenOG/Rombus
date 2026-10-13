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
    # Free models come and go; this one supported tool calling and answered in Oct 2026
    OPENROUTER_MODEL: str = "nvidia/nemotron-3.5-lightning:free"

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

    # Code cells (user- or AI-written Python). Code never runs inside the app: it runs in a sandbox
    # process (app/sandbox.py), chosen by CODE_SANDBOX:
    #   local     your own machine: a separate worker with a time limit, no other protection
    #   hardened  a server: another user, memory/CPU limits, no network, no starting programs (Linux,
    #             the Dockerfile starts it); falls back to Modal if its self-test fails and Modal is set
    #   modal     a hosted Modal sandbox (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)
    ALLOW_CUSTOM_CODE: bool = True
    CODE_SANDBOX: str = "local"
    CODE_FOR: str = "everyone"  # or "accounts": only notebooks in a signed-in account may run code
    CODE_TIME_LIMIT_SECONDS: int = 60  # per cell (training custom models gets 5x)
    CODE_MEMORY_MB: int = 2048
    SANDBOX_SOCKET: str = "/run/rhombus/sandbox.sock"
    MODAL_TOKEN_ID: str = ""
    MODAL_TOKEN_SECRET: str = ""

    # CORS: the sites allowed to call this API (comma-separated). Set to your real site when deployed.
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Usage limits (protect free AI allowances and the server)
    RATE_LIMITS: bool = True
    AI_PER_MINUTE: int = 10          # copilot messages per visitor per minute
    AI_DAILY_LIMIT: int = 500        # copilot messages per day, all visitors together
    UPLOADS_PER_HOUR: int = 20       # uploads per visitor per hour
    TRAININGS_PER_HOUR: int = 60     # model training / AutoML / explanation runs per visitor per hour

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 20
    MAX_ROWS: int = 200_000
    MAX_COLUMNS: int = 200

    # Session management
    # Delete notebooks unused for this many hours (0 = keep forever, the right choice on your own laptop;
    # a public demo should set e.g. 72 so its disk doesn't fill up)
    SESSION_TTL_HOURS: int = 0
    # Notebooks kept loaded in memory (most recently used); the rest wait in storage. Each can hold
    # a dataset of up to MAX_ROWS rows, so this caps the server's memory use.
    MAX_SESSIONS_IN_MEMORY: int = 30

    # Accounts (Supabase Auth). Your project's URL, e.g. https://abcd.supabase.co. No secret needed:
    # login tokens are checked with the public key Supabase publishes. Empty = anonymous visitors only.
    SUPABASE_URL: str = ""

    # File storage. With the secret key, data files go to a private Supabase Storage bucket instead
    # of DATA_DIR. The secret key can read and write everything: server only, never in the frontend.
    SUPABASE_SECRET_KEY: str = Field("", validation_alias=AliasChoices("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"))
    STORAGE_BUCKET: str = "notebooks"

    # Database. Empty = a local SQLite file in DATA_DIR (private, offline).
    # A PostgreSQL URL (e.g. from Supabase) works without code changes.
    DATABASE_URL: str = ""

    # Data storage directory
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


settings = Settings()
