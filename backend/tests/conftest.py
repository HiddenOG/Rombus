"""Configuration for pytest."""

import sys
import os

import pytest

# Add backend directory to path so 'app' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def no_real_ai(monkeypatch):
    """Tests never call real AI services, whatever the developer's .env says.

    The fallback chain and API keys are blanked; tests that need a provider set it themselves
    and talk to fakes.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "AI_CHAIN", "")
    monkeypatch.setattr(settings, "AI_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    # Usage limits are tested on their own (test_access.py); elsewhere they'd only get in the way
    monkeypatch.setattr(settings, "RATE_LIMITS", False)
    monkeypatch.setattr(settings, "SESSION_TTL_HOURS", 0)
    monkeypatch.setattr(settings, "SUPABASE_URL", "")
    # Never touch the real Supabase database or Storage from tests
    monkeypatch.setattr(settings, "SUPABASE_SECRET_KEY", "")
    monkeypatch.setattr(settings, "DATABASE_URL", "")


VISITOR = "test-visitor-0000000001"


@pytest.fixture
def visitor_headers():
    return {"X-Visitor-Id": VISITOR}
