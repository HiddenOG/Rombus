"""Copilot entry point: sends the chat to whichever AI provider is configured (AI_PROVIDER)."""

from app.config import settings
from app.session import DataSession


async def get_copilot_response(session: DataSession, user_message: str) -> dict:
    if settings.AI_PROVIDER == "ollama":
        from app.ollama_client import get_ollama_response
        return await get_ollama_response(session, user_message)

    from app.gemini import get_gemini_response
    return await get_gemini_response(session, user_message)
