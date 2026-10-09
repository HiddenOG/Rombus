"""Any OpenAI-compatible inference server: vLLM, SGLang, LM Studio, llama.cpp server,
or hosted open-model APIs (Groq, Together, OpenRouter, Fireworks …).

Set AI_PROVIDER=openai_compatible with AI_BASE_URL, AI_MODEL and (for hosted APIs) AI_API_KEY.
Uses the same briefing and reply parsing as the local Ollama client.
"""

import logging

import httpx

from app.config import settings
from app.ollama_client import build_messages, build_tools, parse_message
from app.session import DataSession

logger = logging.getLogger(__name__)

NOT_REACHABLE_MESSAGE = "The inference server couldn't be reached. Check AI_BASE_URL and that the server is running."
KEY_MESSAGE = "The inference server rejected the API key. Check AI_API_KEY in backend/.env."
QUOTA_MESSAGE = "The inference server's rate limit or quota was reached. Please try again shortly."
BUSY_MESSAGE = "The inference server is busy or had an error. Please try again in a moment."
SLOW_MESSAGE = "The inference server took too long to answer. Please try again."
GENERIC_MESSAGE = "Something went wrong while asking the inference server. Please try again."


def _fail(message: str) -> dict:
    return {"response": message, "suggested_actions": []}


async def get_openai_compatible_response(session: DataSession, user_message: str) -> dict:
    """Ask the configured server; never raises, returns a plain-English message on failure."""
    if not settings.AI_BASE_URL or not settings.AI_MODEL:
        return _fail("Set AI_BASE_URL and AI_MODEL in backend/.env to use an inference server.")

    payload = {
        "model": settings.AI_MODEL,
        "messages": build_messages(session, user_message),
        "tools": build_tools(session),
        "tool_choice": "auto",
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {settings.AI_API_KEY}"} if settings.AI_API_KEY else {}
    url = settings.AI_BASE_URL.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=settings.AI_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.ConnectError:
        return _fail(NOT_REACHABLE_MESSAGE)
    except httpx.TimeoutException:
        return _fail(SLOW_MESSAGE)
    except Exception:
        logger.exception("Inference server request failed")
        return _fail(GENERIC_MESSAGE)

    if resp.status_code in (401, 403):
        return _fail(KEY_MESSAGE)
    if resp.status_code == 429:
        return _fail(QUOTA_MESSAGE)
    if resp.status_code >= 500:
        logger.warning("Inference server error %s: %s", resp.status_code, resp.text[:300])
        return _fail(BUSY_MESSAGE)
    if resp.status_code >= 400:
        logger.warning("Inference server refused the request %s: %s", resp.status_code, resp.text[:300])
        return _fail(GENERIC_MESSAGE)

    try:
        message = resp.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError):
        logger.warning("Unexpected inference server reply: %s", resp.text[:300])
        return _fail(GENERIC_MESSAGE)
    return parse_message(message)
