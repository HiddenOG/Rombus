"""Gemini copilot — suggests tool actions via native function calling.

The model never executes tools itself: any function calls it emits are returned to the
frontend as suggested actions, which the user clicks to run through /api/action.
"""

import logging
from typing import Any, Dict, List, Optional

from app.config import settings
from app.copilot_prompt import (
    MAX_SUGGESTIONS,  # noqa: F401  (re-exported for tests)
    STATIC_INSTRUCTIONS,
    finish,
    recent_history,
    state_snapshot,
    to_action,
    tool_specs_for,
)
from app.session import DataSession

logger = logging.getLogger(__name__)

# Lazy-loaded client (avoids import cost/errors when the SDK or key is absent, e.g. in tests)
_client = None


# Tries per model before moving to the next one. Short on purpose: when a model is
# overloaded, switching to a backup beats making the user wait through long retries.
ATTEMPTS_PER_MODEL = 3

# HTTP statuses meaning "this model can't serve you right now, another one might"
_SWITCH_MODEL_STATUSES = {429, 500, 502, 503, 504}

BUSY_MESSAGE = "Google's AI is very busy right now. Please try again in a minute."
QUOTA_MESSAGE = (
    "The free Google AI allowance is used up for now (too many requests). "
    "It resets after a short wait, or daily for the daily limit."
)
KEY_MESSAGE = "The AI couldn't be reached: the Google API key is missing or invalid. Check GOOGLE_API_KEY in backend/.env."
GENERIC_MESSAGE = "Something went wrong while asking the AI. Please try again."


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(
            api_key=settings.GOOGLE_API_KEY,
            http_options={"retry_options": {"attempts": ATTEMPTS_PER_MODEL}},
        )
    return _client


def model_chain() -> List[str]:
    """Main model first, then the backups, without duplicates."""
    backups = [m.strip() for m in settings.GEMINI_FALLBACK_MODELS.split(",") if m.strip()]
    return list(dict.fromkeys([settings.GEMINI_MODEL, *backups]))


def build_tool_declarations(session: Optional[DataSession] = None) -> List[Dict[str, Any]]:
    """Tool schemas in Gemini's function-declaration format."""
    return [
        {"name": s["name"], "description": s["description"], "parameters_json_schema": s["parameters"]}
        for s in tool_specs_for(session)
    ]


def build_system_instruction(session: DataSession) -> str:
    """Static guidance plus a snapshot of the current session state."""
    return f"{STATIC_INSTRUCTIONS}\n\n{state_snapshot(session)}"


def build_contents(session: DataSession, user_message: str) -> List[Dict[str, Any]]:
    """Recent conversation history followed by the current user message, in Gemini's format."""
    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        for m in recent_history(session, user_message)
    ]
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


def parse_response(response) -> Dict[str, Any]:
    """Split a Gemini response into reply text and suggested actions."""
    texts: List[str] = []
    actions: List[Dict[str, Any]] = []

    candidates = response.candidates or []
    parts = (candidates[0].content.parts or []) if candidates and candidates[0].content else []
    for part in parts:
        if part.function_call:
            action = to_action(part.function_call.name, part.function_call.args)
            if action:
                actions.append(action)
        elif part.text and not part.thought:
            texts.append(part.text)
    return finish(texts, actions)


async def get_gemini_response(session: DataSession, user_message: str) -> dict:
    """Get a response from Gemini, falling back through backup models when one is busy."""
    from google.genai import errors

    if not settings.GOOGLE_API_KEY:
        return {"response": KEY_MESSAGE, "suggested_actions": []}

    request = {
        "contents": build_contents(session, user_message),
        "config": {
            "system_instruction": build_system_instruction(session),
            "tools": [{"function_declarations": build_tool_declarations(session)}],
            "automatic_function_calling": {"disable": True},
        },
    }

    message = BUSY_MESSAGE
    for model in model_chain():
        try:
            response = await _get_client().aio.models.generate_content(model=model, **request)
            return parse_response(response)
        except errors.APIError as e:
            logger.warning("Gemini model %s failed: %s %s", model, e.code, e.status)
            # Remember the most telling reason: "out of allowance" beats "busy"
            if e.code == 429:
                message = QUOTA_MESSAGE
            if e.code in (401, 403) or (e.code == 400 and "API key" in str(e)):
                return {"response": KEY_MESSAGE, "suggested_actions": []}
            if e.code not in _SWITCH_MODEL_STATUSES:
                message = GENERIC_MESSAGE
                break
        except Exception:
            logger.exception("Gemini request failed with an unexpected error")
            message = GENERIC_MESSAGE
            break

    return {"response": message, "suggested_actions": []}
