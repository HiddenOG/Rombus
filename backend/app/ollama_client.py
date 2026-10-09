"""Local copilot via Ollama (e.g. Qwen 3.5): free, private, no rate limits.

Message layout keeps the start of every request identical (system instructions, tools,
earlier messages) so Ollama can reuse its cache and only read the new part. The changing
session state rides along with the newest user message instead of the system prompt.
"""

import json
import logging
import re
from typing import Any, Dict, List

import httpx

from app.config import settings
from app.copilot_prompt import STATIC_INSTRUCTIONS, finish, recent_history, state_snapshot, to_action, tool_specs_for
from app.session import DataSession

logger = logging.getLogger(__name__)

NOT_RUNNING_MESSAGE = "The local AI isn't running. Start Ollama (open the Ollama app or run `ollama serve`) and try again."
SLOW_MESSAGE = "The local AI took too long to answer. Try again; the first message after starting is the slowest."
GENERIC_MESSAGE = "Something went wrong while asking the local AI. Please try again."

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def missing_model_message(model: str) -> str:
    return f"The local model '{model}' isn't downloaded yet. Run `ollama pull {model}` in a terminal."


def build_messages(session: DataSession, user_message: str) -> List[Dict[str, str]]:
    """System instructions, earlier messages, then the current state + new message."""
    return [
        {"role": "system", "content": STATIC_INSTRUCTIONS},
        *recent_history(session, user_message),
        {"role": "user", "content": f"{state_snapshot(session)}\n\nUSER MESSAGE: {user_message}"},
    ]


def build_tools(session: DataSession) -> List[Dict[str, Any]]:
    """Tool schemas in Ollama's (OpenAI-style) format."""
    return [
        {"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
        for s in tool_specs_for(session)
    ]


def parse_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Split an Ollama chat message into reply text and suggested actions."""
    actions = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):  # some models send arguments as a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                logger.warning("Unreadable tool arguments from local model: %r", args)
                continue
        action = to_action(fn.get("name", ""), args)
        if action:
            actions.append(action)
    text = _THINK_BLOCK.sub("", message.get("content") or "")
    return finish([text], actions)


async def get_ollama_response(session: DataSession, user_message: str) -> dict:
    """Ask the local model; never raises, returns a plain-English message on failure."""
    model = settings.OLLAMA_MODEL
    payload = {
        "model": model,
        "messages": build_messages(session, user_message),
        "tools": build_tools(session),
        "stream": False,
        "think": False,          # skip slow step-by-step reasoning; answers here are short
        "keep_alive": "30m",     # keep the model loaded between messages
        "options": {"temperature": 0.2, "num_ctx": settings.OLLAMA_CONTEXT},
    }
    try:
        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{settings.OLLAMA_URL}/api/chat", json=payload)
        if resp.status_code == 404:
            return {"response": missing_model_message(model), "suggested_actions": []}
        resp.raise_for_status()
        return parse_message(resp.json().get("message", {}))
    except httpx.ConnectError:
        return {"response": NOT_RUNNING_MESSAGE, "suggested_actions": []}
    except httpx.TimeoutException:
        logger.warning("Local model %s timed out after %ss", model, settings.OLLAMA_TIMEOUT_SECONDS)
        return {"response": SLOW_MESSAGE, "suggested_actions": []}
    except Exception:
        logger.exception("Local model request failed")
        return {"response": GENERIC_MESSAGE, "suggested_actions": []}
