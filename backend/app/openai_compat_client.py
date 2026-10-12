"""Any OpenAI-compatible inference server: vLLM, SGLang, LM Studio, llama.cpp server,
or hosted open-model APIs (Groq, Together, OpenRouter, Fireworks …).

Used directly with AI_PROVIDER=openai_compatible (AI_BASE_URL / AI_MODEL / AI_API_KEY), and by
the fallback chain for named providers like Groq and OpenRouter.
Uses the same briefing and reply parsing as the local Ollama client.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import settings
from app.copilot_prompt import failure, promises_action
from app.ollama_client import build_messages, build_tools, parse_message
from app.session import DataSession

logger = logging.getLogger(__name__)


@dataclass
class Server:
    label: str      # shown to people, e.g. "Groq"
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 120


def configured_server() -> Server:
    return Server("Inference server", settings.AI_BASE_URL, settings.AI_MODEL, settings.AI_API_KEY,
                  settings.AI_TIMEOUT_SECONDS)


def groq_server() -> Server:
    return Server("Groq", "https://api.groq.com/openai/v1", settings.GROQ_MODEL, settings.GROQ_API_KEY,
                  settings.AI_TIMEOUT_SECONDS)


def openrouter_server() -> Server:
    return Server("OpenRouter", "https://openrouter.ai/api/v1", settings.OPENROUTER_MODEL, settings.OPENROUTER_API_KEY,
                  settings.AI_TIMEOUT_SECONDS)


async def get_openai_compatible_response(session: DataSession, user_message: str, server: Optional[Server] = None) -> dict:
    """Ask the server; never raises. Failures come back with a `failure` reason for the chain."""
    server = server or configured_server()
    name = server.label
    if not server.base_url or not server.model:
        return failure(f"{name} isn't set up: its address or model name is missing in backend/.env.", "config")
    if not server.api_key and "localhost" not in server.base_url and "127.0.0.1" not in server.base_url:
        return failure(f"{name} needs an API key in backend/.env.", "config")

    payload = {
        "model": server.model,
        "messages": build_messages(session, user_message),
        "tools": build_tools(session),
        "tool_choice": "auto",
        "temperature": 0.2,
    }
    reply = await _ask(server, payload)
    if reply.get("failure") or reply["suggested_actions"] or not promises_action(reply["response"]):
        return reply

    # It announced an action but didn't suggest one ("Let me run it now:" and nothing else).
    # Ask once more, this time requiring a function call, and keep its original words.
    retry = await _ask(server, {**payload, "tool_choice": "required"})
    if not retry.get("failure") and retry["suggested_actions"]:
        reply["suggested_actions"] = retry["suggested_actions"]
        reply["usage"] = {k: (reply["usage"].get(k) or 0) + (retry["usage"].get(k) or 0) for k in reply["usage"]}
    else:
        reply["notice"] = "_(No step came with this answer. You can add it yourself with **Add step**, or ask again.)_"
    return reply


async def _ask(server: Server, payload: dict) -> dict:
    """One request to the server, parsed into reply text and suggested actions (or a failure)."""
    name = server.label
    headers = {"Authorization": f"Bearer {server.api_key}"} if server.api_key else {}
    url = server.base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=server.timeout) as client:
            # A hard limit on the whole request: some servers send keep-alive whitespace while
            # thinking, which would otherwise reset httpx's "no data received" timer forever
            resp = await asyncio.wait_for(client.post(url, json=payload, headers=headers), timeout=server.timeout)
    except httpx.ConnectError:
        return failure(f"{name} couldn't be reached. Check the address and that it's running.", "unreachable")
    except (httpx.TimeoutException, asyncio.TimeoutError):
        return failure(f"{name} took too long to answer. Please try again.", "timeout")
    except Exception:
        logger.exception("%s request failed", name)
        return failure(f"Something went wrong while asking {name}. Please try again.", "error")

    if resp.status_code in (401, 403):
        return failure(f"{name} rejected the API key. Check it in backend/.env.", "auth")
    if resp.status_code == 429:
        # Shared free models can be overloaded "upstream": that's busy, not your allowance running out
        if "upstream" in resp.text.lower() or "temporarily" in resp.text.lower():
            return failure(f"{name}'s free model is overloaded right now. Please try again shortly.", "busy")
        return failure(f"{name}'s free allowance or rate limit was reached. Please try again shortly.", "quota")
    if resp.status_code >= 500:
        logger.warning("%s error %s: %s", name, resp.status_code, resp.text[:300])
        return failure(f"{name} is busy or had an error. Please try again in a moment.", "busy")
    if resp.status_code >= 400:
        logger.warning("%s refused the request %s: %s", name, resp.status_code, resp.text[:300])
        return failure(f"{name} couldn't handle the request.", "error")

    try:
        body = resp.json()
        message = body["choices"][0]["message"]
    except (ValueError, KeyError, IndexError):
        logger.warning("Unexpected reply from %s: %s", name, resp.text[:300])
        return failure(f"{name} sent a reply the app couldn't read.", "error")

    reply = parse_message(message)
    usage = body.get("usage") or {}
    reply["usage"] = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
    return reply
