"""Copilot entry point: one provider (AI_PROVIDER), or a fallback chain (AI_CHAIN).

With AI_CHAIN=groq,gemini,ollama the providers are tried in order until one answers. A provider
that fails is skipped for a while (how long depends on why it failed), so the app doesn't keep
knocking on a closed door. Every answer says which provider and model gave it.
"""

import logging
import time
from typing import Awaitable, Callable, Dict, List, Tuple

from app.config import settings
from app.copilot_prompt import failure
from app.intent import asked_for, obvious_step
from app.session import DataSession

logger = logging.getLogger(__name__)

Ask = Callable[[DataSession, str], Awaitable[dict]]

# How long to skip a provider after it fails, by reason (seconds)
COOLDOWN_SECONDS = {
    "quota": 600,        # out of allowance: wait for it to refill
    "busy": 60,
    "unreachable": 120,
    "timeout": 30,
    "error": 30,
    "auth": 3600,        # a wrong key won't fix itself
    "config": 3600,
}

_skip_until: Dict[str, float] = {}


def _providers() -> Dict[str, Tuple[str, Ask, Callable[[], str]]]:
    """name -> (label for people, ask function, current model name)."""
    from app.gemini import get_gemini_response
    from app.ollama_client import get_ollama_response
    from app.openai_compat_client import (
        configured_server, get_openai_compatible_response, groq_server, openrouter_server,
    )

    return {
        "groq": ("Groq", lambda s, m: get_openai_compatible_response(s, m, groq_server()), lambda: settings.GROQ_MODEL),
        "openrouter": ("OpenRouter", lambda s, m: get_openai_compatible_response(s, m, openrouter_server()),
                       lambda: settings.OPENROUTER_MODEL),
        "openai_compatible": ("Inference server", lambda s, m: get_openai_compatible_response(s, m, configured_server()),
                              lambda: settings.AI_MODEL),
        "gemini": ("Gemini", get_gemini_response, lambda: settings.GEMINI_MODEL),
        "ollama": ("Local Qwen (Ollama)", get_ollama_response, lambda: settings.OLLAMA_MODEL),
    }


def chain() -> List[str]:
    names = [p.strip().lower() for p in settings.AI_CHAIN.split(",") if p.strip()]
    return names or [settings.AI_PROVIDER]


async def get_copilot_response(session: DataSession, user_message: str) -> dict:
    providers = _providers()
    names = chain()
    unknown = [n for n in names if n not in providers]
    if unknown:
        return failure(f"Unknown AI provider(s) in AI_CHAIN: {', '.join(unknown)}. "
                       f"Use: {', '.join(providers)}.", "config")

    now = time.time()
    # Skip providers that recently failed, but always try at least the last one in the chain
    ready = [n for n in names if _skip_until.get(n, 0) <= now] or names[-1:]
    tried: List[str] = []
    last: dict = {}
    for name in ready:
        label, ask, model = providers[name]
        reply = await ask(session, user_message)
        reason = reply.get("failure")
        if not reason:
            reply["provider"] = label
            reply.setdefault("model", model())
            # Judgement-call steps (outliers, dropping rows, new features...) only when the user asked
            suggested = reply.get("suggested_actions") or []
            kept = [a for a in suggested if asked_for(a["tool"], user_message, session)]
            if len(kept) < len(suggested):
                logger.info("Dropped unrequested suggestions: %s", [a["tool"] for a in suggested if a not in kept])
                reply["suggested_actions"] = kept
            if not reply.get("suggested_actions"):
                # The safety net: an obvious request still gets its Run button (see app/intent.py)
                step = obvious_step(session, user_message)
                if step:
                    reply["suggested_actions"] = [step]
            if tried:
                reply["fallback_note"] = f"Answered by {label} because {', '.join(tried)}."
            return reply
        if len(names) > 1:
            _skip_until[name] = time.time() + COOLDOWN_SECONDS.get(reason, 30)
        logger.warning("Provider %s failed (%s); trying the next one", name, reason)
        tried.append(f"{label} was unavailable ({reason})")
        last = reply

    if len(tried) > 1:
        return failure(f"None of the AI providers could answer: {'; '.join(tried)}. Please try again later.",
                       last.get("failure", "error"))
    return last
