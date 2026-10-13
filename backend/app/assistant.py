"""The notebook assistant: a junior data scientist working alongside the user.

The user leads the analysis. Asked to DO something, the assistant writes the code for one notebook
cell, which the user reads, edits and runs themselves; asked a question, it answers from what's in
the notebook. It never runs anything.

Three kinds of code cell, matching what the app can run and replay:
  look     free code on the training rows (`train`), sets `result`: changes are thrown away
  change   learn(train) / apply(df, learned): replayed on the test rows and on new data
  models   models = {name: sklearn model}: trained and graded like the built-in ones
"""

import ast
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import settings
from app.session import DataSession

logger = logging.getLogger(__name__)

SYSTEM = """
You are Rhombus's assistant: a junior data scientist working alongside an experienced one in a notebook.
They decide the analysis; you help them do it, follow along, and point out problems you notice.

When they ask you to DO something with the data, write the code for ONE notebook cell in a single
```python block, plus at most two short sentences saying what it does. They will read it, edit it if they
want, and run it themselves: never say it has run, and never describe results you haven't seen.
When they ask a QUESTION, answer briefly from the notebook below. No code unless code is the answer.

There are three kinds of cell:
1. LOOK (explore, plot, summarise): the training rows are `train` (also `df`), a copy: changes are thrown
   away. Set `result` to what should be shown (DataFrame, Series, number, text). print() and matplotlib
   figures are shown too.
2. CHANGE (clean, transform, engineer features: anything that must also happen to new data):
       def learn(train):        # look at the TRAINING rows; return a dict of what apply needs
           return {"median": float(train["age"].median())}
       def apply(df, learned):  # change rows using ONLY `learned`; return df
           df["age"] = df["age"].fillna(learned["median"])
           return df
   Put TRAIN_ONLY = True at the top when it only drops or filters training rows (outliers, duplicates).
   apply also runs later on the test rows and on single new rows, so it must not compute statistics from df.
3. MODELS: define `models = {"name": SomeScikitLearnModel(...)}` (imports included). They're trained on the
   training rows and graded on the locked test rows by the final exam.

Rules:
- import pandas as pd / numpy as np at the top; scikit-learn and matplotlib are fine. No files, network or OS.
- Use only column names listed below, spelled exactly.
- Never build a feature from the target column (a new row won't have it).
- The test rows are locked away: you can't see or use them, and neither can the code.
- If what they ask would leak information from the target or the test rows, say so and offer a safe version.
- Short, readable code with brief comments.
""".strip()

MAX_CELLS_IN_CONTEXT = 6


def notebook_context(session: DataSession) -> str:
    """What the assistant knows: the data, the mode, the recipe so far, the latest cells."""
    from app.copilot_prompt import recipe_lines
    state = session.get_summary()
    if state["is_split"]:
        mode = (f"predicting '{state['target']}' ({state['problem_type']}); {state['test_rows']} test rows locked away"
                + (f", by time ('{state['time_column']}': the test rows are the most recent)" if state["time_column"] else ""))
    elif state["clean_only"]:
        mode = "cleaning only: no target and no model; every step works on all rows"
    else:
        mode = "not started: no target chosen yet, so only column names and types are known"
    columns = session.get_compact_column_summary() if session.can_explore else \
        "\n".join(f"{c} ({session.df[c].dtype})" for c in session.df.columns)
    recent = []
    for cell in session.cells[-MAX_CELLS_IN_CONTEXT:]:
        if cell.get("type") == "ask":
            recent.append(f"- asked: {str(cell.get('prompt'))[:200]}")
        elif cell.get("type") == "code":
            recent.append(f"- unrun {cell.get('kind', 'look')} cell:\n{str(cell.get('source', ''))[:600]}")
        elif cell.get("type") == "run":
            code = (cell.get("args") or {}).get("code")
            line = f"- ran {cell.get('tool')}: {str(cell.get('message', ''))[:200]}"
            recent.append(line + (f"\n{code[:600]}" if code else ""))
    return f"""NOTEBOOK
Dataset: {state['filename']} ({state['shape'][0]} rows x {state['shape'][1]} columns)
Mode: {mode}
Trained models: {', '.join(state['trained_models']) or 'none'}
Recipe (in order): {recipe_lines(state['recipe'])}

Columns:
{columns}

Latest cells:
{chr(10).join(recent) or '- none yet'}"""


# ── Talking to the AI providers (plain text: no tools) ───────────────────────────

class ProviderError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


async def _openai_style(server, system: str, messages: List[Dict[str, str]]) -> str:
    if not server.base_url or not server.model:
        raise ProviderError("config")
    if not server.api_key and "localhost" not in server.base_url:
        raise ProviderError("config")
    payload = {"model": server.model, "temperature": 0.2,
               "messages": [{"role": "system", "content": system}, *messages]}
    headers = {"Authorization": f"Bearer {server.api_key}"} if server.api_key else {}
    try:
        async with httpx.AsyncClient(timeout=server.timeout) as client:
            r = await client.post(server.base_url.rstrip("/") + "/chat/completions", json=payload, headers=headers)
    except httpx.ConnectError:
        raise ProviderError("unreachable")
    except httpx.TimeoutException:
        raise ProviderError("timeout")
    if r.status_code in (401, 403):
        raise ProviderError("auth")
    if r.status_code == 429:
        raise ProviderError("busy" if "upstream" in r.text.lower() else "quota")
    if r.status_code >= 400:
        raise ProviderError("busy" if r.status_code >= 500 else "error")
    return r.json()["choices"][0]["message"].get("content") or ""


async def _ollama(system: str, messages: List[Dict[str, str]]) -> str:
    payload = {"model": settings.OLLAMA_MODEL, "stream": False, "think": False, "keep_alive": "30m",
               "options": {"temperature": 0.2, "num_ctx": settings.OLLAMA_CONTEXT},
               "messages": [{"role": "system", "content": system}, *messages]}
    try:
        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
            r = await client.post(f"{settings.OLLAMA_URL}/api/chat", json=payload)
    except httpx.ConnectError:
        raise ProviderError("unreachable")
    except httpx.TimeoutException:
        raise ProviderError("timeout")
    if r.status_code >= 400:
        raise ProviderError("config" if r.status_code == 404 else "error")
    return re.sub(r"<think>.*?</think>", "", r.json().get("message", {}).get("content", ""), flags=re.S)


async def _gemini(system: str, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    from google.genai import errors
    from app.gemini import _get_client, model_chain
    if not settings.GOOGLE_API_KEY:
        raise ProviderError("config")
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
    reason = "busy"
    for model in model_chain():
        try:
            response = await _get_client().aio.models.generate_content(
                model=model, contents=contents, config={"system_instruction": system})
            return response.text or "", model
        except errors.APIError as e:
            if e.code in (401, 403):
                raise ProviderError("auth")
            reason = "quota" if e.code == 429 else reason
            if e.code not in (429, 500, 502, 503, 504):
                raise ProviderError("error")
    raise ProviderError(reason)


async def ask_provider(name: str, system: str, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """One provider's plain-text answer: (text, model)."""
    from app.openai_compat_client import configured_server, groq_server, openrouter_server
    if name in ("groq", "openrouter", "openai_compatible"):
        server = {"groq": groq_server, "openrouter": openrouter_server, "openai_compatible": configured_server}[name]()
        return await _openai_style(server, system, messages), server.model
    if name == "ollama":
        return await _ollama(system, messages), settings.OLLAMA_MODEL
    if name == "gemini":
        return await _gemini(system, messages)
    raise ProviderError("config")


async def complete(system: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Ask the providers in AI_CHAIN order (skipping ones that recently failed) until one answers."""
    from app.copilot import COOLDOWN_SECONDS, _providers, _skip_until, chain
    names = chain()
    labels = {n: p[0] for n, p in _providers().items()}
    ready = [n for n in names if _skip_until.get(n, 0) <= time.time()] or names[-1:]
    tried = []
    for name in ready:
        try:
            text, model = await ask_provider(name, system, messages)
            note = f"Answered by {labels.get(name, name)} because {', '.join(tried)}." if tried else None
            return {"text": text, "provider": labels.get(name, name), "model": model, "fallback_note": note}
        except ProviderError as e:
            if len(names) > 1:
                _skip_until[name] = time.time() + COOLDOWN_SECONDS.get(e.reason, 30)
            tried.append(f"{labels.get(name, name)} was unavailable ({e.reason})")
        except Exception:
            logger.exception("Assistant provider %s failed", name)
            tried.append(f"{labels.get(name, name)} failed")
    return {"failed": True, "text": "The AI couldn't answer right now: " + "; ".join(tried) + ". Please try again shortly."}


# ── Reading the answer ───────────────────────────────────────────────────────────

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def split_reply(text: str) -> Tuple[str, Optional[str]]:
    """(the words, the code of the first python block or None)."""
    match = _CODE_BLOCK.search(text or "")
    if not match:
        return (text or "").strip(), None
    words = (text[:match.start()] + text[match.end():]).strip()
    return words, match.group(1).strip("\n")


def cell_kind(code: str) -> str:
    """Which kind of cell the code is, from its shape."""
    if re.search(r"^def learn\s*\(", code, re.M) and re.search(r"^def apply\s*\(", code, re.M):
        return "change"
    if re.search(r"^models\s*=", code, re.M):
        return "models"
    return "look"


def syntax_error(code: str) -> Optional[str]:
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"


async def assist(session: DataSession, message: str) -> Dict[str, Any]:
    """Answer, or write one cell of code (checked for syntax; one retry if it doesn't parse)."""
    history = [{"role": m["role"], "content": m["content"]} for m in session.memory[-8:]
               if m.get("role") in ("user", "assistant") and m.get("kind") == "assist"]
    messages = [*history, {"role": "user", "content": f"{notebook_context(session)}\n\nREQUEST: {message}"}]
    reply = await complete(SYSTEM, messages)
    if reply.get("failed"):
        return {"failed": True, "reply": reply["text"]}

    words, code = split_reply(reply["text"])
    if code and syntax_error(code):
        retry = await complete(SYSTEM, [*messages, {"role": "assistant", "content": reply["text"]},
                                        {"role": "user", "content": f"That code has a syntax error ({syntax_error(code)}). "
                                                                    "Send the corrected cell."}])
        if not retry.get("failed"):
            fixed_words, fixed = split_reply(retry["text"])
            if fixed and not syntax_error(fixed):
                code, words = fixed, words or fixed_words
    result = {"reply": words or ("Here's the code. Read it, edit it if you like, then run it." if code else ""),
              "provider": reply["provider"], "model": reply["model"], "fallback_note": reply.get("fallback_note")}
    if code:
        result |= {"code": code, "kind": cell_kind(code), "syntax_error": syntax_error(code)}
    return result
