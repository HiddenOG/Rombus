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
import json
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
3. MODELS: define `models = {"name": SomeScikitLearnModel(...)}` (imports included), nothing else needed.
   The app trains them on the cleaned training rows and the final exam grades them on the locked test rows.
   There's no `train` here: only X_learn (features) and y_learn (target), if you must look at them.

Rules:
- import pandas as pd / numpy as np at the top; scikit-learn and matplotlib are fine. No files, network or OS.
- Use only column names listed below, spelled exactly.
- Never build a feature from the target column (a new row won't have it).
- The test rows are locked away: you can't see or use them, and neither can the code.
- If what they ask would leak information from the target or the test rows, say so and offer a safe version.
- Short, readable code with brief comments.

Follow on from the notebook's code, like a colleague continuing someone's notebook:
- Reuse the names already used (variables, helper functions, new columns, keys in `learned`) for the
  same things, and the same style. Don't invent a second name for something that already has one.
- If they edited your earlier code, their version (shown in the notebook) is the one to follow.
- Each cell starts fresh: a variable from an earlier LOOK cell doesn't exist in the next one. When you
  need it again, recompute it with the same name and the same lines, so it reads as one notebook.
- Rename only when they ask or a name has become misleading, and then say so (rename_column for a
  column), so later cells and the recipe use the new name from then on.
""".strip()

LIBRARY_RULES = """
VERIFIED LIBRARY FIRST. The app has built-in steps and views (listed below as name(settings): what it
does). They are tested, replay exactly on the test rows and in the exported script, and can't leak.
When ONE of them does what was asked, don't write code: reply with one sentence and a ```step block:
```step
{"tool": "impute_missing", "arguments": {"column": "bath", "strategy": "median"}}
```
Write a ```python cell only when no built-in does it (a custom rule, an unusual chart, several steps
at once that no single built-in covers). Never both in one reply. Settings: * means required;
a|b|c means pick one of those.
Don't ask whether to go ahead: the user's Run button is the confirmation. Ask a question only when
the request can't be done without an answer (e.g. which of two columns they mean).
A LOOK cell changes nothing: renaming, cleaning, dropping or adding columns must be a built-in step
or a CHANGE cell.
""".strip()

NO_CODE_RULES = """
Code can't run in this notebook (switched off on this server, or it needs a signed-in account). Use a built-in step
whenever one fits; if none does, say so plainly and suggest the closest built-in, or write the code
only if they ask for it (they can run it on their own machine).
""".strip()

# Never in the catalogue: running code (that's what cells are), re-locking the test rows (shopping for a
# lucky split), and moving through the notebook (buttons do that)
NOT_IN_LIBRARY = {"train_test_split", "transition_stage", "undo"}

# Settings that name a column, beyond those described as "Exact column name(s)"
COLUMN_SETTINGS = {"group_by", "order_by", "time_column", "align_on", "hue"}
COLUMN_SETTINGS_FOR = {("plot_line", "y"), ("pivot_table", "columns"), ("pivot_table", "rows"),
                       ("pivot_table", "value"), ("plot_correlation", "value"), ("group_summary", "value")}
# Settings that take several columns comma-separated in one string
COMMA_LISTS = {("group_summary", "value")}

HISTORY_TURNS = 8  # the last exchanges with the assistant, word for word (older ones live on in the notebook)


def library(session: DataSession) -> List[str]:
    """The built-ins the assistant may propose right now."""
    from app.tools import HIDDEN_FROM_AI, available_tools
    names = [n for n in available_tools(session) if n not in HIDDEN_FROM_AI and n not in NOT_IN_LIBRARY]
    if session.is_split:  # changing the target or the mode is a deliberate edit, not a suggestion
        names = [n for n in names if n not in ("select_target", "clean_only")]
    return names


def catalogue(session: DataSession) -> str:
    """One line per built-in: name(settings): description. Compact, so small models read it quickly."""
    from app.tools import TOOL_SPECS
    from app.tools.model import CLASSIFICATION_MODELS, REGRESSION_MODELS
    lines = []
    for name in library(session):
        spec = TOOL_SPECS[name]
        required = set(spec["parameters"]["required"])
        settings_ = []
        for key, prop in spec["parameters"]["properties"].items():
            enum = prop.get("enum") or (prop.get("items") or {}).get("enum")
            if name == "train_models" and key == "algorithms" and session.problem_type:
                enum = sorted(CLASSIFICATION_MODELS if session.problem_type == "classification" else REGRESSION_MODELS)
            shown = key + ("*" if key in required else "")
            if enum:
                shown += ": " + "|".join(map(str, enum)) + (" (list)" if prop.get("type") == "array" else "")
            elif prop.get("type") == "array":
                shown += " (list)"
            elif key in FREE_TEXT:  # the syntax lives in the setting's own description
                shown += f" [{_first_sentence(prop.get('description', ''))}]"
            settings_.append(shown)
        description = _first_sentence(spec["description"])
        lines.append(f"- {name}({', '.join(settings_)}): {description}")
    return "\n".join(lines)


FREE_TEXT = {"conditions", "formula", "value", "into", "pattern", "characters", "find"}


def _first_sentence(text: str) -> str:
    """Up to the first full stop that isn't part of 'e.g.' / 'i.e.'."""
    return re.split(r"(?<!e\.g)(?<!i\.e)\.\s", text.strip(), maxsplit=1)[0].rstrip(".")


def _is_column_setting(tool: str, key: str, prop: Dict[str, Any]) -> bool:
    described = str(prop.get("description", "")).lower().startswith("exact column name")
    return described or key in COLUMN_SETTINGS or (tool, key) in COLUMN_SETTINGS_FOR


def check_step(session: DataSession, step: Any, request: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Check a proposed built-in before anyone sees it: (step, None) when it's sound, else (None, why).

    The tool must be in the library now, settings known, required ones present, choices valid, column
    names real, conditions and formulas readable, and judgement steps only when they were asked for."""
    from app.expressions import parse_filter
    from app.intent import asked_for
    from app.recipe import STEP_TYPES, Line, run_line
    from app.tools import TOOL_SPECS
    if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
        return None, 'the step block must be JSON like {"tool": "...", "arguments": {...}}'
    # Settings written beside "tool" instead of inside "arguments" (models do this) are still settings
    loose = {k: v for k, v in step.items() if k not in ("tool", "arguments", "explanation")}
    tool, args = step["tool"], step.get("arguments") or loose
    if tool not in TOOL_SPECS or tool not in library(session):
        return None, f"'{tool}' isn't a built-in available right now"
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object"
    params = TOOL_SPECS[tool]["parameters"]
    args = {k: v for k, v in args.items() if v not in (None, "")}
    unknown = [k for k in args if k not in params["properties"]]
    if unknown:
        return None, f"{tool} has no setting {', '.join(unknown)} (its settings: {', '.join(params['properties']) or 'none'})"
    missing = [k for k in params["required"] if k not in args]
    if missing:
        return None, f"{tool} needs {', '.join(missing)}"

    columns = [str(c) for c in session.df.columns]
    for key, value in args.items():
        prop = params["properties"][key]
        enum = prop.get("enum") or (prop.get("items") or {}).get("enum")
        values = value if isinstance(value, list) else [value]
        if prop.get("type") == "array" and not isinstance(value, list):
            args[key] = values = [v.strip() for v in str(value).split(",")] if isinstance(value, str) else [value]
        if enum and tool != "train_models":  # algorithm names have aliases; the tool itself checks those
            bad = [v for v in values if v not in enum]
            if bad:
                return None, f"{key} must be one of {', '.join(map(str, enum))} (got {', '.join(map(str, bad))})"
        if _is_column_setting(tool, key, prop):
            names = [n.strip() for v in values for n in (str(v).split(",") if (tool, key) in COMMA_LISTS else [str(v)])]
            unknown_cols = [n for n in names if n not in columns]
            if unknown_cols:
                return None, f"no column named {', '.join(unknown_cols)} (columns: {', '.join(columns)})"

    try:
        if tool in STEP_TYPES:
            # A dry run on a copy of the training rows: every error the step itself would raise (a formula
            # using the target, an unknown column in a condition, a one-hot too big) is caught here
            if tool == "time_feature" and not args.get("order_by") and session.time_column:
                args["order_by"] = session.time_column
            run_line(Line(tool, dict(args)), session.df.copy(), session.ctx)
        elif tool == "filter_rows":
            parse_filter(session.df.head(200), str(args["conditions"]))
    except ValueError as e:
        return None, str(e)
    except Exception:
        logger.exception("Dry run of a proposed %s failed", tool)
        return None, "it fails on this data"

    if not asked_for(tool, request):
        return None, (f"{tool} is a judgement call the user didn't ask for: only propose it when they ask for "
                      "that kind of change")
    return {"tool": tool, "arguments": args}, None


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
    names = names_in_use(session)
    return f"""NOTEBOOK
Dataset: {state['filename']} ({state['shape'][0]} rows x {state['shape'][1]} columns)
Mode: {mode}
Trained models: {', '.join(state['trained_models']) or 'none'}
Recipe (in order): {recipe_lines(state['recipe'])}

Columns:
{columns}

What they've asked for so far (oldest first):
{chr(10).join(f'- {r}' for r in past_requests(session)) or '- nothing yet'}

Names already used in the notebook's code: {', '.join(names) or 'none yet'}

The notebook so far, cell by cell (their edits included: this is the code as it is now):
{notebook_code(session) or '(no cells yet)'}"""


# ── Memory of the notebook: its code, names and requests ────────────────────────

CODE_BUDGET = 12_000  # characters of past code shown in full; older cells beyond it are summarised
MAX_REQUESTS = 20

# Names every cell has anyway: not worth listing as "in use"
GIVEN_NAMES = {"train", "df", "result", "learned", "models", "learn", "apply", "pd", "np", "plt", "sns",
               "TRAIN_ONLY", "X_learn", "y_learn"}


def _cell_code(cell: Dict[str, Any]) -> Optional[str]:
    if cell.get("type") == "code":
        return str(cell.get("source") or "")
    if cell.get("type") == "run":
        return (cell.get("args") or {}).get("code")
    return None


def notebook_code(session: DataSession) -> str:
    """Every cell in order, numbered: code in full (newest first within the budget), built-in steps and
    requests as one line each. Older code past the budget is summarised by the names it defines."""
    entries, used = [], 0
    for n, cell in reversed(list(enumerate(session.cells, 1))):
        kind = cell.get("type")
        code = _cell_code(cell)
        if code is not None:
            what = cell.get("kind") or {"custom_view": "look", "custom_code": "change",
                                        "custom_models": "models"}.get(cell.get("tool"), "code")
            state = "not run yet" if kind == "code" else "ran"
            if used + len(code) <= CODE_BUDGET:
                used += len(code)
                entries.append(f"[{n}] {what} cell ({state}):\n```python\n{code.strip()}\n```")
            else:
                defined = ", ".join(_defined_names(code)) or "nothing named"
                entries.append(f"[{n}] {what} cell ({state}), {len(code.splitlines())} lines; defines: {defined}")
        elif kind == "run":
            args = ", ".join(f"{k}={v}" for k, v in (cell.get("args") or {}).items())
            entries.append(f"[{n}] built-in {cell.get('tool')}({args}): {str(cell.get('message', ''))[:160]}")
        elif kind == "ask":
            entries.append(f"[{n}] they asked: {str(cell.get('prompt'))[:200]}")
    return "\n".join(reversed(entries))


def _defined_names(code: str) -> List[str]:
    """Variables, functions and new columns a piece of code defines (in order, no repeats)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found.append(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.append(node.id)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            found.append(f'["{node.slice.value}"]')  # df["pps"] = ...: a column (or a key) it makes
        elif isinstance(node, ast.Dict):  # the keys of what learn() returns
            found += [f'"{k.value}"' for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    return [n for n in dict.fromkeys(found) if n not in GIVEN_NAMES and not n.startswith("_")]


def names_in_use(session: DataSession) -> List[str]:
    names: List[str] = []
    for cell in session.cells:
        code = _cell_code(cell)
        if code:
            names += _defined_names(code)
    return list(dict.fromkeys(names))[:80]


def past_requests(session: DataSession) -> List[str]:
    """What the user asked the assistant for, oldest first (the notebook's ask cells, else memory)."""
    asked = [str(c.get("prompt")) for c in session.cells if c.get("type") == "ask" and c.get("prompt")]
    if not asked:
        asked = [m["content"] for m in session.memory if m.get("role") == "user" and m.get("kind") == "assist"]
    return [a[:200] for a in asked[-MAX_REQUESTS:]]


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

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_STEP_BLOCK = re.compile(r"```(?:step|json)\s*(\{.*?\})\s*```", re.S)  # on its own lines or on one line


def split_reply(text: str) -> Tuple[str, Optional[str]]:
    """(the words, the code of the first python block or None)."""
    match = _CODE_BLOCK.search(text or "")
    if not match:
        return (text or "").strip(), None
    words = (text[:match.start()] + text[match.end():]).strip()
    return words, match.group(1).strip("\n")


def split_step(text: str) -> Tuple[str, Any]:
    """(the words without the step block, the parsed step, or the raw text if it isn't JSON, or None)."""
    match = _STEP_BLOCK.search(text or "")
    if not match:
        return text or "", None
    words = (text[:match.start()] + text[match.end():]).strip()
    try:
        return words, json.loads(match.group(1))
    except ValueError:
        return words, match.group(1)


# Enough rows to meet the odd values ("188.89Sq. Yards" sat past row 2,000 in the Bengaluru data)
DRY_RUN_ROWS = 50_000


def code_problem(session: DataSession, code: str) -> Optional[str]:
    """Why this code won't work, found by a syntax check and a trial run on a copy of (some of) the
    training rows. None when it runs. The notebook itself never changes."""
    from app import sandbox
    error = syntax_error(code)
    if error:
        return f"a syntax error ({error})"
    if sandbox.not_allowed(session):
        return None  # code can't run for this notebook: nothing more to check
    rows = session.df.head(DRY_RUN_ROWS)
    kind = cell_kind(code)
    if kind == "models":
        if not session.target_column or session.target_column not in rows:
            return None
        frames = {"X_learn": rows.drop(columns=[session.target_column]), "y_learn": rows[[session.target_column]]}
    else:
        frames = {"train": rows}
    try:
        sandbox.run({"kind": "check", "cell": kind, "code": code, "frames": frames})  # in the sandbox, like Run
    except ValueError as e:
        return f"an error when run: {e}"
    return None


def read_reply(session: DataSession, text: str, request: str) -> Dict[str, Any]:
    """What an answer holds: words, and a checked built-in step OR a code cell.
    {"words", "step"?, "code"?, "problem"?}: problem is why a proposed step can't be used."""
    words, step = split_step(text)
    words, code = split_reply(words)
    if step is None:
        return {"words": words, "code": code}
    checked, problem = check_step(session, step, request)
    if checked:
        return {"words": words, "step": checked}  # the verified step wins over any code alongside it
    return {"words": words, "code": code, "problem": problem}


def examine(session: DataSession, text: str, request: str) -> Dict[str, Any]:
    """read_reply, plus a trial run of any code (blocking: called on the notebook's worker thread)."""
    answer = read_reply(session, text, request)
    if answer.get("code") and not answer.get("problem"):
        answer["code_problem"] = code_problem(session, answer["code"])
    return answer


def _briefing(session: DataSession) -> Tuple[str, str]:
    return notebook_context(session), system_prompt(session)


def _request_with_context(session: DataSession, message: str) -> str:
    """The request, plus the assistant's previous answer when this is a short "yes / do it"."""
    from app.intent import FOLLOW_UP
    if FOLLOW_UP.match(message) and len(message.split()) <= 6:
        earlier = [m["content"] for m in session.memory if m.get("role") == "assistant" and m.get("kind") == "assist"]
        return message + " " + (earlier[-1] if earlier else "")
    return message


def system_prompt(session: DataSession) -> str:
    parts = [SYSTEM, LIBRARY_RULES]
    if session.target_column:
        target = session.target_column
        parts.append(f"THE TARGET IS '{target}'.\n"
                     f"- A new COLUMN (feature) made from '{target}' leaks: a new row won't have its {target}. If asked "
                     f"for one, propose nothing (not even a step that prepares for it); explain why in a sentence and "
                     f"offer the safe version.\n"
                     f"- Using '{target}' to choose which TRAINING rows to drop is fine and common (outliers in "
                     f"{target} per sqft, rules comparing groups): remove_outliers, drop_rows_where, or a TRAIN_ONLY "
                     f"change cell. Do those normally.")
    from app.sandbox import not_allowed
    if not_allowed(session):
        parts.append(NO_CODE_RULES)
    parts.append("BUILT-INS AVAILABLE NOW:\n" + catalogue(session))
    return "\n\n".join(parts)


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
    """Answer, propose a verified built-in step, or write one cell of code.

    A proposed built-in is checked before anyone sees it, and code is checked for syntax; either gets
    one retry, told exactly what was wrong."""
    from app.session import run_exclusive
    # Reading the notebook and checking answers (dry runs, trial runs of code) is blocking work: it runs
    # on the notebook's worker thread, under its lock, so the server keeps serving everyone else
    history = [{"role": m["role"], "content": m["content"]} for m in session.memory
               if m.get("role") in ("user", "assistant") and m.get("kind") == "assist"][-HISTORY_TURNS * 2:]
    context, system = await run_exclusive(session.session_id, _briefing, session)
    messages = [*history, {"role": "user", "content": f"{context}\n\nREQUEST: {message}"}]
    request = _request_with_context(session, message)
    reply = await complete(system, messages)
    if reply.get("failed"):
        return {"failed": True, "reply": reply["text"]}

    answer = await run_exclusive(session.session_id, examine, session, reply["text"], request)
    trouble =(f"That step can't be used: {answer['problem']}. Send a corrected ```step block, or a ```python "
               "cell if no built-in fits." if answer.get("problem")
               else f"That code has {answer['code_problem']}. Send the corrected cell."
               if answer.get("code_problem") else None)
    if trouble:
        retry = await complete(system, [*messages, {"role": "assistant", "content": reply["text"]},
                                        {"role": "user", "content": trouble}])
        if not retry.get("failed"):
            second = await run_exclusive(session.session_id, examine, session, retry["text"], request)
            fixed = second.get("step") or (second.get("code") and not second.get("code_problem"))
            if fixed:
                # Its first words described the broken version only if there were none the second time
                answer = {**second, "words": second["words"] or answer["words"]}

    result = {"provider": reply["provider"], "model": reply["model"], "fallback_note": reply.get("fallback_note")}
    words = answer["words"]
    if answer.get("step"):
        return result | {"reply": words or "There's a built-in step for this.", "step": answer["step"]}
    if answer.get("problem") and not answer.get("code"):
        # Still no usable step: say so instead of showing a button that would fail
        note = f"(I tried to propose a built-in step, but it doesn't fit: {answer['problem']}.)"
        return result | {"reply": f"{words}\n\n{note}".strip()}
    code = answer.get("code")
    result["reply"] = words or ("Here's the code. Read it, edit it if you like, then run it." if code else "")
    if code:
        result |= {"code": code, "kind": cell_kind(code), "problem": answer.get("code_problem")}
    return result
