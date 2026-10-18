"""Tool registry — maps tool names to handler functions and their schemas."""

import logging
import math
from typing import Dict, Callable, Any, List, Optional, Set
import pandas as pd
from app.session import DataSession

logger = logging.getLogger(__name__)


# Registry of all available tools
TOOL_REGISTRY: Dict[str, Callable] = {}

# JSON-schema specs for each tool, used to build Gemini function declarations
TOOL_SPECS: Dict[str, Dict[str, Any]] = {}

# Tools usable before the test rows are locked away (strict mode: no statistics, no changes)
BEFORE_SPLIT_TOOLS: Set[str] = set()

# What kind of thing each tool does, for the notebook:
#   view    looks only (charts, tables) and can be safely re-run on the current data
#   action  changes the session but isn't a recipe step (choose target, train, undo)
#   fixed / learn / train_only   recipe steps (see app/recipe.py)
TOOL_KINDS: Dict[str, str] = {}

# Tools people can use but the AI is never offered (e.g. running arbitrary code)
HIDDEN_FROM_AI: Set[str] = set()


def register_tool(
    name: str,
    description: str,
    properties: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
    before_split: bool = False,
    kind: str = "view",
    ai: bool = True,
):
    """Decorator to register a tool function along with its argument schema."""
    def decorator(func):
        TOOL_REGISTRY[name] = func
        TOOL_SPECS[name] = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        }
        if before_split:
            BEFORE_SPLIT_TOOLS.add(name)
        TOOL_KINDS[name] = kind
        if not ai:
            HIDDEN_FROM_AI.add(name)
        return func
    return decorator


# Tools that build or use a model: they need a target (and so locked test rows)
NEEDS_TARGET: Set[str] = {"train_test_split", "train_models", "auto_ml", "final_exam", "predict",
                          "explain_model", "save_model", "custom_models"}


DEFAULT_SEAL = 0.2  # rows sealed when someone starts exploring without choosing


def available_tools(session: DataSession) -> List[str]:
    """Tools the session may use right now."""
    if session.is_split:
        return list(TOOL_REGISTRY)
    if session.clean_only:
        return [t for t in TOOL_REGISTRY if t not in NEEDS_TARGET]
    return [t for t in TOOL_REGISTRY if t in BEFORE_SPLIT_TOOLS]


def execute_tool(tool_name: str, session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered tool by name."""
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "message": f"Unknown tool: {tool_name}"}

    if tool_name not in available_tools(session):
        if session.clean_only:
            return {
                "success": False,
                "message": "This notebook is in clean-only mode, so there's no model yet. To build one, choose "
                           "the column to predict with Select target: your cleaning steps are kept and "
                           "replayed on the training rows.",
            }
        if tool_name in NEEDS_TARGET:
            return {
                "success": False,
                "message": "Models need a column to predict: choose it with Predict a column (Select target), "
                           "which locks away the test rows first so the final score is honest.",
            }
        # Nothing chosen yet, and they've started working: that's exploring. No gate: a share of the rows
        # is sealed first (the default 20%), so a column predicted later still gets an honest test set
        started = session.start_exploring(DEFAULT_SEAL)
        result = execute_tool(tool_name, session, args)
        result["message"] = f"({started}) " + str(result.get("message", ""))
        result["started_exploring"] = True
        return result

    if tool_name in ("custom_view", "custom_code", "custom_models"):
        from app.sandbox import not_allowed
        reason = not_allowed(session)
        if reason:
            return {"success": False, "message": reason}

    # An argument sent as null (LLMs often do this) means "not provided": let the tool use its default
    args = {k: v for k, v in (args or {}).items() if v is not None}

    try:
        args = _typed(tool_name, args)
        return _with_column_order(TOOL_REGISTRY[tool_name](session, args))
    except ValueError as e:
        # Our own validation errors (bad column name, test size out of range) are user-facing
        return {"success": False, "message": str(e)}
    except Exception:
        logger.exception("Tool %s crashed with arguments %s", tool_name, args)
        return {
            "success": False,
            "message": f"Something went wrong while running '{tool_name}'. "
                       "The details were logged on the server; try different settings or undo the last step.",
        }


def _with_column_order(result: Dict[str, Any]) -> Dict[str, Any]:
    """A table's columns in their real order, sent alongside it: a browser lists number-like keys ("1", "2")
    before the others whatever order they came in, so a pivot table's row labels would end up last."""
    evidence = result.get("evidence") if isinstance(result, dict) else None
    preview = evidence.get("preview") if isinstance(evidence, dict) else None
    if preview and isinstance(preview, list) and isinstance(preview[0], dict):
        evidence.setdefault("preview_columns", list(preview[0]))
    return result


def _typed(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Settings in the types the tool expects: "5" -> 5 for numbers, "rf, gb" -> ["rf", "gb"] for lists.
    Raises ValueError with a plain message for what can't be read ("many" rows)."""
    properties = TOOL_SPECS[tool_name]["parameters"]["properties"]
    typed = dict(args)
    for key, value in args.items():
        kind = properties.get(key, {}).get("type")
        if kind in ("integer", "number") and not isinstance(value, bool):
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float("nan")
            if not math.isfinite(number):
                raise ValueError(f"'{key}' must be a number (got {value!r}).")
            typed[key] = int(number) if kind == "integer" and number == int(number) else number
        elif kind == "array" and isinstance(value, str):
            typed[key] = [v.strip() for v in value.split(",") if v.strip()]
    return typed


def validate_column(session: DataSession, col: str) -> None:
    """Validate that a column exists in the session DataFrame."""
    if col not in session.df.columns:
        available = ", ".join(session.df.columns.tolist())
        raise ValueError(f"Column '{col}' not found. Available columns: {available}")


def categorical_columns(df: pd.DataFrame) -> List[str]:
    """Columns holding text or categories (object, pandas 3 `str`, or category dtype)."""
    return [
        c for c in df.columns
        if isinstance(df[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
    ]


# Reusable schema fragments
COLUMN = {"type": "string", "description": "Exact column name."}
COLUMNS = {"type": "array", "items": {"type": "string"}, "description": "Exact column names."}


# Import all tool modules to trigger registration
from app.tools import acquire, scrub, explore, analysis, model, workflow, custom  # noqa: E402,F401
