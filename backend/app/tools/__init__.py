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
        return {
            "success": False,
            "message": "First choose what to do: pick the column to predict (Select target, which locks away "
                       "the test rows), or Clean only, to clean the data without a model. Until then you can "
                       "only preview a few rows.",
        }

    if tool_name in ("custom_view", "custom_code", "custom_models"):
        from app.sandbox import not_allowed
        reason = not_allowed(session)
        if reason:
            return {"success": False, "message": reason}

    # An argument sent as null (LLMs often do this) means "not provided": let the tool use its default
    args = {k: v for k, v in (args or {}).items() if v is not None}

    try:
        args = _typed(tool_name, args)
        return TOOL_REGISTRY[tool_name](session, args)
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
