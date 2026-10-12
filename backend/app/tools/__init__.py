"""Tool registry — maps tool names to handler functions and their schemas."""

import logging
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


def available_tools(session: DataSession) -> List[str]:
    """Tools the session may use right now."""
    return list(TOOL_REGISTRY) if session.is_split else [t for t in TOOL_REGISTRY if t in BEFORE_SPLIT_TOOLS]


def execute_tool(tool_name: str, session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered tool by name."""
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "message": f"Unknown tool: {tool_name}"}

    if tool_name not in available_tools(session):
        return {
            "success": False,
            "message": "Choose the target column first. That locks away the test rows; "
                       "until then you can only preview a few rows.",
        }

    # An argument sent as null (LLMs often do this) means "not provided": let the tool use its default
    args = {k: v for k, v in (args or {}).items() if v is not None}

    try:
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
