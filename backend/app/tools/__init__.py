"""Tool registry — maps tool names to handler functions and their schemas."""

from typing import Dict, Callable, Any, List, Optional
import pandas as pd
from app.session import DataSession


# Registry of all available tools
TOOL_REGISTRY: Dict[str, Callable] = {}

# JSON-schema specs for each tool, used to build Gemini function declarations
TOOL_SPECS: Dict[str, Dict[str, Any]] = {}


def register_tool(
    name: str,
    description: str,
    properties: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
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
        return func
    return decorator


def execute_tool(tool_name: str, session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered tool by name."""
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "message": f"Unknown tool: {tool_name}"}

    try:
        return TOOL_REGISTRY[tool_name](session, args)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"success": False, "message": str(e)}


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
COLUMN = {"type": "string", "description": "Exact column name from the dataset."}
COLUMNS = {"type": "array", "items": {"type": "string"}, "description": "Exact column names from the dataset."}


# Import all tool modules to trigger registration
from app.tools import acquire, scrub, explore, model, workflow  # noqa: E402,F401
