"""Tool registry — maps tool names to handler functions."""

from typing import Dict, Callable, Any
from app.session import DataSession


# Registry of all available tools
TOOL_REGISTRY: Dict[str, Callable] = {}


def register_tool(name: str):
    """Decorator to register a tool function."""
    def decorator(func):
        TOOL_REGISTRY[name] = func
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


# Import all tool modules to trigger registration
from app.tools import acquire, scrub, explore, model, workflow
