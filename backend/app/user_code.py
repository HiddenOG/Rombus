"""Run user-written code for custom views and custom models, and collect what it produced.

Like custom recipe steps, this runs with full access to the machine (as a Jupyter cell would),
so it only runs when ALLOW_CUSTOM_CODE is on.
"""

import base64
import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.config import settings
from app.recipe import CODE_FILENAME, _user_error

MAX_OUTPUT_CHARS = 5000


@dataclass
class CodeRun:
    namespace: Dict[str, Any]
    output: str = ""
    images: List[str] = field(default_factory=list)  # PNG data URLs of matplotlib figures


def run_user_code(code: str, variables: Dict[str, Any]) -> CodeRun:
    """Execute code with the given variables; capture printed output and matplotlib figures.

    Raises ValueError with a readable message (and line number) if the code fails.
    """
    if not settings.ALLOW_CUSTOM_CODE:
        raise ValueError("Custom code is switched off on this server (ALLOW_CUSTOM_CODE=false).")

    plt = _pyplot()
    if plt:
        plt.close("all")
    namespace: Dict[str, Any] = {"pd": pd, "np": np, **variables}
    printed = io.StringIO()
    try:
        compiled = compile(code, CODE_FILENAME, "exec")
        with contextlib.redirect_stdout(printed):
            exec(compiled, namespace)
    except SyntaxError as e:
        raise ValueError(f"Your code has a syntax error on line {e.lineno}: {e.msg}") from None
    except Exception as e:
        raise _user_error(e) from None

    run = CodeRun(namespace=namespace, output=printed.getvalue()[:MAX_OUTPUT_CHARS])
    if plt:
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            run.images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
        plt.close("all")
    return run


def _pyplot():
    """matplotlib.pyplot with a file-only backend, or None if matplotlib isn't installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def result_evidence(value: Any) -> Dict[str, Any]:
    """Turn a view's `result` into something the notebook can show."""
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame):
        table = value.head(50)
        if not isinstance(table.index, pd.RangeIndex):
            table = table.reset_index()
        return {"preview": table.astype(object).where(table.notna(), None).to_dict(orient="records")}
    if value is None:
        return {}
    return {"output": str(value)[:MAX_OUTPUT_CHARS]}
