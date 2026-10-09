"""Static checks on custom step code: catch peeking before it runs.

apply() runs on test rows and new customers. If it calculates a statistic (mean, median,
max …) from the rows it's given, then on the test rows it is learning from the exam:
leakage. Those numbers belong in learn(), which only ever sees training rows.

This is a reading check (no code is executed), so it can miss cleverly hidden cases;
it's a safety net, not a proof.
"""

import ast
from typing import List

# Methods that compute something from the data they're called on
LEARNING_METHODS = {
    "mean", "median", "mode", "std", "var", "min", "max", "sum", "quantile",
    "unique", "nunique", "value_counts", "describe", "corr", "cov", "rank", "fit", "fit_transform",
}


def peeking_warnings(code: str) -> List[str]:
    """Warnings for statistics computed inside apply() from its own input rows."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # the runner reports syntax errors itself

    warnings: List[str] = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "apply" and fn.args.args):
            continue
        rows_name = fn.args.args[0].arg  # usually "df"
        seen = set()
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LEARNING_METHODS
                and _starts_from(node.func.value, rows_name)
                and node.func.attr not in seen
            ):
                seen.add(node.func.attr)
                warnings.append(
                    f"Line {node.lineno}: apply() calculates {node.func.attr}() from the rows it's given. "
                    f"On the test rows that's peeking. Calculate it in learn() and read it from `learned`."
                )
    return warnings


def _starts_from(node: ast.AST, name: str) -> bool:
    """Does an expression like df["age"].fillna(0) start from the variable `name`?"""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return isinstance(node, ast.Name) and node.id == name
