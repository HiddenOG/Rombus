"""Reading what people type, safely: row conditions and column formulas.

Neither is ever run as Python. pandas' df.query() and df.eval() evaluate text as an expression,
which would let a visitor run code on the server. Instead:
  - conditions ("price > 100 and location == 'Hebbal'") go through a small parser
  - formulas ("QUANTITY * PRICE", "log(total_sqft) / bath") are parsed with Python's ast module
    and only arithmetic, comparisons, numbers, column names and a few maths functions are allowed

Each can also be written out as plain pandas code (for the code tab and the exported script).
"""

import ast
import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Conditions ───────────────────────────────────────────────────────────────────

OPERATORS = [">=", "<=", "!=", "==", "=", ">", "<", "not contains", "contains",
             "is not blank", "is not null", "is not missing", "is blank", "is null", "is missing"]


def _parse_value(text: str, series: pd.Series) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    if pd.api.types.is_bool_dtype(series):
        return text.lower() in ("true", "yes", "1")
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(text.replace(",", ""))
        except ValueError:
            raise ValueError(f"'{text}' isn't a number, but '{series.name}' holds numbers.")
    return text


def _split_condition(df: pd.DataFrame, text: str) -> Tuple[str, str, str]:
    """`price >= 100` -> ('price', '>=', '100'). Column names may contain spaces, or be `backticked`."""
    text = text.strip()
    quoted = re.match(r"`([^`]+)`\s*(.*)$", text)
    if quoted:
        col, rest = quoted.group(1), quoted.group(2)
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    else:
        matches = [c for c in df.columns if text.lower().startswith(str(c).lower())]
        if not matches:
            raise ValueError(f"Couldn't find a column at the start of '{text}'. Columns: {', '.join(map(str, df.columns))}.")
        col = max(matches, key=lambda c: len(str(c)))
        rest = text[len(str(col)):]
    rest = rest.strip()
    op = next((o for o in OPERATORS if rest.lower().startswith(o)), None)
    if op is None:
        raise ValueError(f"In '{text}': use one of > >= < <= == != contains, or 'is blank' / 'is not blank'.")
    if op in (">", ">=", "<", "<=") and not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"'{col}' holds text, so it can't be compared with {op}. Use == or contains.")
    return col, op, rest[len(op):].strip()


def _condition(df: pd.DataFrame, text: str) -> Tuple[pd.Series, str]:
    col, op, value_text = _split_condition(df, text)
    series = df[col]
    if op.startswith("is not"):
        return series.notna(), f"{col} is not blank"
    if op.startswith("is "):
        return series.isna(), f"{col} is blank"
    if "contains" in op:
        needle = _parse_value(value_text, pd.Series(dtype=str))
        hit = series.astype(str).str.contains(needle, case=False, regex=False) & series.notna()
        return (~hit if op.startswith("not") else hit), f"{col} {op} {value_text}"
    value = _parse_value(value_text, series)
    compare = {">=": series.ge, "<=": series.le, ">": series.gt, "<": series.lt,
               "==": series.eq, "=": series.eq, "!=": series.ne}[op]
    return compare(value).fillna(False).astype(bool), f"{col} {op} {value_text}"


def _alternatives(text: str) -> List[List[str]]:
    if not text or not text.strip():
        raise ValueError("Write at least one condition, like: price > 100")
    return [re.split(r"\s+and\s+", alt, flags=re.IGNORECASE)
            for alt in re.split(r"\s+or\s+", text.strip(), flags=re.IGNORECASE)]


def parse_filter(df: pd.DataFrame, text: str) -> Tuple[pd.Series, str]:
    """Conditions joined with `and` / `or` (and binds tighter, as in Python) -> (row mask, description)."""
    keep = pd.Series(False, index=df.index)
    described = []
    for pieces in _alternatives(text):
        part = pd.Series(True, index=df.index)
        names = []
        for piece in pieces:
            mask, name = _condition(df, piece)
            part &= mask
            names.append(name)
        keep |= part
        described.append(" and ".join(names))
    return keep, " or ".join(described)


def filter_code(df: pd.DataFrame, text: str, frame: str = "df") -> str:
    """The same conditions as a plain pandas mask, e.g. (df['price'] > 100.0) & (df['location'] == 'Hebbal')."""
    def one(piece: str) -> str:
        col, op, value_text = _split_condition(df, piece)
        ref = f"{frame}[{col!r}]"
        if op.startswith("is not"):
            return f"{ref}.notna()"
        if op.startswith("is "):
            return f"{ref}.isna()"
        if "contains" in op:
            code = f"{ref}.astype(str).str.contains({_parse_value(value_text, pd.Series(dtype=str))!r}, case=False, regex=False)"
            return f"~{code}" if op.startswith("not") else f"({code})"
        return f"({ref} {'==' if op == '=' else op} {_parse_value(value_text, df[col])!r})"

    alternatives = [" & ".join(one(p) for p in pieces) for pieces in _alternatives(text)]
    return " | ".join(f"({a})" for a in alternatives) if len(alternatives) > 1 else alternatives[0]


# ── Formulas ─────────────────────────────────────────────────────────────────────

FUNCTIONS: Dict[str, Callable] = {
    "log": np.log, "log1p": np.log1p, "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp,
    "round": np.round, "floor": np.floor, "ceil": np.ceil,
}
_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
_COMPARE = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}
MAX_FORMULA = 300


def _prepare(formula: str, columns) -> Tuple[ast.AST, Dict[str, str]]:
    """Parse a formula. `backticked` names (for columns with spaces) become placeholders."""
    if not formula or not formula.strip():
        raise ValueError("Write a formula, like: QUANTITY * PRICE")
    if len(formula) > MAX_FORMULA:
        raise ValueError(f"Formulas are limited to {MAX_FORMULA} characters.")
    names: Dict[str, str] = {}

    def placeholder(m):
        key = f"_col{len(names)}"
        names[key] = m.group(1)
        return key

    text = re.sub(r"`([^`]+)`", placeholder, formula.strip())
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ValueError(f"Couldn't read the formula '{formula}'. Use + - * / **, numbers, column names and "
                         f"{', '.join(FUNCTIONS)}( ).")
    for node in ast.walk(tree):
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load,
                   ast.Call, ast.Compare, *_BINARY, *_COMPARE)
        if not isinstance(node, allowed):
            raise ValueError(f"'{formula}' uses something formulas can't: only arithmetic, comparisons, numbers, "
                             f"column names and {', '.join(FUNCTIONS)}( ) are allowed.")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
            raise ValueError("Formulas can only contain numbers, not text.")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS
                                           or len(node.args) != 1 or node.keywords):
            raise ValueError(f"Functions allowed in formulas: {', '.join(f'{f}(x)' for f in FUNCTIONS)}.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not _is_function_name(tree, node):
            col = names.get(node.id, node.id)
            if col not in columns:
                raise ValueError(f"Column '{col}' not found. Columns with spaces go in backticks, like `total sqft`.")
    return tree, names


def _is_function_name(tree: ast.AST, name: ast.Name) -> bool:
    return any(isinstance(n, ast.Call) and n.func is name for n in ast.walk(tree))


def evaluate_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    """The formula's value for every row. Division by zero gives a blank, not infinity."""
    tree, names = _prepare(formula, df.columns)

    def value(node):
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)  # floats: 10 ** 10 ** 10 overflows to inf at once instead of hanging
        if isinstance(node, ast.Name):
            series = df[names.get(node.id, node.id)]
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"'{series.name}' holds text, so it can't be used in a formula. Encode it first.")
            return series.astype(float)
        if isinstance(node, ast.UnaryOp):
            return -value(node.operand) if isinstance(node.op, ast.USub) else value(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = value(node.left), value(node.right)
            with np.errstate(all="ignore"):
                return {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
                        ast.FloorDiv: np.floor_divide, ast.Mod: np.mod, ast.Pow: np.power}[type(node.op)](left, right)
        if isinstance(node, ast.Compare):
            result, left = True, value(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = value(comparator)
                result = result & {ast.Gt: np.greater, ast.GtE: np.greater_equal, ast.Lt: np.less,
                                   ast.LtE: np.less_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}[type(op)](left, right)
                left = right
            return result.astype(float)  # True/False -> 1/0, a feature models can use
        if isinstance(node, ast.Call):
            with np.errstate(all="ignore"):
                return FUNCTIONS[node.func.id](value(node.args[0]))
        raise ValueError("Unsupported formula.")  # unreachable: _prepare checked every node

    result = value(tree)
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=df.index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


def formula_code(formula: str, columns, frame: str = "df") -> str:
    """The formula as pandas/numpy code, e.g. df['QUANTITY'] * df['PRICE']."""
    tree, names = _prepare(formula, columns)

    def code(node) -> str:
        if isinstance(node, ast.Expression):
            return code(node.body)
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Name):
            return f"{frame}[{names.get(node.id, node.id)!r}]"
        if isinstance(node, ast.UnaryOp):
            return f"-{code(node.operand)}" if isinstance(node.op, ast.USub) else code(node.operand)
        if isinstance(node, ast.BinOp):
            return f"({code(node.left)} {_BINARY[type(node.op)]} {code(node.right)})"
        if isinstance(node, ast.Compare):
            parts = [code(node.left)] + [f"{_COMPARE[type(o)]} {code(c)}" for o, c in zip(node.ops, node.comparators)]
            return f"({' '.join(parts)}).astype(float)"
        if isinstance(node, ast.Call):
            return f"np.{FUNCTIONS[node.func.id].__name__}({code(node.args[0])})"
        raise ValueError("Unsupported formula.")

    text = code(tree)
    return text[1:-1] if text.startswith("(") and text.endswith(")") and text.count("(") == 1 else text
