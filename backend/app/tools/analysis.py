"""Analysis views: the everyday pandas questions (filter, group & summarise, pivot).

All look-only: they never change the data or the recipe, so they can't leak test rows into
anything. Like every view, they see the training rows only.

Filter conditions are read by a small parser, not pandas' df.query(): query() evaluates its text
as an expression, which would let a visitor run code on the server.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.session import DataSession
from app.tools import COLUMN, register_tool, validate_column

AGGREGATIONS = ["mean", "median", "sum", "count", "min", "max"]
MAX_ROWS_SHOWN = 50


def _table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rows for the notebook's table view (index kept as a column when it means something)."""
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    df = df.head(MAX_ROWS_SHOWN)
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def _rounded(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(4) if hasattr(df, "round") else df


# ── Filter rows ──────────────────────────────────────────────────────────────────

_OPERATORS = [">=", "<=", "!=", "==", "=", ">", "<", "not contains", "contains",
              "is not blank", "is not null", "is not missing", "is blank", "is null", "is missing"]


def _parse_value(text: str, series: pd.Series) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(text.replace(",", ""))
        except ValueError:
            raise ValueError(f"'{text}' isn't a number, but '{series.name}' holds numbers.")
    if pd.api.types.is_bool_dtype(series):
        return text.lower() in ("true", "yes", "1")
    return text


def _condition(df: pd.DataFrame, text: str) -> Tuple[pd.Series, str]:
    """One condition, like `age > 30`, `size == "2 BHK"` or `society is blank`."""
    text = text.strip()
    # The column: longest name the condition starts with (names can contain spaces), or `backticked`
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
    op = next((o for o in _OPERATORS if rest.lower().startswith(o)), None)
    if op is None:
        raise ValueError(f"In '{text}': use one of > >= < <= == != contains, or 'is blank' / 'is not blank'.")
    value_text = rest[len(op):]
    series = df[col]

    if op.startswith("is not"):
        return series.notna(), f"{col} is not blank"
    if op.startswith("is "):
        return series.isna(), f"{col} is blank"
    if "contains" in op:
        hit = series.astype(str).str.contains(_parse_value(value_text, pd.Series(dtype=str)), case=False, regex=False) & series.notna()
        return (~hit if op.startswith("not") else hit), f"{col} {op} {value_text.strip()}"

    value = _parse_value(value_text, series)
    comparisons = {">=": series.ge, "<=": series.le, ">": series.gt, "<": series.lt,
                   "==": series.eq, "=": series.eq, "!=": series.ne}
    if op in (">", ">=", "<", "<=") and not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"'{col}' holds text, so it can't be compared with {op}. Use == or contains.")
    return comparisons[op](value).fillna(False), f"{col} {op} {value_text.strip()}"


def parse_filter(df: pd.DataFrame, text: str) -> Tuple[pd.Series, str]:
    """Conditions joined with `and` / `or` (and binds tighter, as in Python)."""
    if not text or not text.strip():
        raise ValueError("Write at least one condition, like: price > 100")
    keep = pd.Series(False, index=df.index)
    described = []
    for alternative in re.split(r"\s+or\s+", text.strip(), flags=re.IGNORECASE):
        part = pd.Series(True, index=df.index)
        names = []
        for piece in re.split(r"\s+and\s+", alternative, flags=re.IGNORECASE):
            mask, name = _condition(df, piece)
            part &= mask
            names.append(name)
        keep |= part
        described.append(" and ".join(names))
    return keep, " or ".join(described)


@register_tool(
    "filter_rows",
    "Show only the training rows that match conditions, like df.query (look-only: nothing is removed).",
    {"conditions": {"type": "string",
                    "description": 'e.g. price > 100 and location == "Whitefield". Operators: > >= < <= == != '
                                   "contains, is blank, is not blank; join with and / or."}},
    ["conditions"],
)
def filter_rows(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    df = session.df
    keep, described = parse_filter(df, args.get("conditions", ""))
    matched = df[keep]
    return {
        "success": True,
        "message": f"{len(matched)} of {len(df)} training rows ({len(matched) / max(len(df), 1):.1%}) match {described}."
                   + (f" Showing the first {MAX_ROWS_SHOWN}." if len(matched) > MAX_ROWS_SHOWN else ""),
        "evidence": {"preview": _table(matched.reset_index(drop=True))},
    }


# ── Group & summarise ────────────────────────────────────────────────────────────

@register_tool(
    "group_summary",
    "Group the training rows by a column and summarise another per group, like df.groupby(...).agg(...).",
    {"group_by": COLUMN,
     "value": {"type": "string", "description": "Column to summarise (leave out to count rows per group)."},
     "aggregation": {"type": "string", "enum": AGGREGATIONS}},
    ["group_by"],
)
def group_summary(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    df = session.df
    by, value, agg = args.get("group_by"), args.get("value"), args.get("aggregation") or ("mean" if args.get("value") else "count")
    validate_column(session, by)
    if agg not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation '{agg}'. Use one of: {', '.join(AGGREGATIONS)}.")
    if value:
        validate_column(session, value)
        if agg not in ("count",) and not pd.api.types.is_numeric_dtype(df[value]):
            raise ValueError(f"'{value}' holds text, so its {agg} can't be taken. Use count, or a numeric column.")
        result = df.groupby(by, dropna=False)[value].agg(agg)
        label = f"{agg} of {value}"
    else:
        result = df.groupby(by, dropna=False).size()
        label, agg = "rows", "count"
    result = result.rename(label).sort_values(ascending=False)
    groups = len(result)
    table = result.round(4).reset_index()
    table[by] = table[by].astype(object).where(table[by].notna(), "(blank)")
    shown = table.head(MAX_ROWS_SHOWN)
    top = ", ".join(f"{g} {v:,.4g}" for g, v in zip(shown[by].head(3), shown[label].head(3)))
    return {
        "success": True,
        "message": f"{label.capitalize()} for each of {groups} '{by}' groups. Highest: {top}."
                   + (f" Showing the top {MAX_ROWS_SHOWN}." if groups > MAX_ROWS_SHOWN else ""),
        "evidence": {
            "chart_type": "bar",
            "chart_data": [{"group": str(g), "value": float(v)} for g, v in zip(shown[by].head(20), shown[label].head(20))],
            "x_key": "group", "y_key": "value",
            "title": f"{label.capitalize()} by {by}" + (" (top 20)" if groups > 20 else ""),
            "preview": shown.astype(object).where(shown.notna(), None).to_dict(orient="records"),
        },
    }


# ── Pivot table ──────────────────────────────────────────────────────────────────

@register_tool(
    "pivot_table",
    "Cross-tabulate the training rows: one column's values as rows, another's as columns, like df.pivot_table.",
    {"rows": COLUMN,
     "columns": {"type": "string", "description": "Column whose values become the table's columns."},
     "value": {"type": "string", "description": "Column to summarise in each cell (leave out to count rows)."},
     "aggregation": {"type": "string", "enum": AGGREGATIONS}},
    ["rows", "columns"],
)
def pivot_table(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    df = session.df
    rows, cols, value = args.get("rows"), args.get("columns"), args.get("value")
    agg = args.get("aggregation") or ("mean" if value else "count")
    for c in (rows, cols) + ((value,) if value else ()):
        validate_column(session, c)
    if agg not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation '{agg}'. Use one of: {', '.join(AGGREGATIONS)}.")
    if value and agg != "count" and not pd.api.types.is_numeric_dtype(df[value]):
        raise ValueError(f"'{value}' holds text, so its {agg} can't be taken. Use count, or a numeric column.")

    # Keep it readable: the 20 most common column values, the 50 most common row values
    top_cols = df[cols].value_counts().index[:20]
    data = df[df[cols].isin(top_cols)]
    if value:
        table = data.pivot_table(index=rows, columns=cols, values=value, aggfunc=agg)
    else:
        table = pd.crosstab(data[rows], data[cols])
    table = table.loc[df[rows].value_counts().index.intersection(table.index)[:MAX_ROWS_SHOWN]]
    table.columns = [str(c) for c in table.columns]
    what = f"{agg} of {value}" if value else "row counts"
    trimmed = df[cols].nunique() > 20 or df[rows].nunique() > MAX_ROWS_SHOWN
    return {
        "success": True,
        "message": f"Pivot table of {what}: '{rows}' down the side, '{cols}' across the top."
                   + (" Showing the most common values only." if trimmed else ""),
        "evidence": {"preview": _table(_rounded(table))},
    }
