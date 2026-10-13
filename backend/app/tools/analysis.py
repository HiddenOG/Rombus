"""Analysis views: the everyday pandas questions (filter, group & summarise, pivot).

All look-only: they never change the data or the recipe, so they can't leak test rows into
anything. Like every view, they see the training rows only. Filter conditions are read by
app/expressions.py, never by df.query() (which would run the text as code).
"""

from typing import Any, Dict, List

import pandas as pd

from app.expressions import parse_filter
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


# ── Filter rows ──────────────────────────────────────────────────────────────────

CONDITIONS = {"type": "string",
              "description": 'e.g. price > 100 and location == "Whitefield". Operators: > >= < <= == != '
                             "contains, is blank, is not blank; join with and / or."}


@register_tool(
    "filter_rows",
    "Show only the training rows that match conditions, like df.query (look-only: nothing is removed).",
    {"conditions": CONDITIONS},
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
     "value": {"type": "string",
               "description": "Column(s) to summarise, comma-separated for several (leave out to count rows per group)."},
     "aggregation": {"type": "string", "enum": AGGREGATIONS}},
    ["group_by"],
)
def group_summary(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    df = session.df
    by, value, agg = args.get("group_by"), args.get("value"), args.get("aggregation") or ("mean" if args.get("value") else "count")
    validate_column(session, by)
    if agg not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation '{agg}'. Use one of: {', '.join(AGGREGATIONS)}.")
    values = _value_columns(df, value)
    if len(values) > 1:
        return _group_several(session, by, values, agg)
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


def _value_columns(df: pd.DataFrame, value: Any) -> List[str]:
    """One column, or several: a list, or comma-separated names (a single name with a comma still works)."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if value in df.columns:
        return [value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _group_several(session: DataSession, by: str, values: List[str], agg: str) -> Dict[str, Any]:
    """Several columns summarised per group in one table, like df.groupby(by)[[a, b, c]].mean()."""
    df = session.df
    for v in values:
        validate_column(session, v)
    text = [v for v in values if not pd.api.types.is_numeric_dtype(df[v])]
    if text and agg != "count":
        raise ValueError(f"{text} hold text, so their {agg} can't be taken. Use count, or numeric columns.")
    table = df.groupby(df[by].astype(str).where(df[by].notna(), "(blank)"), sort=False)[values].agg(agg).round(4)
    table = table.rename(columns={v: f"{agg} of {v}" for v in values})
    return {
        "success": True,
        "message": f"{agg.capitalize()} of {len(values)} columns for each of {len(table)} '{by}' groups.",
        "evidence": {"preview": _table(table.reset_index())},
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
        "evidence": {"preview": _table(table.round(4))},
    }
