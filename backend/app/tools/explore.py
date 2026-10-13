"""Explore tools — EDA, charts, and statistical analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from app.tools import register_tool, validate_column, COLUMN
from app.session import DataSession


@register_tool(
    "plot_histogram",
    "Histogram of a numeric column, or value counts of a categorical one.",
    {"column": COLUMN},
    ["column"],
)
def plot_histogram(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate histogram data with bins for chart rendering."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}
    validate_column(session, col)

    series = session.df[col].dropna()

    if pd.api.types.is_numeric_dtype(series):
        # Create binned histogram data
        counts, bin_edges = np.histogram(series, bins=min(30, max(10, len(series) // 10)))
        bins_data = []
        for i in range(len(counts)):
            bins_data.append({
                "bin": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                "binStart": round(float(bin_edges[i]), 4),
                "binEnd": round(float(bin_edges[i+1]), 4),
                "count": int(counts[i]),
            })
        chart_type = "histogram"
    else:
        # Value counts for categorical data
        value_counts = series.value_counts().head(20)
        bins_data = [
            {"bin": str(k), "count": int(v)}
            for k, v in value_counts.items()
        ]
        chart_type = "bar"

    return {
        "success": True,
        "message": f"Generated histogram for '{col}'.",
        "evidence": {
            "chart_type": chart_type,
            "chart_data": bins_data,
            "column": col,
            "title": f"Distribution of {col}",
        },
    }


@register_tool("plot_boxplot", "Box plot (quartiles, whiskers, outliers) of a numeric column.", {"column": COLUMN}, ["column"])
def plot_boxplot(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate box plot statistics."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}
    validate_column(session, col)

    if not pd.api.types.is_numeric_dtype(session.df[col]):
        return {"success": False, "message": f"Column '{col}' is not numeric. Cannot create boxplot."}

    series = session.df[col].dropna()
    q1, median, q3 = float(series.quantile(0.25)), float(series.median()), float(series.quantile(0.75))
    iqr = q3 - q1
    whisker_low = float(series[series >= q1 - 1.5 * iqr].min())
    whisker_high = float(series[series <= q3 + 1.5 * iqr].max())
    outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)].tolist()

    return {
        "success": True,
        "message": f"Generated boxplot for '{col}'.",
        "evidence": {
            "chart_type": "boxplot",
            "chart_data": {
                "column": col,
                "min": round(whisker_low, 4),
                "q1": round(q1, 4),
                "median": round(median, 4),
                "q3": round(q3, 4),
                "max": round(whisker_high, 4),
                "outliers": [round(o, 4) for o in outliers[:50]],
                "outlier_count": len(outliers),
            },
            "title": f"Box Plot of {col}",
        },
    }


@register_tool(
    "plot_correlation",
    "Correlation heatmap (or covariance table) of the numeric columns, or BETWEEN GROUPS: e.g. how the daily "
    "returns of different companies move together, matched on date (value + group_by + align_on).",
    {"measure": {"type": "string", "enum": ["correlation", "covariance"]},
     "value": {"type": "string", "description": "Between groups: the numbers to compare (column), e.g. returns."},
     "group_by": {"type": "string", "description": "Between groups: one row/column per value of this column, e.g. company."},
     "align_on": {"type": "string", "description": "Between groups: the column that matches rows up, e.g. Date."}},
)
def plot_correlation(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Correlation (or covariance) of numeric columns, or of one value between groups."""
    measure = args.get("measure") or "correlation"
    if measure not in ("correlation", "covariance"):
        raise ValueError("Measure must be 'correlation' or 'covariance'.")
    value, group, align = args.get("value"), args.get("group_by"), args.get("align_on")
    if any((value, group, align)):
        if not all((value, group, align)):
            raise ValueError("Between groups needs all three: value, group_by and align_on.")
        for c in (value, group, align):
            validate_column(session, c)
        if not pd.api.types.is_numeric_dtype(session.df[value]):
            raise ValueError(f"'{value}' holds text; correlation needs numbers.")
        # One column per group, one row per matching key (e.g. a column per company, a row per date)
        numeric_df = session.df.pivot_table(index=align, columns=group, values=value, aggfunc="mean")
        numeric_df.columns = [str(c) for c in numeric_df.columns]
        about = f"of '{value}' between '{group}' groups, matched on '{align}'"
    else:
        numeric_df = session.df.select_dtypes(include=[np.number])
        about = "of the numeric columns"

    if numeric_df.shape[1] < 2:
        return {"success": False, "message": f"Need at least 2 numeric columns (or groups) for {measure}."}

    if measure == "covariance":
        cov = numeric_df.cov().round(6)
        table = cov.reset_index().rename(columns={"index": ""})
        return {
            "success": True,
            "message": f"Covariance {about}: how much each pair moves together, in the data's own units "
                       "(unlike correlation, it isn't on a -1 to 1 scale).",
            "evidence": {"preview": table.astype(object).where(table.notna(), None).to_dict(orient="records")},
        }

    corr = numeric_df.corr()

    # Convert to list-of-dicts format for heatmap rendering
    heatmap_data = []
    columns = corr.columns.tolist()
    for i, row_name in enumerate(columns):
        for j, col_name in enumerate(columns):
            heatmap_data.append({
                "x": col_name,
                "y": row_name,
                "value": round(float(corr.iloc[i, j]), 4),
            })

    # Also provide the matrix dict for backward compat
    corr_dict = {}
    for col in columns:
        corr_dict[col] = {c: round(float(corr.loc[col, c]), 4) for c in columns}

    return {
        "success": True,
        "message": f"Correlation {about}: {len(columns)} of them.",
        "evidence": {
            "chart_type": "heatmap",
            "chart_data": heatmap_data,
            "columns": columns,
            "correlation": corr_dict,
            "title": f"Correlation of {value} by {group}" if group else "Feature Correlation Heatmap",
        },
    }


MAX_LINES = 8          # the validated categorical palette has 8 colors; more lines fold into "others not shown"
MAX_POINTS_PER_LINE = 500


@register_tool(
    "plot_line",
    "Line chart: y over x (dates understood), one line per group (like hue) or one per y column. "
    "For trends over time: prices, sales, returns.",
    {"x": COLUMN,
     "y": {"type": "array", "items": {"type": "string"}, "description": "Column(s) to draw as lines."},
     "group_by": {"type": "string", "description": "Optional: one line per value of this column, e.g. company."}},
    ["x", "y"],
)
def plot_line(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    from app.expressions import as_dates, looks_like_dates
    df = session.df
    x, group = args.get("x"), args.get("group_by")
    ys = args.get("y") or []
    ys = [ys] if isinstance(ys, str) else list(ys)
    for c in [x, *ys] + ([group] if group else []):
        validate_column(session, c)
    if not ys:
        raise ValueError("Choose at least one column to draw (y).")
    not_numbers = [c for c in ys if not pd.api.types.is_numeric_dtype(df[c])]
    if not_numbers:
        raise ValueError(f"{not_numbers} hold text; a line needs numbers. Clean them into numbers first.")
    if group and len(ys) > 1:
        raise ValueError("With one line per group, draw one column at a time.")

    is_date = pd.api.types.is_datetime64_any_dtype(df[x]) or (not pd.api.types.is_numeric_dtype(df[x]) and looks_like_dates(df[x]))
    if is_date:
        dates = as_dates(df[x])  # sent as milliseconds since 1970, which the chart turns back into dates
        xs = pd.Series(dates.values.astype("datetime64[ms]").astype("int64"), index=df.index).where(dates.notna())
    elif pd.api.types.is_numeric_dtype(df[x]):
        xs = df[x].astype(float)
    else:
        raise ValueError(f"'{x}' is neither numbers nor dates, so it can't be an x-axis. Try a bar chart (histogram).")

    if group:
        keys = df[group].astype(str).where(df[group].notna(), "(blank)")
        names = keys.value_counts().index.tolist()
        lines = [(name, xs[keys == name], df.loc[keys == name, ys[0]]) for name in names[:MAX_LINES]]
        hidden = names[MAX_LINES:]
    else:
        lines = [(y, xs, df[y]) for y in ys[:MAX_LINES]]
        hidden = ys[MAX_LINES:]

    series = []
    for name, xv, yv in lines:
        points = pd.DataFrame({"x": xv, "y": yv.astype(float)}).dropna().sort_values("x", kind="stable")
        if len(points) > MAX_POINTS_PER_LINE:  # evenly spaced, keeping the first and last
            points = points.iloc[np.unique(np.linspace(0, len(points) - 1, MAX_POINTS_PER_LINE).astype(int))]
        series.append({"name": str(name), "points": points.round(6).values.tolist()})

    title = f"{', '.join(ys)} over {x}" + (f" by {group}" if group else "")
    msg = f"Line chart of {title}: {len(series)} line{'s' if len(series) != 1 else ''}."
    if hidden:
        msg += f" Showing the {MAX_LINES} largest; not shown: {', '.join(map(str, hidden[:10]))}{'…' if len(hidden) > 10 else ''}."
    return {
        "success": True,
        "message": msg,
        "evidence": {"chart_type": "line", "title": title, "x": x, "x_is_date": bool(is_date),
                     "y_label": ys[0] if len(ys) == 1 else "value", "series": series},
    }


@register_tool(
    "show_unique_values",
    "List a column's distinct values with how many rows have each (like value_counts / unique()).",
    {"column": COLUMN},
    ["column"],
)
def show_unique_values(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    col = args.get("column")
    validate_column(session, col)
    series = session.df[col]
    counts = series.value_counts(dropna=False)
    shown = counts.head(100)
    table = [{"value": None if pd.isna(v) else (v.item() if hasattr(v, "item") else v),
              "rows": int(n), "share": f"{n / len(series):.1%}"} for v, n in shown.items()]
    values = [repr(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for v in counts.index[:30]]
    listed = "[" + ", ".join(values) + (", …" if len(counts) > 30 else "") + "]"
    more = f" Showing the 100 most common of {len(counts)}." if len(counts) > 100 else ""
    return {
        "success": True,
        "message": f"'{col}' has {series.nunique()} distinct values ({int(series.isna().sum())} blank): {listed}.{more}",
        "evidence": {"preview": table},
    }


@register_tool("plot_missing", "Map of where the blanks (missing values) are: share per column, and a row-by-column picture.")
def plot_missing(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    df = session.df
    missing = df.isna()
    share = missing.mean()
    # Columns with blanks first (most blanks first), then the rest; at most 40 so the picture stays readable
    order = share.sort_values(ascending=False, kind="stable").index.tolist()[:40]
    # Up to 120 rows, evenly spaced, in their original order (like seaborn's heatmap of df.isnull())
    rows = np.unique(np.linspace(0, len(df) - 1, min(len(df), 120)).astype(int)) if len(df) else []
    blank_cols = [c for c in order if share[c] > 0]
    any_blank = int(missing.any(axis=1).sum())
    return {
        "success": True,
        "message": (f"{len(blank_cols)} of {df.shape[1]} columns have blanks; {any_blank} of {len(df)} training rows "
                    f"({any_blank / max(len(df), 1):.1%}) have at least one."
                    + (" Worst: " + ", ".join(f"{c} {share[c]:.1%}" for c in blank_cols[:5]) + "." if blank_cols else "")),
        "evidence": {
            "chart_type": "missing_map",
            "title": "Where the blanks are",
            "columns": [str(c) for c in order],
            "missing_share": {str(c): round(float(share[c]), 4) for c in order},
            "rows": [[int(v) for v in missing.iloc[i][order].to_numpy()] for i in rows],
            "row_numbers": [int(i) for i in rows],
            "total_rows": len(df),
        },
    }


@register_tool(
    "plot_scatterplot",
    "Scatter plot of two columns, optionally colored by a third.",
    {"x": COLUMN, "y": COLUMN, "hue": COLUMN},
    ["x", "y"],
)
def plot_scatterplot(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate scatter plot data."""
    x_col = args.get("x")
    y_col = args.get("y")
    hue_col = args.get("hue")

    if not x_col or not y_col:
        return {"success": False, "message": "Missing 'x' and/or 'y' arguments."}

    validate_column(session, x_col)
    validate_column(session, y_col)
    if hue_col:
        validate_column(session, hue_col)

    cols = [x_col, y_col]
    if hue_col:
        cols.append(hue_col)

    # Limit to 500 points for performance
    sample_df = session.df[cols].dropna()
    if len(sample_df) > 500:
        sample_df = sample_df.sample(500, random_state=42)

    scatter_data = sample_df.to_dict(orient='records')

    return {
        "success": True,
        "message": f"Generated scatter plot: {x_col} vs {y_col}" + (f" (colored by {hue_col})" if hue_col else ""),
        "evidence": {
            "chart_type": "scatter",
            "chart_data": scatter_data,
            "x": x_col,
            "y": y_col,
            "hue": hue_col,
            "title": f"{x_col} vs {y_col}",
        },
    }
