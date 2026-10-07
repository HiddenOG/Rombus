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


@register_tool("plot_correlation", "Correlation heatmap of all numeric columns.")
def plot_correlation(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate correlation matrix for numeric features."""
    numeric_df = session.df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return {"success": False, "message": "Need at least 2 numeric columns for correlation."}

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
        "message": f"Generated correlation matrix for {len(columns)} numeric features.",
        "evidence": {
            "chart_type": "heatmap",
            "chart_data": heatmap_data,
            "columns": columns,
            "correlation": corr_dict,
            "title": "Feature Correlation Heatmap",
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
