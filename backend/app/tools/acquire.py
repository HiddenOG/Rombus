"""Acquire tools — data loading and acquisition."""

import pandas as pd
from typing import Dict, Any
from app.tools import register_tool, validate_column, COLUMN
from app.session import DataSession


@register_tool("acquire_iris_data", "Replace the current dataset with the classic Iris dataset from the UCI repository.")
def acquire_iris_data(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Load the Iris dataset from UCI repository."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    col_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']
    df = pd.read_csv(url, names=col_names)
    session.df = df
    session.original_df = df.copy()
    session.filename = "iris.data"
    return {
        "success": True,
        "message": f"Iris dataset acquired from UCI Repository. Shape: {df.shape[0]} rows × {df.shape[1]} columns.",
    }


@register_tool(
    "select_target",
    "Set the target/label column to predict. Infers classification vs regression.",
    {"column": COLUMN},
    ["column"],
)
def select_target(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Set the target column for modeling."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}

    validate_column(session, col)

    session.target_column = col
    nunique = session.df[col].nunique()
    session.problem_type = "classification" if nunique < 20 else "regression"

    return {
        "success": True,
        "message": f"Target set to '{col}'. Unique values: {nunique}. Inferred problem type: {session.problem_type}.",
    }


@register_tool(
    "preview_data",
    "Preview the first N rows of the dataset.",
    {"rows": {"type": "integer", "description": "Number of rows (default 5)."}},
)
def preview_data(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Preview first N rows of the dataset."""
    rows = args.get("rows", 5)
    preview = session.df.head(rows).to_dict(orient='records')
    return {
        "success": True,
        "message": f"Showing first {rows} rows.",
        "evidence": {"preview": preview},
    }


@register_tool("get_data_info", "Show dtypes, non-null counts and descriptive statistics (df.info + df.describe).")
def get_data_info(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Get comprehensive data info and descriptive statistics."""
    import io
    buffer = io.StringIO()
    session.df.info(buf=buffer)
    info_str = buffer.getvalue()

    describe_dict = {}
    describe_df = session.df.describe(include='all')
    for col in describe_df.columns:
        describe_dict[col] = {}
        for stat in describe_df.index:
            val = describe_df.loc[stat, col]
            if pd.isna(val):
                describe_dict[col][stat] = None
            elif isinstance(val, (int, float)):
                describe_dict[col][stat] = round(float(val), 4)
            else:
                describe_dict[col][stat] = str(val)

    return {
        "success": True,
        "message": "Generated data info and descriptive statistics.",
        "evidence": {"info": info_str, "describe": describe_dict},
    }
