"""Acquire tools — data loading and acquisition."""

import pandas as pd
from typing import Dict, Any
from app.tools import register_tool, validate_column, COLUMN
from app.session import DataSession


DEFAULT_TEST_SIZE = 0.2
MIN_TEST_SIZE, MAX_TEST_SIZE = 0.05, 0.5
PREVIEW_ROWS_BEFORE_SPLIT = 10


def parse_test_size(value: Any) -> float:
    """Accept a fraction (0.3) or a percentage (30) and return a fraction within limits."""
    size = float(value)
    if size >= 1:
        size /= 100
    if not MIN_TEST_SIZE <= size <= MAX_TEST_SIZE:
        raise ValueError(
            f"Test size must be between {MIN_TEST_SIZE:.0%} and {MAX_TEST_SIZE:.0%} of the rows "
            f"(got {size:.0%})."
        )
    return size


@register_tool(
    "acquire_iris_data",
    "Replace the current dataset with the classic Iris dataset from the UCI repository.",
    before_split=True,
)
def acquire_iris_data(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Load the Iris dataset from UCI repository."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    col_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']
    df = pd.read_csv(url, names=col_names)
    session.df = df
    session.original_df = df.copy()
    session.filename = "iris.data"
    # New dataset: any previous target and lock no longer apply
    session.target_column = session.problem_type = None
    session.test_df = session.test_size = None
    return {
        "success": True,
        "message": f"Iris dataset acquired from UCI Repository. Shape: {df.shape[0]} rows × {df.shape[1]} columns.",
    }


@register_tool(
    "select_target",
    "Set the target/label column to predict and lock away a share of rows as the final test set. "
    "Ask the user what percentage to lock away if they haven't said (default 20%).",
    {
        "column": COLUMN,
        "test_size": {
            "type": "number",
            "description": f"Fraction of rows to lock away as the test set, "
                           f"{MIN_TEST_SIZE} to {MAX_TEST_SIZE} (default {DEFAULT_TEST_SIZE}).",
        },
    },
    ["column"],
    before_split=True,
)
def select_target(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Set the target column and lock away the test rows."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}

    validate_column(session, col)

    # Changing the split after cleaning would mean the cleaning was learned from different rows
    if session.is_split and session.can_undo:
        return {
            "success": False,
            "message": "The training rows have already been changed, so the split can't be redone. "
                       "Undo the cleaning steps first.",
        }

    test_size = parse_test_size(args.get("test_size", DEFAULT_TEST_SIZE))

    session.target_column = col
    nunique = session.original_df[col].nunique()
    session.problem_type = "classification" if nunique < 20 else "regression"
    dropped = session.lock_test_rows(test_size)

    msg = (
        f"Target set to '{col}' ({session.problem_type}, {nunique} unique values). "
        f"Locked away {len(session.test_df)} rows ({test_size:.0%}) as the final test set; "
        f"working with {len(session.df)} training rows."
    )
    if dropped:
        msg += f" Removed {dropped} rows with a blank '{col}'."
    return {"success": True, "message": msg}


@register_tool(
    "preview_data",
    f"Preview the first N rows of the dataset (at most {PREVIEW_ROWS_BEFORE_SPLIT} before the target is chosen).",
    {"rows": {"type": "integer", "description": "Number of rows (default 5)."}},
    before_split=True,
)
def preview_data(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Preview first N rows of the dataset."""
    rows = int(args.get("rows", 5))
    if not session.is_split:
        rows = min(rows, PREVIEW_ROWS_BEFORE_SPLIT)
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
