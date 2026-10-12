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
    kind="action",
)
def select_target(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Set the target column and lock away the test rows."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}

    validate_column(session, col)

    test_size = parse_test_size(args.get("test_size", DEFAULT_TEST_SIZE))

    previous = (session.target_column, session.problem_type)
    session.target_column = col
    nunique = session.original_df[col].nunique()
    session.problem_type = "classification" if nunique < 20 else "regression"
    # Re-locking replays the existing recipe on the new training rows
    try:
        dropped = session.lock_test_rows(test_size)
    except ValueError as e:
        session.target_column, session.problem_type = previous
        raise ValueError(f"Couldn't re-lock: the existing recipe doesn't fit the new split. {e}") from None

    if session.current_stage == "ACQUIRE":
        session.current_stage = "SCRUB"

    msg = (
        f"Target set to '{col}' ({session.problem_type}, {nunique} unique values). "
        f"Locked away {len(session.test_df)} rows ({test_size:.0%}) as the final test set; "
        f"working with {len(session.df)} training rows."
    )
    if dropped:
        msg += f" Removed {dropped} rows with a blank '{col}'."
    if session.recipe:
        msg += f" Replayed {len(session.recipe)} recipe steps on the new training rows."
    if session.lock_count > 1:
        msg += f" (Locked {session.lock_count} times so far.)"
    return {"success": True, "message": msg}


MAX_PREVIEW_ROWS = 200


@register_tool(
    "preview_data",
    f"Preview N rows of the dataset: the first (head), the last (tail) or a random sample "
    f"(at most {PREVIEW_ROWS_BEFORE_SPLIT} before the target is chosen).",
    {"rows": {"type": "integer", "description": "Number of rows (default 5)."},
     "position": {"type": "string", "enum": ["first", "last", "random"]}},
    before_split=True,
)
def preview_data(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Preview rows: head, tail or a random sample."""
    rows = max(1, min(int(args.get("rows", 5)), MAX_PREVIEW_ROWS))
    if not session.is_split:
        rows = min(rows, PREVIEW_ROWS_BEFORE_SPLIT)
    position = args.get("position") or "first"
    if position not in ("first", "last", "random"):
        raise ValueError("Position must be first, last or random.")
    df = session.df
    shown = {"first": df.head, "last": df.tail}.get(position, lambda n: df.sample(min(n, len(df)), random_state=0))(rows)
    return {
        "success": True,
        "message": f"Showing {'a random' if position == 'random' else 'the ' + position} {len(shown)} rows.",
        "evidence": {"preview": shown.to_dict(orient='records')},
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
