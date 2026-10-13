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


MAX_CLASSES = 20


def problem_type_of(target: pd.Series) -> str:
    """Classification for categories and a few repeated whole numbers; regression for amounts.
    A handful of distinct numbers isn't enough: 10 prices in a 10-row file are still amounts."""
    values = target.dropna()
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return "classification"
    distinct = values.nunique()
    whole = bool((values % 1 == 0).all())
    return "classification" if whole and distinct < MAX_CLASSES and distinct <= len(values) / 2 else "regression"


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
        "split": {"type": "string", "enum": ["random", "time"],
                  "description": "random (default), or time: lock the MOST RECENT rows, for data over time "
                                 "(prices, sales). A random split would let the model peek between days."},
        "time_column": {"type": "string", "description": "For split=time: the date/time column that orders the rows."},
    },
    ["column"],
    before_split=True,
    kind="action",
)
def select_target(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Set the target column and lock away the test rows (a random share, or the most recent ones)."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}

    validate_column(session, col)

    test_size = parse_test_size(args.get("test_size", DEFAULT_TEST_SIZE))
    split = args.get("split") or ("time" if args.get("time_column") else "random")
    time_column = None
    if split == "time":
        from app.expressions import looks_like_dates
        time_column = args.get("time_column")
        if not time_column:
            raise ValueError("A time split needs the date column that orders the rows (time_column).")
        validate_column(session, time_column)
        if time_column == col:
            raise ValueError("The time column orders the rows; predict a different column.")
        if not looks_like_dates(session.original_df[time_column]):
            raise ValueError(f"'{time_column}' doesn't look like dates or times, so it can't order the rows.")
    elif split != "random":
        raise ValueError("Split must be 'random' or 'time'.")

    previous = (session.target_column, session.problem_type, session.time_column)
    session.target_column, session.time_column = col, time_column
    session.problem_type = problem_type_of(session.original_df[col])
    # Re-locking replays the existing recipe on the new training rows
    try:
        dropped = session.lock_test_rows(test_size)
    except ValueError as e:
        session.target_column, session.problem_type, session.time_column = previous
        raise ValueError(f"Couldn't re-lock: the existing recipe doesn't fit the new split. {e}") from None

    if session.current_stage == "ACQUIRE":
        session.current_stage = "SCRUB"

    which = f"the most recent {len(session.test_df)} rows by '{time_column}'" if time_column else f"{len(session.test_df)} rows"
    msg = (
        f"Target set to '{col}' ({session.problem_type}, {session.original_df[col].nunique()} unique values). "
        f"Locked away {which} ({test_size:.0%}) as the final test set; "
        f"working with {len(session.df)} training rows."
    )
    if time_column:
        from app.expressions import as_dates
        cutoff = as_dates(session.test_df[time_column]).min()
        msg += (f" Time split: training rows are before {cutoff:%Y-%m-%d}, test rows from then on, so the model is "
                "always tested on a future it hasn't seen.")
    if dropped:
        msg += f" Removed {dropped} rows with a blank '{col}'" + (" or an unreadable date." if time_column else ".")
    if session.recipe:
        msg += f" Replayed {len(session.recipe)} recipe steps on the new training rows."
    if session.lock_count > 1:
        msg += f" (Locked {session.lock_count} times so far.)"
    return {"success": True, "message": msg}


@register_tool(
    "clean_only",
    "Clean the data without building a model: no target and no locked test rows, every step works on ALL "
    "rows, and the whole cleaned table can be exported. Use when the user only wants to clean or prepare data. "
    "A target can still be chosen later (the cleaning is kept and replayed on the training rows).",
    before_split=True,
    kind="action",
)
def clean_only(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    session.start_clean_only()
    if session.current_stage == "ACQUIRE":
        session.current_stage = "SCRUB"
    return {
        "success": True,
        "message": f"Clean-only mode: working on all {len(session.df):,} rows, with no target and no model. "
                   "Statistics are visible now, and Export gives you the whole cleaned table. To build a model "
                   "later, choose the column to predict: the test rows are then locked from the original data "
                   "and your cleaning steps replay on the training rows, so nothing leaks.",
    }


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
    if not session.can_explore:
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
