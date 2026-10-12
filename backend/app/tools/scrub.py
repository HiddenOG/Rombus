"""Scrub tools — data cleaning, imputation, encoding, and transformation.

Every data-changing tool here is a thin wrapper: it adds a line to the session's recipe
(app/recipe.py), where the actual learn/apply logic lives. That keeps every change
replayable on the locked test rows and on new customers.
"""

from typing import Any, Dict, List, Optional

from app.recipe import CUSTOM_TEMPLATE, STEP_TYPES
from app.session import DataSession
from app.tools import COLUMN, COLUMNS, register_tool

STRATEGY = {"type": "string", "enum": ["mean", "median", "mode"]}
ENCODING = {"type": "string", "enum": ["label", "onehot"]}


def recipe_tool(name: str, description: str, properties: Optional[Dict[str, Any]] = None,
                required: Optional[List[str]] = None, ai: bool = True) -> None:
    """Register a tool that adds a step named `name` to the recipe."""
    def run(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
        line = session.add_step(name, args)
        return {"success": True, "message": line.message, "step_id": line.id}

    run.__name__ = name
    register_tool(name, description, properties, required, kind=STEP_TYPES[name].kind, ai=ai)(run)


recipe_tool("remove_columns", "Drop one or more columns.", {"columns": COLUMNS}, ["columns"])

recipe_tool("remove_duplicates", "Remove duplicate rows from the training rows.")

recipe_tool(
    "drop_missing_rows",
    "Drop training rows with blanks in the given columns (or in any column if none given). "
    "Good when few rows are affected; otherwise prefer filling (impute).",
    {"columns": {**COLUMNS, "description": "Only rows blank in these columns (optional)."}},
)

recipe_tool(
    "drop_rows_where",
    "Drop TRAINING rows matching a condition, e.g. outliers (test rows and new customers keep theirs).",
    {"conditions": {"type": "string",
                    "description": "e.g. price > 10000 or bath > 8. Operators: > >= < <= == != contains, "
                                   "is blank; join with and / or."}},
    ["conditions"],
)

recipe_tool(
    "change_type",
    "Convert a column's type, like astype: number, whole number or text (unconvertible values become blanks).",
    {"column": COLUMN, "to": {"type": "string", "enum": ["number", "whole number", "text"]}},
    ["column", "to"],
)

recipe_tool(
    "rename_column",
    "Rename a column.",
    {"column": COLUMN, "new_name": {"type": "string", "description": "The new name."}},
    ["column", "new_name"],
)

recipe_tool(
    "fill_value",
    "Fill a column's blanks with a fixed value you choose (e.g. 0 or 'unknown'), like fillna(value).",
    {"column": COLUMN, "value": {"type": "string", "description": "The value to put in the blanks."}},
    ["column", "value"],
)

recipe_tool(
    "group_rare_categories",
    "Merge rare values of a text feature into one label ('other'), e.g. locations with 10 or fewer rows. "
    "Which values are common is learned from the training rows.",
    {"column": COLUMN,
     "max_rows": {"type": "integer", "description": "Values in this many training rows or fewer are merged (default 10)."},
     "label": {"type": "string", "description": "Name for the merged group (default 'other')."}},
    ["column"],
)

recipe_tool(
    "remove_outliers",
    "Drop TRAINING rows whose value is extreme within its group, e.g. price / total_sqft more than 1 "
    "standard deviation from its location's mean. The value may use the target (it only filters training rows).",
    {"value": {"type": "string",
               "description": "What to check: a name or formula, e.g. price or price / total_sqft."},
     "group_by": {"type": "string", "description": "Optional: compare within each group of this column, e.g. location."},
     "method": {"type": "string", "enum": ["std", "iqr"]},
     "threshold": {"type": "number", "description": "How many standard deviations (default 3) or IQRs (default 1.5)."}},
    ["value"],
)

recipe_tool(
    "add_column",
    "Create a new feature from a formula of existing numeric ones, e.g. total_sqft / bath.",
    {"name": {"type": "string", "description": "Name for the new feature."},
     "formula": {"type": "string",
                 "description": "e.g. QUANTITY * PRICE or log(total_sqft). + - * / ** > <, numbers, and "
                                "log log1p sqrt abs exp round floor ceil. Names with spaces in `backticks`."}},
    ["name", "formula"],
)

recipe_tool(
    "clean_numeric_string",
    "Strip $, commas and other non-numeric characters from a column and convert it to numeric.",
    {"column": COLUMN},
    ["column"],
)

recipe_tool("standardize_text", "Lowercase and strip whitespace in a text column.", {"column": COLUMN}, ["column"])

recipe_tool(
    "impute_missing",
    "Fill missing values in ONE column. mean/median need a numeric column.",
    {"column": COLUMN, "strategy": STRATEGY},
    ["column"],
)

recipe_tool(
    "impute_all_missing",
    "Fill missing values in ALL columns at once. Without a strategy, uses mean for numeric and mode for "
    "categorical. Prefer this over repeated impute_missing calls.",
    {"strategy": STRATEGY},
)

recipe_tool(
    "encode_categorical",
    "Encode ONE categorical column with label or one-hot encoding.",
    {"column": COLUMN, "method": ENCODING},
    ["column"],
)

recipe_tool(
    "encode_all_categorical",
    "Encode ALL categorical columns at once (skips the target). Prefer this over repeated encode_categorical calls.",
    {"method": ENCODING},
)

recipe_tool(
    "scale_features",
    "Standard-scale the given numeric columns (all numeric features if none given).",
    {"columns": COLUMNS},
)

recipe_tool("scale_all_features", "Standard-scale ALL numeric features at once (skips the target).")

recipe_tool("log_transform", "Apply log1p to a skewed numeric column.", {"column": COLUMN}, ["column"])

# People only: the AI is never offered this (running AI-written code needs a sandbox first)
recipe_tool(
    "custom_code",
    "Your own Python step: learn(train) reads the training rows, apply(df, learned) changes any rows.",
    {"code": {"type": "string", "format": "code", "description": "Python defining learn(train) and apply(df, learned).",
              "default": CUSTOM_TEMPLATE}},
    ["code"],
    ai=False,
)


@register_tool("check_duplicates", "Count duplicate rows.")
def check_duplicates(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Look-only: counts duplicates, changes nothing."""
    dup_count = int(session.df.duplicated().sum())
    return {
        "success": True,
        "message": f"Found {dup_count} duplicate rows out of {len(session.df)} total rows.",
        "evidence": {"duplicate_count": dup_count, "total_rows": len(session.df)},
    }
