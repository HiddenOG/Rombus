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
