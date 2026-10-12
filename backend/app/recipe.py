"""The recipe: every data-changing step, written down so it can be replayed and reused.

Each step works in two moments (the "sticky note" idea):
  learn(train)          looks at the TRAINING rows only and returns what it needs to remember
  apply(df, learned)    makes the change using only what was learned, on any rows

Kinds of step:
  fixed       nothing to learn (drop a column, lowercase text)
  learn       remembers numbers/categories from the training rows (fill blanks, encode, scale)
  train_only  only ever applied to the training rows (removing duplicate rows)

Sticky notes are plain, readable values (numbers, lists, dicts), so the notebook can show them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd


# ── Registry ────────────────────────────────────────────────────────────────────

STEP_TYPES: Dict[str, Type["Step"]] = {}


def step_type(name: str):
    def decorator(cls):
        cls.name = name
        STEP_TYPES[name] = cls
        return cls
    return decorator


@dataclass
class Context:
    """What every step may know about the session besides the data itself."""
    target: Optional[str] = None


class Step:
    name = "step"
    kind = "fixed"  # "fixed" | "learn" | "train_only"

    def __init__(self, args: Dict[str, Any], ctx: Context):
        self.args = args
        self.ctx = ctx

    def learn(self, train: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def apply(self, df: pd.DataFrame, learned: Dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError

    def summary(self, before: pd.DataFrame, after: pd.DataFrame, learned: Dict[str, Any]) -> str:
        return f"Applied {self.name}."

    @property
    def train_only(self) -> bool:
        """Skip this step for test rows and new customers?"""
        return self.kind == "train_only"

    # helpers
    def _require_column(self, df: pd.DataFrame, col: Optional[str], arg: str = "column") -> str:
        if not col:
            raise ValueError(f"Missing '{arg}' argument.")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available columns: {', '.join(df.columns)}")
        return col

    def _features(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c != self.ctx.target]


# ── The recipe ──────────────────────────────────────────────────────────────────

def _new_id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass
class Line:
    """One recipe line: which step, with which settings, and its sticky note once learned.

    `id` stays the same when the line is edited or other lines are deleted, so a notebook
    cell can always find its step even after positions shift.
    """
    tool: str
    args: Dict[str, Any]
    learned: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    id: str = field(default_factory=_new_id)
    # The columns and types this line saw last time it ran (no rows): lets its code be written
    # with the right column references and value types. Not saved; replaying fills it in again.
    schema: Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "tool": self.tool, "args": self.args}


def run_line(line: Line, df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """Learn from these (training) rows, apply, and record the sticky note and a summary."""
    step = STEP_TYPES[line.tool](line.args, ctx)
    line.schema = df.head(0)
    line.learned = step.learn(df)
    after = step.apply(df.copy(), line.learned)
    line.message = step.summary(df, after, line.learned)
    return after


def replay(raw_train: pd.DataFrame, lines: List[Line], ctx: Context) -> pd.DataFrame:
    """Rebuild the training rows from scratch by running every line in order."""
    df = raw_train.copy()
    for line in lines:
        df = run_line(line, df, ctx)
    return df


def apply_to(df: pd.DataFrame, lines: List[Line], ctx: Context) -> pd.DataFrame:
    """Apply learned lines to other rows (test rows, new customers). Never learns anything new."""
    df = df.copy()
    for line in lines:
        step = STEP_TYPES[line.tool](line.args, ctx)
        if step.train_only:
            continue
        df = step.apply(df, line.learned)
    return df


# ── Fixed steps ─────────────────────────────────────────────────────────────────

@step_type("remove_columns")
class RemoveColumns(Step):
    kind = "fixed"

    def apply(self, df, learned):
        cols = [c for c in self.args.get("columns", []) if c in df.columns]
        return df.drop(columns=cols)

    def summary(self, before, after, learned):
        asked = self.args.get("columns", [])
        removed = [c for c in asked if c in before.columns]
        missing = [c for c in asked if c not in before.columns]
        msg = f"Removed columns: {removed}."
        if missing:
            msg += f" Not found (skipped): {missing}."
        return msg

    def learn(self, train):
        if not self.args.get("columns"):
            raise ValueError("No columns specified.")
        if self.ctx.target in self.args["columns"]:
            raise ValueError(f"'{self.ctx.target}' is the target column and can't be removed.")
        return {}


@step_type("clean_numeric_string")
class CleanNumericString(Step):
    kind = "fixed"

    def learn(self, train):
        self._require_column(train, self.args.get("column"))
        return {}

    def apply(self, df, learned):
        col = self.args["column"]
        if col in df.columns:
            df[col] = clean_number(df[col])
        return df

    def summary(self, before, after, learned):
        col = self.args["column"]
        ranges = int(before[col].astype(str).str.match(RANGE_PATTERN).sum())
        new_blanks = int(after[col].isna().sum() - before[col].isna().sum())
        msg = f"Cleaned '{col}': {before[col].dtype} → {after[col].dtype}."
        if ranges:
            msg += f" {ranges} ranges (like '2100 - 2850') became their midpoint."
        if new_blanks:
            msg += f" {new_blanks} values had no usable number and became blanks."
        return msg


# "2100 - 2850", "1,200-1,500": a range, read as its midpoint
RANGE_PATTERN = r"^\s*(\d[\d,]*\.?\d*)\s*-\s*(\d[\d,]*\.?\d*)\s*$"


def clean_number(series: pd.Series) -> pd.Series:
    """Text to numbers: "$1,200" -> 1200, "2 BHK" -> 2, "2100 - 2850" -> 2475 (the midpoint)."""
    text = series.astype(str)
    ends = text.str.extract(RANGE_PATTERN)
    low = pd.to_numeric(ends[0].str.replace(",", ""), errors="coerce")
    high = pd.to_numeric(ends[1].str.replace(",", ""), errors="coerce")
    plain = pd.to_numeric(text.str.replace(r"[^\d.\-]", "", regex=True), errors="coerce").where(series.notna())
    is_range = low.notna() & high.notna()
    if not is_range.any():
        return plain  # no ranges: exactly as before (whole numbers stay whole numbers)
    return ((low + high) / 2).where(is_range, plain)


@step_type("standardize_text")
class StandardizeText(Step):
    kind = "fixed"

    def learn(self, train):
        self._require_column(train, self.args.get("column"))
        return {}

    def apply(self, df, learned):
        col = self.args["column"]
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.lower().str.strip())
        return df

    def summary(self, before, after, learned):
        return f"Standardized text in '{self.args['column']}' (lowercased, trimmed spaces)."


@step_type("log_transform")
class LogTransform(Step):
    kind = "fixed"

    def learn(self, train):
        col = self._require_column(train, self.args.get("column"))
        if not pd.api.types.is_numeric_dtype(train[col]):
            raise ValueError(f"Column '{col}' is not numeric. Cannot apply log transform.")
        if (train[col].dropna() < 0).any():
            raise ValueError(f"Column '{col}' has negative values; log1p needs values ≥ 0.")
        return {}

    def apply(self, df, learned):
        col = self.args["column"]
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
        return df

    def summary(self, before, after, learned):
        col = self.args["column"]
        return f"Log-transformed '{col}'. Skewness: {before[col].skew():.3f} → {after[col].skew():.3f}."


# ── Training-only steps ─────────────────────────────────────────────────────────

@step_type("remove_duplicates")
class RemoveDuplicates(Step):
    kind = "train_only"

    def apply(self, df, learned):
        return df.drop_duplicates().reset_index(drop=True)

    def summary(self, before, after, learned):
        return f"Removed {len(before) - len(after)} duplicate rows. Row count: {len(before)} → {len(after)}."


@step_type("drop_missing_rows")
class DropMissingRows(Step):
    """Drop training rows that have blanks (in the chosen columns, or in any column).

    Training rows only: you can't refuse to predict for a customer because a field is blank, so
    test rows and new customers keep their rows and still need a fill rule for those blanks.
    """
    kind = "train_only"

    def _columns(self, df) -> List[str]:
        return [c for c in self.args.get("columns") or [] if c in df.columns]

    def learn(self, train):
        for c in self.args.get("columns") or []:
            self._require_column(train, c, "columns")
        return {}

    def apply(self, df, learned):
        return df.dropna(subset=self._columns(df) or None).reset_index(drop=True)

    def summary(self, before, after, learned):
        where = f"in {', '.join(self._columns(before))}" if self._columns(before) else "in any column"
        dropped = len(before) - len(after)
        msg = f"Dropped {dropped} training rows with blanks {where}. Row count: {len(before)} → {len(after)}."
        if dropped / max(len(before), 1) > 0.2:
            msg += f" That's {dropped / len(before):.0%} of the training rows: filling the blanks might keep more to learn from."
        return (msg + " Test rows and new customers aren't dropped (you still have to predict for them), "
                      "so their blanks need a fill step, or they'll be filled with 0 when predicting.")


@step_type("drop_rows_where")
class DropRowsWhere(Step):
    """Drop training rows matching a condition (e.g. outliers: price > 10000).

    Training rows only, like dropping duplicates: a model may learn better without odd rows, but you
    can't refuse to predict for an odd customer, so the test rows keep theirs (an honest exam).
    """
    kind = "train_only"

    def learn(self, train):
        from app.expressions import parse_filter
        parse_filter(train, self.args.get("conditions", ""))  # check it reads before anything runs
        return {}

    def apply(self, df, learned):
        from app.expressions import parse_filter
        drop, _ = parse_filter(df, self.args["conditions"])
        return df[~drop].reset_index(drop=True)

    def summary(self, before, after, learned):
        from app.expressions import parse_filter
        _, described = parse_filter(before, self.args["conditions"])
        dropped = len(before) - len(after)
        msg = f"Dropped {dropped} training rows where {described}. Row count: {len(before)} → {len(after)}."
        if dropped / max(len(before), 1) > 0.2:
            msg += f" That's {dropped / len(before):.0%} of the training rows: is the condition right?"
        return msg + " Test rows and new customers keep theirs: you still have to predict for them."


@step_type("change_type")
class ChangeType(Step):
    """astype: number (decimal), whole number, or text. Values that can't convert become blanks."""
    kind = "fixed"
    TYPES = ("number", "whole number", "text")

    def learn(self, train):
        self._require_column(train, self.args.get("column"))
        if self.args.get("to") not in self.TYPES:
            raise ValueError(f"Convert to one of: {', '.join(self.TYPES)}.")
        if self.args["column"] == self.ctx.target:
            raise ValueError("The target column's type is set when it's chosen; change the other columns.")
        return {}

    def apply(self, df, learned):
        col, to = self.args["column"], self.args["to"]
        if col in df.columns:
            if to == "text":
                df[col] = df[col].astype(str).where(df[col].notna())
            else:
                values = pd.to_numeric(df[col], errors="coerce")
                df[col] = values.round().astype("Int64") if to == "whole number" else values.astype(float)
        return df

    def summary(self, before, after, learned):
        col = self.args["column"]
        lost = int(after[col].isna().sum() - before[col].isna().sum())
        msg = f"Changed '{col}' to {self.args['to']}: {before[col].dtype} → {after[col].dtype}."
        if lost > 0:
            msg += f" {lost} values couldn't be converted and became blanks."
        return msg


@step_type("rename_column")
class RenameColumn(Step):
    kind = "fixed"

    def learn(self, train):
        col = self._require_column(train, self.args.get("column"))
        new = (self.args.get("new_name") or "").strip()
        if not new:
            raise ValueError("Give the new name.")
        if col == self.ctx.target:
            raise ValueError("The target column can't be renamed (the lock, the exam and predictions refer to it).")
        if new in train.columns and new != col:
            raise ValueError(f"There's already a column called '{new}'.")
        return {}

    def apply(self, df, learned):
        return df.rename(columns={self.args["column"]: self.args["new_name"].strip()})

    def summary(self, before, after, learned):
        return f"Renamed '{self.args['column']}' to '{self.args['new_name'].strip()}'."


@step_type("fill_value")
class FillValue(Step):
    """fillna with a value you choose (0, 'unknown', …). Nothing is learned: the same value everywhere."""
    kind = "fixed"

    def _value(self, series):
        raw = self.args.get("value")
        if pd.api.types.is_numeric_dtype(series):
            try:
                number = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"'{series.name}' holds numbers, so fill it with a number (not '{raw}').")
            if pd.api.types.is_integer_dtype(series):
                if number != round(number):
                    raise ValueError(f"'{series.name}' holds whole numbers, so fill it with a whole number.")
                return int(number)
            return number
        return str(raw)

    def learn(self, train):
        col = self._require_column(train, self.args.get("column"))
        if self.args.get("value") in (None, ""):
            raise ValueError("Give the value to fill blanks with.")
        self._value(train[col])
        return {}

    def apply(self, df, learned):
        col = self.args["column"]
        if col in df.columns:
            df[col] = df[col].fillna(self._value(df[col]))
        return df

    def summary(self, before, after, learned):
        col = self.args["column"]
        return f"Filled {int(before[col].isna().sum())} blanks in '{col}' with {self._value(before[col])!r}."


@step_type("add_column")
class AddColumn(Step):
    """A new column from a formula of existing ones (feature engineering), e.g. QUANTITY * PRICE.

    Row by row, so nothing is learned from other rows: it's the same on training rows, test rows and
    new customers. The formula is read safely by app/expressions.py, never run as Python.
    """
    kind = "fixed"

    def learn(self, train):
        from app.expressions import evaluate_formula
        name = (self.args.get("name") or "").strip()
        if not name:
            raise ValueError("Give the new column a name.")
        if name in train.columns:
            raise ValueError(f"There's already a column called '{name}'.")
        formula = self.args.get("formula", "")
        try:
            evaluate_formula(self._without_target(train), formula)
        except ValueError:
            if self.ctx.target and self.ctx.target in train.columns:
                try:  # does it only fail because it uses the target?
                    evaluate_formula(train, formula)
                except ValueError:
                    raise
                raise ValueError(
                    f"'{self.ctx.target}' is the column you're predicting, so it can't go into a new feature: "
                    f"for a new row you won't know its {self.ctx.target} (that's what the model is for), so "
                    f"the feature couldn't be made. To spot odd values, use a view (Group & summarise, Filter "
                    f"rows); to drop extreme rows from training, use Remove rows where…") from None
            raise
        return {}

    def _without_target(self, df):
        # The answer must never feed a feature: the model wouldn't have it when predicting
        return df.drop(columns=[self.ctx.target], errors="ignore")

    def apply(self, df, learned):
        from app.expressions import evaluate_formula
        df[self.args["name"].strip()] = evaluate_formula(self._without_target(df), self.args["formula"])
        return df

    def summary(self, before, after, learned):
        name = self.args["name"].strip()
        values = after[name]
        msg = f"Added '{name}' = {self.args['formula']}. Mean {values.mean():.4g}, range {values.min():.4g} to {values.max():.4g}."
        blanks = int(values.isna().sum())
        if blanks:
            msg += f" {blanks} rows are blank (a blank input, or dividing by zero)."
        return msg


# ── Text fixes (row by row: nothing learned, the same on every row set) ─────────

BLANK_WORDS = ("", "blank", "(blank)", "nothing", "none", "null", "nan")


@step_type("replace_values")
class ReplaceValues(Step):
    """Replace a value with another (Yes → Y), or with a blank (N/a → blank). Matches the whole
    value by default, or any part of the text. One column, or every text column."""
    kind = "fixed"

    def _columns(self, df) -> List[str]:
        if self.args.get("column"):
            return [self.args["column"]] if self.args["column"] in df.columns else []
        from app.tools import categorical_columns
        return [c for c in categorical_columns(df) if c != self.ctx.target]

    def _replacement(self):
        new = self.args.get("replace_with")
        return None if new is None or str(new).strip().lower() in BLANK_WORDS else str(new)

    def learn(self, train):
        if self.args.get("column"):
            self._require_column(train, self.args["column"])
        if self.args.get("find") in (None, ""):
            raise ValueError("Say which value to find.")
        if self.args.get("match", "whole value") not in ("whole value", "part of text"):
            raise ValueError("Match must be 'whole value' or 'part of text'.")
        return {}

    def apply(self, df, learned):
        find, new = str(self.args["find"]), self._replacement()
        whole = self.args.get("match", "whole value") == "whole value"
        for col in self._columns(df):
            text = df[col].astype(str).str.strip()
            if whole:
                hit = (text.str.lower() == find.strip().lower()) & df[col].notna()
                df[col] = df[col].where(~hit, new if new is not None else np.nan)
            else:
                replaced = df[col].astype(str).str.replace(find, new or "", regex=False)
                changed = df[col].notna() & (replaced != df[col].astype(str))
                cleaned = replaced.where(replaced.str.strip() != "", np.nan)  # emptied text becomes a blank
                df[col] = df[col].where(~changed, cleaned)
        return df

    def summary(self, before, after, learned):
        cols = self._columns(before)
        changed = sum(int((before[c].astype(str) != after[c].astype(str)).sum()) for c in cols if c in after)
        new = self._replacement()
        where = f"'{cols[0]}'" if self.args.get("column") else f"{len(cols)} text columns"
        return f"Replaced '{self.args['find']}' with {repr(new) if new is not None else 'a blank'} in {where}: {changed} values changed."


@step_type("strip_characters")
class StripCharacters(Step):
    """Remove chosen characters (e.g. ._/) from the ends of a text value, or from anywhere in it."""
    kind = "fixed"

    def learn(self, train):
        self._require_column(train, self.args.get("column"))
        if not self.args.get("characters"):
            raise ValueError("Say which characters to remove, e.g. ._/")
        if self.args.get("where", "ends") not in ("ends", "anywhere"):
            raise ValueError("Where must be 'ends' or 'anywhere'.")
        return {}

    def apply(self, df, learned):
        col, chars = self.args["column"], str(self.args["characters"])
        if col in df.columns:
            text = df[col].astype(str)
            if self.args.get("where", "ends") == "anywhere":
                text = text.str.translate(str.maketrans("", "", chars))
            else:
                text = text.str.strip(chars + " ")
            df[col] = text.where(df[col].notna())
        return df

    def summary(self, before, after, learned):
        col = self.args["column"]
        changed = int((before[col].astype(str) != after[col].astype(str)).sum())
        where = "anywhere in" if self.args.get("where") == "anywhere" else "the ends of"
        return f"Removed {self.args['characters']!r} from {where} '{col}': {changed} values changed."


@step_type("format_text")
class FormatText(Step):
    """Keep only some characters (digits, letters), then optionally lay them out with a pattern:
    "(123) 456/7890" → keep digits → "1234567890" → pattern ###-###-#### → "123-456-7890".
    A value with nothing left (e.g. "N/a" when keeping digits) becomes a blank."""
    kind = "fixed"
    KEEP = {"digits": r"[^0-9]", "letters": r"[^A-Za-z]", "letters and digits": r"[^A-Za-z0-9]"}

    def learn(self, train):
        self._require_column(train, self.args.get("column"))
        if self.args.get("keep", "digits") not in self.KEEP:
            raise ValueError(f"Keep must be one of: {', '.join(self.KEEP)}.")
        pattern = self.args.get("pattern") or ""
        if pattern and "#" not in pattern:
            raise ValueError("A pattern uses # for each character, e.g. ###-###-####.")
        return {}

    def _format(self, value: str) -> Any:
        pattern = self.args.get("pattern") or ""
        if not isinstance(value, str) or not value:
            return np.nan  # a blank, or nothing left after keeping
        slots = pattern.count("#")
        if not pattern or len(value) < slots:
            return value  # too short for the pattern: left as the kept characters
        chars, out = iter(value[:slots]), []
        for p in pattern:
            out.append(next(chars) if p == "#" else p)
        return "".join(out)

    def apply(self, df, learned):
        col = self.args["column"]
        if col in df.columns:
            kept = df[col].astype(str).str.replace(self.KEEP[self.args.get("keep", "digits")], "", regex=True)
            df[col] = kept.map(self._format).where(df[col].notna())
        return df

    def summary(self, before, after, learned):
        col, pattern = self.args["column"], self.args.get("pattern")
        blanks = int(after[col].isna().sum() - before[col].isna().sum())
        msg = f"Kept only {self.args.get('keep', 'digits')} in '{col}'" + (f", laid out as {pattern}" if pattern else "") + "."
        if pattern:
            slots = pattern.count("#")
            short = int(after[col].dropna().astype(str).str.len().lt(len(pattern)).sum())
            if short:
                msg += f" {short} values had fewer than {slots} characters and were left unformatted."
        if blanks:
            msg += f" {blanks} values had nothing left and became blanks."
        return msg


@step_type("split_column")
class SplitColumn(Step):
    """Split a text column on a separator into new columns: "1 Main St, NY, 10001" → street, state, zip.
    The last new column takes the rest if there are more pieces than names."""
    kind = "fixed"

    def _names(self) -> List[str]:
        return [n.strip() for n in str(self.args.get("into") or "").split(",") if n.strip()]

    def learn(self, train):
        col = self._require_column(train, self.args.get("column"))
        names = self._names()
        if len(names) < 2:
            raise ValueError("Give at least two names for the new columns, separated by commas, e.g. Street, State, Zip.")
        clash = [n for n in names if n in train.columns and n != col]
        if clash:
            raise ValueError(f"There are already columns called {', '.join(clash)}.")
        if col == self.ctx.target:
            raise ValueError("The column you're predicting can't be split.")
        return {}

    def apply(self, df, learned):
        col, names = self.args["column"], self._names()
        if col in df.columns:
            separator = self.args.get("separator") or ","
            parts = df[col].astype(str).where(df[col].notna()).str.split(separator, n=len(names) - 1, regex=False, expand=True)
            for i, name in enumerate(names):
                piece = parts[i] if i in parts.columns else pd.Series(np.nan, index=df.index)
                df[name] = piece.str.strip().replace("", np.nan)
        return df

    def summary(self, before, after, learned):
        names = self._names()
        filled = ", ".join(f"{n} ({int(after[n].notna().sum())} filled)" for n in names)
        return f"Split '{self.args['column']}' on {self.args.get('separator') or ','!r} into {filled}. The original column is kept."


@step_type("group_rare_categories")
class GroupRareCategories(Step):
    """Values seen in at most `max_rows` training rows become one label ('other'), e.g. the many
    locations with a handful of houses each. Which values count as common is LEARNED from the
    training rows: test rows and new customers use that same list (a value never seen in training
    becomes 'other' too)."""
    kind = "learn"

    def _settings(self):
        try:
            max_rows = int(self.args.get("max_rows", 10))
        except (TypeError, ValueError):
            raise ValueError("'max_rows' must be a whole number.")
        if max_rows < 1:
            raise ValueError("'max_rows' must be at least 1.")
        return max_rows, str(self.args.get("label") or "other")

    def learn(self, train):
        col = self._require_column(train, self.args.get("column"))
        if col == self.ctx.target:
            raise ValueError("Group the values of a feature, not of the column you're predicting.")
        max_rows, _ = self._settings()
        counts = train[col].dropna().astype(str).str.strip().value_counts()
        return {"keep": sorted(counts[counts > max_rows].index.tolist())}

    def apply(self, df, learned):
        col = self.args["column"]
        if col in df.columns:
            _, label = self._settings()
            values = df[col].astype(str).str.strip().where(df[col].notna())
            df[col] = values.where(values.isin(learned["keep"]) | values.isna(), label)
        return df

    def summary(self, before, after, learned):
        col = self.args["column"]
        max_rows, label = self._settings()
        was = before[col].nunique()
        return (f"'{col}': {was} values → {len(learned['keep'])} kept + '{label}' (values in {max_rows} or fewer "
                f"training rows). {int((after[col] == label).sum())} training rows are now '{label}'.")


@step_type("remove_outliers")
class RemoveOutliers(Step):
    """Drop training rows whose value is extreme within its group, e.g. price per sqft more than
    1 standard deviation from its location's average.

    Training rows only. That's also why the value may use the target (price / total_sqft): it only
    decides which rows the model learns from, it never becomes a feature, and test rows and new
    customers are never dropped (you still have to predict for them)."""
    kind = "train_only"
    METHODS = ("std", "iqr")

    def _settings(self):
        method = self.args.get("method") or "std"
        if method not in self.METHODS:
            raise ValueError("Method must be 'std' (standard deviations from the mean) or 'iqr' (interquartile range).")
        default = 3.0 if method == "std" else 1.5
        try:
            k = float(self.args.get("threshold") if self.args.get("threshold") not in (None, "") else default)
        except (TypeError, ValueError):
            raise ValueError("'threshold' must be a number.")
        if k <= 0:
            raise ValueError("'threshold' must be more than 0.")
        return method, k

    def _keep(self, df):
        from app.expressions import evaluate_formula
        value = evaluate_formula(df, self.args.get("value", ""))
        method, k = self._settings()
        group = self.args.get("group_by")
        groups = df[group].astype(str) if group else pd.Series("all", index=df.index)
        grouped = value.groupby(groups)
        if method == "std":
            center, spread = grouped.transform("mean"), grouped.transform("std")
            low, high = center - k * spread, center + k * spread
        else:
            q1, q3 = grouped.transform(lambda v: v.quantile(0.25)), grouped.transform(lambda v: v.quantile(0.75))
            low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        # Blank values, and groups too small to have a spread, are kept
        return value.isna() | low.isna() | high.isna() | value.between(low, high)

    def learn(self, train):
        if self.args.get("group_by"):
            self._require_column(train, self.args.get("group_by"), "group_by")
        self._settings()
        self._keep(train)  # check the formula reads before anything runs
        return {}

    def apply(self, df, learned):
        return df[self._keep(df)].reset_index(drop=True)

    def summary(self, before, after, learned):
        method, k = self._settings()
        rule = f"more than {k:g} standard deviations from the mean" if method == "std" else f"beyond {k:g}× the IQR"
        per = f" of its {self.args['group_by']}" if self.args.get("group_by") else ""
        dropped = len(before) - len(after)
        msg = (f"Dropped {dropped} training rows where {self.args['value']} is {rule}{per}. "
               f"Row count: {len(before)} → {len(after)}.")
        if dropped / max(len(before), 1) > 0.25:
            msg += f" That's {dropped / len(before):.0%} of the training rows: a larger threshold keeps more."
        return msg + " Test rows and new customers aren't dropped: you still have to predict for them."


# ── Learning steps ──────────────────────────────────────────────────────────────

def _fill_value(series: pd.Series, strategy: str) -> Any:
    if strategy in ("mean", "median"):
        value = series.mean() if strategy == "mean" else series.median()
        if pd.api.types.is_integer_dtype(series):
            # A whole-number column (e.g. rooms, after "change type") gets a whole number: 1.88 rooms -> 2
            return int(round(float(value)))
        return round(float(value), 6)
    mode = series.mode()
    if mode.empty:
        return 0
    value = mode.iloc[0]
    return value.item() if hasattr(value, "item") else value


def _check_strategy(series: pd.Series, col: str, strategy: str) -> None:
    if strategy not in ("mean", "median", "mode"):
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'mean', 'median', or 'mode'.")
    if strategy in ("mean", "median") and not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Cannot use '{strategy}' on non-numeric column '{col}'. Use 'mode' instead.")


@step_type("impute_missing")
class ImputeMissing(Step):
    kind = "learn"

    def learn(self, train):
        col = self._require_column(train, self.args.get("column"))
        strategy = self.args.get("strategy", "mean")
        _check_strategy(train[col], col, strategy)
        return {"fill": {col: _fill_value(train[col], strategy)}}

    def apply(self, df, learned):
        return df.fillna(learned["fill"])

    def summary(self, before, after, learned):
        (col, value), = learned["fill"].items()
        missing = int(before[col].isna().sum())
        if missing == 0:
            return f"Column '{col}' has no missing values; future blanks will be filled with {value}."
        return f"Filled {missing} blanks in '{col}' with {self.args.get('strategy', 'mean')} = {value}."


@step_type("impute_all_missing")
class ImputeAllMissing(Step):
    kind = "learn"

    def learn(self, train):
        requested = self.args.get("strategy")
        fill = {}
        for col in self._features(train):
            numeric = pd.api.types.is_numeric_dtype(train[col])
            strategy = requested or ("mean" if numeric else "mode")
            if strategy in ("mean", "median") and not numeric:
                strategy = "mode"
            _check_strategy(train[col], col, strategy)
            fill[col] = _fill_value(train[col], strategy)
        return {"fill": fill}

    def apply(self, df, learned):
        return df.fillna({c: v for c, v in learned["fill"].items() if c in df.columns})

    def summary(self, before, after, learned):
        filled = [f"  • {c}: {int(before[c].isna().sum())} blanks → {v}"
                  for c, v in learned["fill"].items() if before[c].isna().any()]
        if not filled:
            return "No blanks in the training rows. Fill values were still learned for future rows."
        return f"Filled blanks in {len(filled)} columns:\n" + "\n".join(filled)


def _categories(series: pd.Series) -> List[Any]:
    return sorted(series.dropna().astype(str).unique().tolist())


def _label_codes(df: pd.DataFrame, col: str, classes: List[str]) -> pd.Series:
    mapping = {c: i for i, c in enumerate(classes)}
    # Categories never seen in training get -1
    return df[col].astype(str).map(mapping).where(df[col].notna()).fillna(-1).astype(int)


def _onehot(df: pd.DataFrame, col: str, classes: List[str]) -> pd.DataFrame:
    values = df[col].astype(str).where(df[col].notna())
    columns = pd.DataFrame({f"{col}_{c}": (values == c).astype(int) for c in classes}, index=df.index)
    return pd.concat([df.drop(columns=[col]), columns], axis=1)


# One-hot makes one new column per category. Refused when that would be harmful:
#   ID-like: nearly every row has its own value (names, IDs): one column per passenger teaches nothing
#   too many categories, or too many new cells for the server's memory
# Real high-cardinality features (e.g. ~240 Bengaluru locations) are fine.
MAX_ONEHOT_CATEGORIES = 300
MAX_ONEHOT_CELLS = 30_000_000
ID_LIKE_MIN_CATEGORIES = 50


def _check_encoding(method: str) -> None:
    if method not in ("label", "onehot"):
        raise ValueError(f"Unknown encoding method '{method}'. Use 'label' or 'onehot'.")


def _check_onehot_size(classes: Dict[str, List[str]], rows: int) -> None:
    id_like = {c: len(v) for c, v in classes.items() if len(v) >= ID_LIKE_MIN_CATEGORIES and len(v) > 0.5 * rows}
    too_many = {c: len(v) for c, v in classes.items() if len(v) > MAX_ONEHOT_CATEGORIES and c not in id_like}
    if id_like:
        listed = ", ".join(f"'{c}' ({n} different values in {rows} rows)" for c, n in id_like.items())
        raise ValueError(
            f"{listed}: almost every row has its own value, so it looks like an ID, a name or free text. One-hot "
            "would add a column per row and teach the model nothing. Remove it, clean it into a number if it's "
            "really numeric (e.g. prices stored as text), or use label encoding.")
    if too_many:
        listed = ", ".join(f"'{c}' ({n} values)" for c, n in too_many.items())
        raise ValueError(
            f"{listed} has more than {MAX_ONEHOT_CATEGORIES} values. Group the rare ones into 'other' first "
            "(Group rare categories), or use label encoding.")
    cells = rows * sum(len(v) for v in classes.values())
    if cells > MAX_ONEHOT_CELLS:
        raise ValueError(
            f"One-hot here would add {cells:,} cells, more than this server allows ({MAX_ONEHOT_CELLS:,}). "
            "Group rare categories into 'other' first, or use label encoding.")


@step_type("encode_categorical")
class EncodeCategorical(Step):
    kind = "learn"

    def learn(self, train):
        col = self._require_column(train, self.args.get("column"))
        _check_encoding(self.args.get("method", "label"))
        classes = {col: _categories(train[col])}
        if self.args.get("method", "label") == "onehot":
            _check_onehot_size(classes, len(train))
        return {"classes": classes}

    def apply(self, df, learned):
        method = self.args.get("method", "label")
        for col, classes in learned["classes"].items():
            if col not in df.columns:  # e.g. the target, when predicting for new rows
                continue
            df = _onehot(df, col, classes) if method == "onehot" else df.assign(**{col: _label_codes(df, col, classes)})
        return df

    def summary(self, before, after, learned):
        (col, classes), = learned["classes"].items()
        shown = ", ".join(classes[:10]) + ("…" if len(classes) > 10 else "")
        if self.args.get("method", "label") == "onehot":
            return f"One-hot encoded '{col}' into {len(classes)} columns ({shown})."
        return f"Label-encoded '{col}': {shown} → 0…{len(classes) - 1}."


@step_type("encode_all_categorical")
class EncodeAllCategorical(Step):
    kind = "learn"

    def learn(self, train):
        from app.tools import categorical_columns
        _check_encoding(self.args.get("method", "label"))
        cols = [c for c in categorical_columns(train) if c != self.ctx.target]
        classes = {c: _categories(train[c]) for c in cols}
        if self.args.get("method", "label") == "onehot":
            _check_onehot_size(classes, len(train))
        return {"classes": classes}

    apply = EncodeCategorical.apply

    def summary(self, before, after, learned):
        if not learned["classes"]:
            return "No categorical columns to encode (all features are already numeric)."
        method = self.args.get("method", "label")
        lines = [f"  • {c}: {len(cl)} categories" for c, cl in learned["classes"].items()]
        return f"Encoded {len(lines)} categorical columns ({method}):\n" + "\n".join(lines)


def _numeric_features(df: pd.DataFrame, target: Optional[str]) -> List[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != target]


@step_type("scale_features")
class ScaleFeatures(Step):
    kind = "learn"

    def _columns(self, train) -> List[str]:
        cols = self.args.get("columns") or _numeric_features(train, self.ctx.target)
        for c in cols:
            self._require_column(train, c, "columns")
            if not pd.api.types.is_numeric_dtype(train[c]):
                raise ValueError(f"Column '{c}' is not numeric, so it can't be scaled.")
        if not cols:
            raise ValueError("No numeric columns available to scale.")
        return cols

    def learn(self, train):
        stats = {}
        for c in self._columns(train):
            std = float(train[c].std(ddof=0))
            stats[c] = {"mean": round(float(train[c].mean()), 6), "std": round(std, 6) if std > 0 else 1.0}
        return {"scale": stats}

    def apply(self, df, learned):
        for c, s in learned["scale"].items():
            if c in df.columns:
                df[c] = (df[c] - s["mean"]) / s["std"]
        return df

    def summary(self, before, after, learned):
        cols = list(learned["scale"])
        return f"Scaled {len(cols)} features to mean 0 / spread 1: {', '.join(cols)}."


@step_type("scale_all_features")
class ScaleAllFeatures(ScaleFeatures):
    def _columns(self, train):
        cols = _numeric_features(train, self.ctx.target)
        if not cols:
            raise ValueError("No numeric columns available to scale.")
        return cols


# ── Custom code ─────────────────────────────────────────────────────────────────

CUSTOM_TEMPLATE = '''import numpy as np
import pandas as pd

TRAIN_ONLY = False  # True = only change the training rows (e.g. when dropping rows)


def learn(train):
    # Look at the TRAINING rows and return what apply() needs to remember
    return {}


def apply(df, learned):
    # Change the rows using only `learned`: this also runs on test rows and new customers
    return df
'''

CODE_FILENAME = "<your code>"


def _plain(value: Any) -> Any:
    """Make learned values plain (JSON-safe) so they can be shown, saved and replayed."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict"):  # pandas Series / DataFrame
        return _plain(value.to_dict())
    if hasattr(value, "tolist"):  # numpy arrays
        return _plain(value.tolist())
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _user_error(e: Exception) -> ValueError:
    """Turn an exception from user code into a readable message with the line number."""
    import traceback
    lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == CODE_FILENAME]
    where = f" (line {lines[-1]})" if lines else ""
    return ValueError(f"Your code raised {type(e).__name__}: {e}{where}")


@step_type("custom_code")
class CustomCode(Step):
    """A step written by the user. Runs on this machine with full access, like a Jupyter cell."""
    kind = "custom"

    def _namespace(self) -> Dict[str, Any]:
        from app.config import settings
        if not settings.ALLOW_CUSTOM_CODE:
            raise ValueError("Custom code is switched off on this server (ALLOW_CUSTOM_CODE=false).")
        code = self.args.get("code") or ""
        namespace: Dict[str, Any] = {"pd": pd, "np": np}
        try:
            exec(compile(code, CODE_FILENAME, "exec"), namespace)
        except SyntaxError as e:
            raise ValueError(f"Your code has a syntax error on line {e.lineno}: {e.msg}") from None
        except Exception as e:
            raise _user_error(e) from None
        for fn in ("learn", "apply"):
            if not callable(namespace.get(fn)):
                raise ValueError(f"Your code must define a function called {fn}().")
        return namespace

    @property
    def train_only(self) -> bool:
        return bool(self._namespace().get("TRAIN_ONLY", False))

    def learn(self, train):
        try:
            learned = self._namespace()["learn"](train.copy())
        except ValueError:
            raise
        except Exception as e:
            raise _user_error(e) from None
        if learned is None:
            learned = {}
        if not isinstance(learned, dict):
            raise ValueError(f"learn() must return a dict (it returned {type(learned).__name__}).")
        return _plain(learned)

    def apply(self, df, learned):
        try:
            out = self._namespace()["apply"](df, learned)
        except ValueError:
            raise
        except Exception as e:
            raise _user_error(e) from None
        if not isinstance(out, pd.DataFrame):
            raise ValueError(f"apply() must return the rows as a DataFrame (it returned {type(out).__name__}).")
        return out

    def summary(self, before, after, learned):
        added = [c for c in after.columns if c not in before.columns]
        dropped = [c for c in before.columns if c not in after.columns]
        parts = [f"Custom code ran: {len(before)} → {len(after)} rows"]
        if added:
            parts.append(f"added columns {added}")
        if dropped:
            parts.append(f"removed columns {dropped}")
        return ", ".join(parts) + "."


# Tools that change data and therefore become recipe lines
RECIPE_TOOLS = set(STEP_TYPES)
