"""Exports: cleaned data (CSV / Excel) and the whole recipe as one runnable Python script."""

import io
import re
from typing import Tuple

import numpy as np
import pandas as pd

from app.action_code import _stratifies
from app.recipe import STEP_TYPES
from app.recipe_code import step_code
from app.session import DataSession


def cleaned_rows(session: DataSession, which: str) -> pd.DataFrame:
    """'train': the cleaned training rows. 'all': training + locked test rows, the test rows cleaned
    with the training sticky notes (never learned from), labelled in a `split` column."""
    if which == "train" or not session.is_split:
        return session.df.copy()
    train = session.df.assign(split="train")
    test = session.prepare(session.test_df).assign(split="test")
    return pd.concat([train, test], ignore_index=True)


def to_file(df: pd.DataFrame, fmt: str) -> Tuple[bytes, str]:
    """(file bytes, media type) for CSV or Excel."""
    if fmt == "xlsx":
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return df.to_csv(index=False).encode("utf-8"), "text/csv"


def _numbered(code: str, n: int) -> str:
    """Rename a step's learn/apply (and TRAIN_ONLY) so many steps can live in one file."""
    code = re.sub(r"^\s*import (numpy|pandas).*\n", "", code, flags=re.M)
    code = re.sub(r"\bdef learn\(", f"def step{n}_learn(", code)
    code = re.sub(r"\bdef apply\(", f"def step{n}_apply(", code)
    code = re.sub(r"\bTRAIN_ONLY\b", f"STEP{n}_TRAIN_ONLY", code)
    return code.strip()


def pipeline_script(session: DataSession) -> str:
    """The notebook's recipe as a standalone script that reproduces the app's cleaned data."""
    if session.clean_only:
        return _cleaning_script(session)
    target, size = session.target_column, session.test_size or 0.2
    stratify = f"df[{target!r}]" if _stratifies(session, size) else "None"
    steps = session.recipe

    parts = [f'''"""Data-cleaning pipeline exported from Rhombus.

Dataset: {session.filename}
Target:  {target}
Steps:   {len(steps)} ({", ".join(l.tool for l in steps) or "none"})

Every step has two parts:
  learn(train)         looks at the TRAINING rows only and returns what it needs (the "sticky note")
  apply(df, learned)   uses only that sticky note, so test rows and new data are never learned from

Run:  python {re.sub(r"[^A-Za-z0-9_.-]", "_", session.filename.rsplit(".", 1)[0])}_pipeline.py
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
''']

    for n, line in enumerate(steps, 1):
        kind = STEP_TYPES[line.tool].kind
        note = "  (training rows only)" if kind == "train_only" else ""
        parts.append(f"\n# ── Step {n}: {line.tool.replace('_', ' ')}{note} " + "─" * 20 + "\n\n"
                     + _numbered(step_code(line, target, with_learned=False), n))

    learn_lines = "\n".join(
        f"    notes[{n}] = step{n}_learn(train)\n    train = step{n}_apply(train, notes[{n}])"
        for n in range(1, len(steps) + 1)
    ) or "    pass"
    apply_lines = []
    for n, line in enumerate(steps, 1):
        if STEP_TYPES[line.tool].kind == "train_only":
            apply_lines.append(f"    # step {n} ({line.tool.replace('_', ' ')}) is for training rows only")
        elif line.tool == "custom_code":
            apply_lines.append(f"    if not globals().get('STEP{n}_TRAIN_ONLY', False):\n"
                               f"        df = step{n}_apply(df, notes[{n}])")
        else:
            apply_lines.append(f"    df = step{n}_apply(df, notes[{n}])")

    parts.append(f'''

# ── The recipe ──────────────────────────────────────────────────────────────────

def fit_recipe(train):
    """Learn every step from the training rows (in order) and apply it. Returns (cleaned, notes)."""
    train = train.copy()
    notes = {{}}
{learn_lines}
    return train, notes


def apply_recipe(df, notes):
    """Clean other rows (the test set, new data) with the training sticky notes. Learns nothing."""
    df = df.copy()
{chr(10).join(apply_lines) or "    pass"}
    return df


def main(path={session.filename!r}):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    df = df.dropna(subset=[{target!r}])
    if df[{target!r}].dtype.kind == "f" and (df[{target!r}] == df[{target!r}].round()).all():
        df[{target!r}] = df[{target!r}].astype("int64")  # 0.0/1.0 back to 0/1 once the blanks are gone
    train, test = train_test_split(df, test_size={size}, random_state=42, stratify={stratify})
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    train_clean, notes = fit_recipe(train)
    test_clean = apply_recipe(test, notes)
    train_clean.to_csv("train_clean.csv", index=False)
    test_clean.to_csv("test_clean.csv", index=False)
    print(f"Saved train_clean.csv ({{len(train_clean)}} rows) and test_clean.csv ({{len(test_clean)}} rows)")
    return train_clean, test_clean


if __name__ == "__main__":
    main(*sys.argv[1:])
''')
    return "\n".join(parts)


def _cleaning_script(session: DataSession) -> str:
    """Clean-only notebooks: every step on the whole file, no split (there's no model to test)."""
    steps = session.recipe
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", session.filename.rsplit(".", 1)[0])
    parts = [f'''"""Data-cleaning script exported from Rhombus (clean-only notebook).

Dataset: {session.filename}
Steps:   {len(steps)} ({", ".join(l.tool for l in steps) or "none"})

There's no model, so every step runs on all the rows, in order.
Run:  python {stem}_pipeline.py
"""

import sys

import numpy as np
import pandas as pd
''']
    for n, line in enumerate(steps, 1):
        parts.append(f"\n# ── Step {n}: {line.tool.replace('_', ' ')} " + "─" * 20 + "\n\n"
                     + _numbered(step_code(line, None, with_learned=False), n))
    runs = "\n".join(f"    df = step{n}_apply(df, step{n}_learn(df))" for n in range(1, len(steps) + 1)) or "    pass"
    parts.append(f'''

def clean(df):
    """Every step, in order, on all the rows."""
    df = df.copy()
{runs}
    return df


def main(path={session.filename!r}):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    cleaned = clean(df)
    cleaned.to_csv({stem + "_clean.csv"!r}, index=False)
    print(f"Saved {stem}_clean.csv ({{len(cleaned)}} rows)")
    return cleaned


if __name__ == "__main__":
    main(*sys.argv[1:])
''')
    return "\n".join(parts)
