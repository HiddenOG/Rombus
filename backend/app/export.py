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
    if not session.is_split and session.sealed and which == "all":
        # Every row, the sealed ones too: that opens the envelope (they've been seen from now on)
        session.open_envelope()
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


# Reading the data exactly as the upload did: any common encoding and separator, several files
# stacked with a `source_file` column
LOAD_CODE = '''

# ── Reading the data (as Rhombus read it) ──────────────────────────────────────

def read_table(path):
    if not path.lower().endswith(".csv"):
        df = pd.read_excel(path)
    else:
        raw = open(path, "rb").read()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        if df.shape[1] == 1:  # separated by something other than commas?
            header = text.split("\\n", 1)[0]
            sep = max([";", "\\t", "|"], key=header.count)
            if header.count(sep):
                df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [str(c) for c in df.columns]
    return df


def load(paths):
    """One file, or several stacked (rows one after the other) with a source_file column."""
    if len(paths) == 1:
        return read_table(paths[0])
    return pd.concat([read_table(p).assign(source_file=os.path.splitext(os.path.basename(p))[0]) for p in paths],
                     ignore_index=True)
'''


def _data_files(session: DataSession) -> list:
    """The file(s) the notebook was made from, as the script's default arguments."""
    if " + " in session.filename and "source_file" in session.original_df:
        return [f"{s}.csv" for s in session.original_df["source_file"].dropna().unique()]
    return [session.filename]


def _run_line(session: DataSession, stem: str) -> str:
    files = " ".join(f'"{f}"' if " " in f else f for f in _data_files(session))
    return f"Run:  python {stem}_pipeline.py {files}"


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
    if session.time_column:
        tc = session.time_column
        split_code = (
            f"    # Time split: oldest rows train, the most recent {size:.0%} are the test rows\n"
            f"    times = pd.to_datetime(df[{tc!r}], errors='coerce', format='mixed')\n"
            f"    df = df.loc[times[times.notna()].sort_values(kind='stable').index]\n"
            f"    n_test = max(1, int(np.ceil(len(df) * {size})))\n"
            f"    train, test = df.iloc[:-n_test], df.iloc[-n_test:]")
    else:
        split_code = f"    train, test = train_test_split(df, test_size={size}, random_state=42, stratify={stratify})"

    parts = [f'''"""Data-cleaning pipeline exported from Rhombus.

Dataset: {session.filename}
Target:  {target}
Steps:   {len(steps)} ({", ".join(l.tool for l in steps) or "none"})

Every step has two parts:
  learn(train)         looks at the TRAINING rows only and returns what it needs (the "sticky note")
  apply(df, learned)   uses only that sticky note, so test rows and new data are never learned from

{_run_line(session, re.sub(r"[^A-Za-z0-9_.-]", "_", session.filename.rsplit(".", 1)[0]))}
"""

import io
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
''' + LOAD_CODE]

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


def main(*paths):
    df = load(paths or {tuple(_data_files(session))!r})
    df = df.dropna(subset=[{target!r}])
    if df[{target!r}].dtype.kind == "f" and (df[{target!r}] == df[{target!r}].round()).all():
        df[{target!r}] = df[{target!r}].astype("int64")  # 0.0/1.0 back to 0/1 once the blanks are gone
{split_code}
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
{_run_line(session, stem)}
"""

import io
import os
import sys

import numpy as np
import pandas as pd
''' + LOAD_CODE]
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


def main(*paths):
    df = load(paths or {tuple(_data_files(session))!r})
    cleaned = clean(df)
    cleaned.to_csv({stem + "_clean.csv"!r}, index=False)
    print(f"Saved {stem}_clean.csv ({{len(cleaned)}} rows)")
    return cleaned


if __name__ == "__main__":
    main(*sys.argv[1:])
''')
    return "\n".join(parts)
