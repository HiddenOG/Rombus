import asyncio
import hashlib
import logging
import pandas as pd
import numpy as np
import uuid
import os
import json
import time
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Optional, Any, List, TypeVar
from starlette.concurrency import run_in_threadpool
from app.config import settings
from app.recipe import STEP_TYPES, Context, Line, apply_to, run_line
from app.recipe_code import step_code
from app.code_checks import peeking_warnings
from app.storage import Database, FileStore, database_url, file_store

logger = logging.getLogger(__name__)

T = TypeVar("T")


def whole_number_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """One blank in a column of 0s and 1s makes pandas store it as 0.0 and 1.0. Once the blank
    rows are gone, turn it back into whole numbers, so predictions read "1", not "1.0"."""
    y = df[target]
    if pd.api.types.is_float_dtype(y) and len(y) and np.isfinite(y).all() and (y == y.round()).all():
        df = df.assign(**{target: y.astype("int64")})
    return df


# Fewer rows than this: no sealed envelope (its test set would be too small to mean much)
MIN_ROWS_TO_SEAL = 100

class DataSession:
    """Represents a single user data science session."""

    def __init__(self, df: pd.DataFrame, filename: str, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.df = df
        self.original_df = df.copy()
        self.filename = filename
        self.current_stage = "ACQUIRE"
        self.completed_steps: List[str] = []
        self.memory: List[Dict[str, str]] = []
        self.workflow_log: List[Dict[str, Any]] = []
        self.target_column: Optional[str] = None
        self.problem_type: Optional[str] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models: Dict[str, Any] = {}
        # Practice-split scores per algorithm, shown next to the final exam score
        self.model_results: Dict[str, Dict[str, Any]] = {}
        # Final exam: rows locked away when the target is chosen, untouched until evaluation
        self.test_df: Optional[pd.DataFrame] = None
        self.test_size: Optional[float] = None
        # The recipe: training rows as they were at the lock, plus every change since.
        # self.df is always replay(raw_train, recipe).
        self.raw_train: Optional[pd.DataFrame] = None
        self.recipe: List[Line] = []
        # How many times the test rows were (re)locked; many re-locks can mean shopping for a lucky split
        self.lock_count = 0
        # Set by recipe changes: True if trained models had to be cleared because the data changed
        self.models_cleared = False
        # The notebook's cells as the frontend shows them (saved so any browser can reopen them)
        self.cells: List[Dict[str, Any]] = []
        # How many times the locked test rows were used for the final exam (each extra look erodes trust)
        self.exam_count = 0
        # The visitor who created this notebook; only they can see or change it
        self.owner: Optional[str] = None
        # Clean-only mode: no target and no model, so no test rows to protect. Steps work on ALL rows.
        # Choosing a target later locks test rows from the original data and replays the recipe.
        self.clean_only = False
        # Time mode: rows are ordered by this column and the test rows are the LATEST ones, so a model
        # is always tested on a future it hasn't seen (a random split would let it peek between days)
        self.time_column: Optional[str] = None
        # Everything (every row) was looked at before a target was chosen: a later final exam says so,
        # since the test rows were seen while exploring
        self.explored_first = False
        # The sealed envelope ({'rows': [...], 'by': date column or None, 'share': 0.2}): rows set aside
        # at the start of exploring, never shown; the test set if a column is predicted later
        self.sealed: Optional[Dict[str, Any]] = None
        self.created_at = time.time()
        self.updated_at = time.time()

    @property
    def ctx(self) -> Context:
        return Context(target=self.target_column, time_column=self.time_column, all_rows=self.clean_only)

    # ── Recipe changes ──────────────────────────────────────────────────────────
    # Every change goes through _set_recipe: replay on a scratch copy first, and keep the
    # result only if every line still runs. A failed change leaves the session untouched.

    def _set_recipe(self, lines: List[Line]) -> None:
        fresh = [Line(l.tool, dict(l.args), id=l.id) for l in lines]
        df = self.raw_train.copy() if self.raw_train is not None else self.df.copy()
        for i, line in enumerate(fresh):
            try:
                df = run_line(line, df, self.ctx)
            except ValueError as e:
                raise ValueError(
                    f"That change would break a later step: '{line.tool.replace('_', ' ')}' "
                    f"(step {i + 1} after the change) can't run: {e}"
                ) from None
        self.df, self.recipe = df, fresh
        self.models_cleared = self._clear_models()
        self.updated_at = time.time()

    def _clear_models(self) -> bool:
        """Data changed, so any split or model built on the old data no longer matches it."""
        had_models = bool(self.trained_models) or self.X_train is not None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.trained_models = {}
        self.model_results = {}
        return had_models

    def add_step(self, tool: str, args: Dict[str, Any]) -> Line:
        """Run a data-changing step on the training rows and write it into the recipe.

        Raises ValueError (with a user-facing message) if the step can't run; nothing changes then.
        """
        line = Line(tool, dict(args))
        self.df = run_line(line, self.df, self.ctx)
        self.recipe.append(line)
        self.models_cleared = self._clear_models()
        self.updated_at = time.time()
        return line

    def undo(self) -> Optional[Line]:
        """Cross out the last recipe line and rebuild the training rows from scratch."""
        if not self.recipe:
            return None
        line = self.recipe[-1]
        self._set_recipe(self.recipe[:-1])
        return line

    def update_step(self, step_id: str, args: Dict[str, Any]) -> Line:
        """Change one line's settings (keeping its id) and replay everything from the top."""
        index = self.index_of(step_id)
        lines = list(self.recipe)
        lines[index] = Line(lines[index].tool, dict(args), id=step_id)
        self._set_recipe(lines)
        return self.recipe[index]

    def replace_step(self, step_id: str, tool: str, args: Dict[str, Any]) -> Line:
        """Swap a line for a different kind of step in the same place (e.g. built-in -> custom code)."""
        index = self.index_of(step_id)
        lines = list(self.recipe)
        lines[index] = Line(tool, dict(args), id=step_id)
        self._set_recipe(lines)
        return self.recipe[index]

    def move_step(self, step_id: str, direction: int) -> Line:
        """Swap a line with its neighbour (-1 = earlier, +1 = later) and replay."""
        index = self.index_of(step_id)
        target = index + (1 if direction > 0 else -1)
        if not 0 <= target < len(self.recipe):
            raise ValueError("That step is already at the " + ("end." if direction > 0 else "start."))
        lines = list(self.recipe)
        lines[index], lines[target] = lines[target], lines[index]
        self._set_recipe(lines)
        return self.recipe[target]

    def restart(self) -> None:
        """Like restarting a Jupyter kernel: forget everything held in memory, keep the notebook.

        Trained models, the practice split and the AI's conversation memory are cleared, and the
        training rows are rebuilt from the raw rows by replaying the recipe from the top.
        """
        self.memory = []
        if self.raw_train is not None:
            self._set_recipe(self.recipe)
        else:
            self._clear_models()
        self.models_cleared = False

    def delete_step(self, step_id: str) -> Line:
        """Remove any line (not just the last) and replay what's left."""
        index = self.index_of(step_id)
        removed = self.recipe[index]
        self._set_recipe(self.recipe[:index] + self.recipe[index + 1:])
        return removed

    def delete_steps(self, step_ids: List[str]) -> List[Line]:
        """Remove several lines as ONE change: replay once, and refuse the whole change if what's
        left can't run (rather than stopping halfway through a one-by-one undo)."""
        wanted = set(step_ids)
        removed = [l for l in self.recipe if l.id in wanted]
        if not removed:
            raise ValueError("None of those steps are in the recipe any more.")
        self._set_recipe([l for l in self.recipe if l.id not in wanted])
        return removed

    def index_of(self, step_id: str) -> int:
        for i, line in enumerate(self.recipe):
            if line.id == step_id:
                return i
        raise ValueError("That step is no longer in the recipe.")

    def step_view(self, line: Line, index: int) -> Dict[str, Any]:
        return {
            "id": line.id,
            "index": index,
            "tool": line.tool,
            "args": line.args,
            "kind": STEP_TYPES[line.tool].kind,
            "learned": line.learned,
            "message": line.message,
            "code": step_code(line, self.target_column),
            "editable_code": step_code(line, self.target_column, with_learned=False),
            "warnings": peeking_warnings(line.args.get("code", "")) if line.tool == "custom_code" else [],
            "based_on": line.args.get("based_on"),
        }

    def recipe_view(self) -> List[Dict[str, Any]]:
        """The recipe as plain data for the UI: what each step did and what it learned."""
        return [self.step_view(l, i) for i, l in enumerate(self.recipe)]

    @property
    def data_version(self) -> str:
        """Changes whenever the training rows change; lets the UI flag out-of-date charts."""
        key = json.dumps([self.test_size, self.target_column, [(l.id, l.args) for l in self.recipe]],
                         sort_keys=True, default=str)
        return hashlib.sha1(key.encode()).hexdigest()[:10]

    def raw_features(self) -> List[Dict[str, Any]]:
        """The original input columns (before any cleaning), for the prediction form.

        Typical values and options come from the raw TRAINING rows only.
        """
        if self.raw_train is None:
            return []
        # Columns the recipe throws away don't need to be asked for
        removed = {c for l in self.recipe if l.tool == "remove_columns" for c in l.args.get("columns", [])}
        fields = []
        for col in self.raw_train.columns:
            if col == self.target_column or col in removed:
                continue
            series = self.raw_train[col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                fields.append({
                    "name": col,
                    "kind": "number",
                    "typical": round(float(series.median()), 4) if len(series) else None,
                    "min": round(float(series.min()), 4) if len(series) else None,
                    "max": round(float(series.max()), 4) if len(series) else None,
                })
            else:
                counts = series.astype(str).value_counts()
                fields.append({
                    "name": col,
                    "kind": "category",
                    "options": counts.index[:30].tolist(),
                    "typical": counts.index[0] if len(counts) else None,
                })
        return fields

    def prepare(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Apply the recipe's sticky notes to other rows (test set, new customers)."""
        return apply_to(rows, self.recipe, self.ctx)

    @property
    def can_undo(self) -> bool:
        return len(self.recipe) > 0

    @property
    def undo_count(self) -> int:
        return len(self.recipe)

    @property
    def is_split(self) -> bool:
        return self.test_df is not None

    @property
    def can_explore(self) -> bool:
        """Statistics and cleaning are open once test rows are locked away, or in clean-only mode
        (no model will ever be scored on these rows, so there's nothing for them to leak into)."""
        return self.is_split or self.clean_only

    def set_header_row(self, row: int) -> str:
        """Take the column names from another row of the uploaded table (see app/tables.py). Only
        before any work starts: every step, the lock and the export refer to these names."""
        from app.tables import header_from_row
        if self.is_split or self.clean_only or self.recipe:
            raise ValueError("The column names can only be changed before you choose what to do with the data "
                             "(predict a column, or clean only). Upload the file again to start over.")
        table = header_from_row(self.original_df, row)
        self.original_df, self.df = table, table.copy()
        self.updated_at = time.time()
        shown = ", ".join(str(c) for c in table.columns[:6]) + ("…" if table.shape[1] > 6 else "")
        return f"Column names taken from row {row}: {shown}. The table now has {table.shape[0]} rows and {table.shape[1]} columns."

    def start_clean_only(self) -> None:
        """Work on all the rows, without a target or a model (and without a sealed envelope: if a column
        is predicted later, its final exam says every row had been seen)."""
        if self.is_split:
            raise ValueError("This notebook already predicts a column, so its test rows stay locked. "
                             "Start a new notebook from the same file to clean it without a model.")
        self.clean_only = True
        self.explored_first = True
        self.sealed = None
        self.raw_train = self.original_df.copy()
        self._set_recipe(self.recipe)

    def start_exploring(self, share: float, seal_by: Optional[str] = None) -> str:
        """Explore freely, with a share of the rows sealed in an envelope nobody sees: if a column is
        predicted later, the sealed rows become its test set, so the final score is fully honest.
        Sealed at random, or the most recent rows by a date column (so a later time split is honest too).
        Returns a one-line description for the person."""
        from app.expressions import as_dates, looks_like_dates
        if self.is_split:
            raise ValueError("This notebook already predicts a column, so its test rows are already locked.")
        rows = len(self.original_df)
        if rows < MIN_ROWS_TO_SEAL:
            self.start_clean_only()
            return (f"Exploring all {rows} rows: with fewer than {MIN_ROWS_TO_SEAL}, a sealed test set would be "
                    "too small to mean much. To predict a column with an honest score, choose it first.")
        n = max(1, int(np.ceil(rows * share)))
        if seal_by:
            if seal_by not in self.original_df.columns:
                raise ValueError(f"Column '{seal_by}' not found.")
            if not looks_like_dates(self.original_df[seal_by]):
                raise ValueError(f"'{seal_by}' doesn't look like dates, so it can't say which rows are most recent.")
            times = as_dates(self.original_df[seal_by])
            sealed = times[times.notna()].sort_values(kind="stable").index[-n:]
        else:
            sealed = pd.Index(np.random.default_rng(42).choice(self.original_df.index, size=n, replace=False))
        self.sealed = {"rows": sorted(int(i) for i in sealed), "by": seal_by, "share": share}
        self.clean_only = True
        self.explored_first = False
        self.raw_train = self.original_df.drop(index=sealed).reset_index(drop=True)
        self._set_recipe(self.recipe)
        how = f"the most recent by '{seal_by}'" if seal_by else "chosen at random"
        return (f"Exploring {len(self.raw_train):,} of {rows:,} rows. {n:,} rows ({share:.0%}, {how}) are sealed: "
                "nobody sees them, so if you predict a column later they're its test set and its score is honest.")

    def open_envelope(self) -> None:
        """The sealed rows were shown (an export of all rows): from now on they count as seen."""
        if self.sealed and not self.is_split:
            self.sealed = None
            self.explored_first = True
            self.raw_train = self.original_df.copy()
            self._set_recipe(self.recipe)

    def lock_test_rows(self, test_size: float) -> int:
        """Set aside test rows from the original data; self.df becomes the training rows.

        Rows with a blank target are dropped first (they can't be learned from or graded).
        Classification splits are stratified when every class has enough rows.
        An existing recipe is replayed on the new training rows; if it can't run there,
        the previous split is restored and ValueError is raised.
        Returns the number of rows dropped for a blank target.
        """
        previous = (self.raw_train, self.test_df, self.test_size, self.df)
        dropped = self._split(test_size)
        try:
            self._set_recipe(self.recipe)
        except ValueError:
            self.raw_train, self.test_df, self.test_size, self.df = previous
            raise
        self.lock_count += 1
        # From clean-only to modelling: the test rows came from the ORIGINAL data and the recipe was
        # replayed on the training rows only, so the cleaning done on all rows leaks nothing
        self.clean_only = False
        return dropped

    def _split(self, test_size: float) -> int:
        from sklearn.model_selection import train_test_split

        full = whole_number_target(self.original_df.dropna(subset=[self.target_column]), self.target_column)

        if self.sealed:
            # The envelope: the sealed rows are the test set, whatever was asked (and on every re-lock),
            # because they're the only rows nobody has seen
            if self.time_column and self.time_column != self.sealed["by"]:
                how = f"by '{self.sealed['by']}'" if self.sealed["by"] else "at random"
                raise ValueError(f"The sealed rows were chosen {how}, so a time split by '{self.time_column}' would "
                                 "test on rows you've explored. Use a random split, or start a new notebook and "
                                 "choose the time split first.")
            is_test = full.index.isin(self.sealed["rows"])
            train, test = full[~is_test], full[is_test]
            if self.time_column:
                from app.expressions import as_dates
                train = train.loc[as_dates(train[self.time_column]).sort_values(kind="stable").index]
                test = test.loc[as_dates(test[self.time_column]).sort_values(kind="stable").index]
            if len(test) == 0 or len(train) == 0:
                raise ValueError("No rows with an answer in this column are left on one side of the seal.")
            dropped = len(self.original_df) - len(full)
            self.raw_train, self.test_df = train.reset_index(drop=True), test.reset_index(drop=True)
            self.df = self.raw_train.copy()
            self.test_size = self.sealed["share"]
            return dropped

        if self.time_column:
            # Time mode: oldest rows train, the most recent share is locked away as the test set.
            # Rows whose time can't be read can't be placed in time, so they're left out (and counted).
            from app.expressions import as_dates
            times = as_dates(full[self.time_column])
            full = full.loc[times[times.notna()].sort_values(kind="stable").index]
            n_test = max(1, int(np.ceil(len(full) * test_size)))
            if n_test >= len(full):
                raise ValueError("Too few rows with a readable date to set some aside.")
            train, test = full.iloc[:-n_test], full.iloc[-n_test:]
        else:
            y = full[self.target_column]
            stratify = None
            if self.problem_type == "classification":
                counts = y.value_counts()
                n_test = int(np.ceil(len(full) * test_size))
                if counts.min() >= 2 and n_test >= len(counts):
                    stratify = y
            train, test = train_test_split(full, test_size=test_size, random_state=42, stratify=stratify)
        dropped = len(self.original_df) - len(full)
        self.raw_train = train.reset_index(drop=True)
        self.df = self.raw_train.copy()
        self.test_df = test.reset_index(drop=True)
        self.test_size = test_size
        return dropped

    def compute_column_evidence(self, col: str) -> Dict[str, Any]:
        """Compute statistical evidence for a single column.

        Before the test rows are locked away, only the name and type are revealed:
        statistics would let test rows influence decisions. (Clean-only mode has no test rows.)
        """
        if col not in self.df.columns:
            return {}
        series = self.df[col]
        evidence = {"name": col, "dtype": str(series.dtype)}
        if not self.can_explore:
            return evidence
        evidence |= {
            "missing_ratio": float(series.isnull().mean()),
            "unique_count": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            clean = series.dropna().astype(float)
            clean = clean[np.isfinite(clean)]
            if not clean.empty:
                from scipy import stats as sp_stats
                spread = float(clean.std(ddof=0))
                # A column holding one value has no shape: skew and outliers are 0, not NaN
                # (NaN can't be sent as JSON and would crash the reply)
                varied = spread > 0 and len(clean) > 2
                evidence.update({
                    "mean": round(float(clean.mean()), 4),
                    "std": round(float(clean.std()) if len(clean) > 1 else 0.0, 4),
                    "skewness": round(float(sp_stats.skew(clean)), 4) if varied else 0.0,
                    "outlier_ratio": round(float((np.abs((clean - clean.mean()) / spread) > 3).mean()), 4) if varied else 0.0,
                })
        return evidence

    def get_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive session summary."""
        df = self.df
        columns = [self.compute_column_evidence(col) for col in df.columns]
        return {
            "stage": self.current_stage,
            "filename": self.filename,
            "shape": list(df.shape),
            "target": self.target_column,
            "problem_type": self.problem_type,
            "has_split": self.X_train is not None,
            "is_split": self.is_split,
            "clean_only": self.clean_only,
            "time_column": self.time_column,
            "test_rows": len(self.test_df) if self.is_split else 0,
            "test_size": self.test_size,
            "lock_count": self.lock_count,
            "recipe": self.recipe_view(),
            "data_version": self.data_version,
            "raw_features": self.raw_features(),
            "completed_steps": self.completed_steps,
            "can_undo": self.can_undo,
            "undo_count": self.undo_count,
            "trained_models": list(self.trained_models.keys()),
            "columns": columns,
            # The envelope while exploring: how many rows are sealed (hidden), and how they were chosen
            "sealed_rows": len(self.sealed["rows"]) if self.sealed and not self.is_split else 0,
            "sealed_share": self.sealed["share"] if self.sealed else None,
            "sealed_by": self.sealed["by"] if self.sealed else None,
            "explored_first": self.explored_first,
            # Before anything is chosen: date columns that could order a seal of the most recent rows
            "date_columns": self.date_columns() if not self.can_explore else [],
        }

    def date_columns(self) -> List[str]:
        from app.expressions import looks_like_dates
        sample = self.original_df.head(500)
        return [str(c) for c in sample.columns
                if pd.api.types.is_datetime64_any_dtype(sample[c]) or (sample[c].dtype == object or
                    pd.api.types.is_string_dtype(sample[c])) and looks_like_dates(sample[c])]

    def get_compact_column_summary(self) -> str:
        """Get a compact text summary of all columns for Gemini prompts."""
        lines = []
        for col in self.df.columns:
            ev = self.compute_column_evidence(col)
            parts = [f"{ev['name']} ({ev['dtype']})"]
            if "missing_ratio" in ev:
                parts.append(f"miss={ev['missing_ratio']:.1%}")
                parts.append(f"uniq={ev['unique_count']}")
            if "mean" in ev:
                parts.append(f"mean={ev['mean']}")
            if "skewness" in ev:
                parts.append(f"skew={ev['skewness']}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)


class SessionManager:
    """Keeps sessions in memory and saves them through the storage layer (app/storage.py)."""

    def __init__(self):
        # Most recently used last. Only MAX_SESSIONS_IN_MEMORY stay loaded; the rest wait in storage.
        self._sessions: "OrderedDict[str, DataSession]" = OrderedDict()
        self._db: Optional[Database] = None
        self._files: Optional[FileStore] = None
        # The table last saved under each file key. The data tables are only ever replaced, never
        # changed in place, so "same object as last time" means "unchanged": skip the upload.
        # (Cells autosave constantly; re-uploading a dataset each time would be slow over the internet.)
        self._saved_files: Dict[str, pd.DataFrame] = {}

    def initialize(self):
        """Open the database (SQLite by default, PostgreSQL via DATABASE_URL) and the file store
        (this machine by default, Supabase Storage with SUPABASE_SECRET_KEY)."""
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        self._db = Database(database_url())
        self._files = file_store()
        self._saved_files.clear()

    def create_session(self, df: pd.DataFrame, filename: str, owner: Optional[str] = None) -> DataSession:
        """Create a new session and persist it."""
        session = DataSession(df, filename)
        session.owner = owner
        self._persist_session(session)
        self._remember(session)
        return session

    def get_session(self, session_id: str) -> Optional[DataSession]:
        """Retrieve a session from memory, or rebuild it from storage."""
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        session = self._load_session(session_id)
        if session:
            self._remember(session)
        return session

    def _remember(self, session: DataSession) -> None:
        """Keep a notebook in memory, unloading the least recently used ones past the limit.
        Unloaded notebooks are safe in storage and reload when opened (trained models, which
        live only in memory, need training again). Never unloads one that's busy."""
        self._sessions[session.session_id] = session
        self._sessions.move_to_end(session.session_id)
        for sid in list(self._sessions):
            if len(self._sessions) <= settings.MAX_SESSIONS_IN_MEMORY:
                break
            if sid != session.session_id and _idle(sid):
                del self._sessions[sid]
                _notebook_locks.pop(sid, None)  # the lock table doesn't grow with every notebook ever opened
                self._saved_files = {k: v for k, v in self._saved_files.items() if not k.startswith(sid)}

    def _save_or_delete(self, key: str, df: Optional[pd.DataFrame]) -> None:
        if df is None:
            already_gone = key in self._saved_files and self._saved_files[key] is None
            if not already_gone:
                self._files.delete(key)
                self._saved_files[key] = None
        elif self._saved_files.get(key) is not df:
            self._files.save(key, df)
            self._saved_files[key] = df

    def _persist_session(self, session: DataSession):
        """Save what's needed to rebuild the session: original rows, the lock, the recipe, the cells.

        The cleaned training rows aren't stored: they're rebuilt on load by replaying the
        recipe on the raw training rows, which also restores every sticky note.
        """
        if not self._db:
            return

        session.updated_at = time.time()
        sid = session.session_id
        self._save_or_delete(f"{sid}_original", session.original_df)
        # Clean-only: the rows are the original upload, already saved above
        self._save_or_delete(f"{sid}_train", None if session.clean_only else session.raw_train)
        self._save_or_delete(f"{sid}_test", session.test_df)

        self._db.save({
            "session_id": sid,
            "filename": session.filename,
            "current_stage": session.current_stage,
            "target_column": session.target_column,
            "problem_type": session.problem_type,
            "completed_steps": json.dumps(session.completed_steps),
            "memory": json.dumps(session.memory[-40:]),  # the last 40 messages (20 exchanges)
            "workflow_log": json.dumps(session.workflow_log[-50:]),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "test_size": session.test_size,
            "recipe": json.dumps([line.to_json() for line in session.recipe]),
            "lock_count": session.lock_count,
            "cells": json.dumps(session.cells, default=str),
            "exam_count": session.exam_count,
            "owner": session.owner,
            "clean_only": int(session.clean_only),
            "time_column": session.time_column,
            "explored_first": int(session.explored_first),
            "sealed": json.dumps(session.sealed) if session.sealed else None,
        })

    def _load_session(self, session_id: str) -> Optional[DataSession]:
        """Rebuild a session from its record and data files."""
        if not self._db:
            return None
        try:
            row = self._db.get(session_id)
            original = self._files.load(f"{session_id}_original") if row else None
            if row is None or original is None:
                return None

            session = DataSession(original, row["filename"], session_id=session_id)
            session.current_stage = row["current_stage"]
            session.target_column = row["target_column"]
            session.problem_type = row["problem_type"]
            session.completed_steps = json.loads(row["completed_steps"] or "[]")
            session.memory = json.loads(row["memory"] or "[]")
            session.workflow_log = json.loads(row["workflow_log"] or "[]")
            session.created_at = row["created_at"]
            session.updated_at = row["updated_at"]
            session.lock_count = row["lock_count"] or 0
            session.cells = json.loads(row["cells"] or "[]")
            session.exam_count = row["exam_count"] or 0
            session.owner = row["owner"]
            session.time_column = row.get("time_column")
            session.explored_first = bool(row.get("explored_first"))
            session.sealed = json.loads(row["sealed"]) if row.get("sealed") else None
            lines = [Line(s["tool"], s["args"], **({"id": s["id"]} if "id" in s else {}))
                     for s in json.loads(row["recipe"] or "[]")]

            # Rebuild the training rows: raw training rows + replayed recipe (restores sticky notes too)
            train, test = self._files.load(f"{session_id}_train"), self._files.load(f"{session_id}_test")
            if train is not None and test is not None:
                session.raw_train, session.test_df = train, test
                session.test_size = row["test_size"]
                session._set_recipe(lines)
                session.models_cleared = False
            elif row.get("clean_only"):
                # Exploring: the rows are the original upload, minus the sealed ones if there's an envelope
                session.recipe = lines
                if session.sealed:
                    session.clean_only = True
                    session.raw_train = session.original_df.drop(index=session.sealed["rows"]).reset_index(drop=True)
                    session._set_recipe(lines)
                else:
                    session.start_clean_only()
                session.models_cleared = False

            # What's in storage now matches memory: nothing to upload until it changes
            self._saved_files.update({f"{session_id}_original": session.original_df,
                                      f"{session_id}_train": session.raw_train,
                                      f"{session_id}_test": session.test_df})
            return session
        except Exception:
            logger.exception("Could not load session %s", session_id)
            return None

    def save(self, session: DataSession):
        """Explicitly save a session."""
        self._persist_session(session)

    def list_sessions(self, limit: int = 200, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """Most recently used sessions of `owner`, for the notebook list."""
        if not self._db:
            return []
        return [
            {"session_id": r["session_id"], "filename": r["filename"], "target": r["target_column"],
             "stage": r["current_stage"], "updated_at": r["updated_at"]}
            for r in self._db.recent(limit, owner)
        ]

    def cleanup_expired(self) -> int:
        """Delete anonymous notebooks (memory, files and records) unused for longer than SESSION_TTL_HOURS.
        Notebooks in an account are kept: that's the point of signing in."""
        if settings.SESSION_TTL_HOURS <= 0:
            return 0
        cutoff = time.time() - (settings.SESSION_TTL_HOURS * 3600)
        expired = set(sid for sid, s in self._sessions.items()
                      if s.updated_at < cutoff and not (s.owner or "").startswith("user:"))
        if self._db:
            expired |= set(self._db.older_than(cutoff))
        for sid in expired:
            self.delete_session(sid)
        return len(expired)

    def transfer_owner(self, old: str, new: str) -> int:
        """Give all of `old`'s notebooks to `new` (used when a visitor signs in)."""
        for session in self._sessions.values():
            if session.owner == old:
                session.owner = new
        return self._db.transfer_owner(old, new) if self._db else 0

    def delete_session(self, session_id: str):
        """Remove a notebook everywhere: memory, data files and its database record."""
        self._sessions.pop(session_id, None)
        if _idle(session_id):
            _notebook_locks.pop(session_id, None)
        if self._files:
            for suffix in ("", "_original", "_train", "_test"):
                self._files.delete(f"{session_id}{suffix}")
                self._saved_files.pop(f"{session_id}{suffix}", None)
        if self._db:
            self._db.delete(session_id)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# Global session manager instance
session_manager = SessionManager()


_notebook_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _idle(session_id: str) -> bool:
    """No request is working on this notebook or waiting to (safe to unload, or to forget its lock)."""
    lock = _notebook_locks.get(session_id)  # .get: looking mustn't create an entry
    return lock is None or (not lock.locked() and not getattr(lock, "_waiters", None))


async def run_exclusive(session_id: str, fn: Callable[..., T], *args: Any) -> T:
    """Run blocking work for one notebook (training, replaying the recipe, saving to the cloud)
    on a worker thread, one request at a time per notebook.

    Worker thread: the server keeps answering everyone else while this notebook works.
    One at a time: two requests (say, training and a cells autosave) never change the same
    notebook at once.
    """
    async with _notebook_locks[session_id]:
        return await run_in_threadpool(fn, *args)
