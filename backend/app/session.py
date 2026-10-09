import logging
import pandas as pd
import numpy as np
import uuid
import os
import json
import sqlite3
import time
from typing import Dict, Optional, Any, List
from app.config import settings
from app.recipe import STEP_TYPES, Context, Line, apply_to, run_line

logger = logging.getLogger(__name__)


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
        self.created_at = time.time()
        self.updated_at = time.time()

    @property
    def ctx(self) -> Context:
        return Context(target=self.target_column)

    # ── Recipe changes ──────────────────────────────────────────────────────────
    # Every change goes through _set_recipe: replay on a scratch copy first, and keep the
    # result only if every line still runs. A failed change leaves the session untouched.

    def _set_recipe(self, lines: List[Line]) -> None:
        fresh = [Line(l.tool, dict(l.args)) for l in lines]
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

    def update_step(self, index: int, args: Dict[str, Any]) -> Line:
        """Change one line's settings and replay everything from the top."""
        self._check_index(index)
        lines = list(self.recipe)
        lines[index] = Line(lines[index].tool, dict(args))
        self._set_recipe(lines)
        return self.recipe[index]

    def delete_step(self, index: int) -> Line:
        """Remove any line (not just the last) and replay what's left."""
        self._check_index(index)
        removed = self.recipe[index]
        self._set_recipe(self.recipe[:index] + self.recipe[index + 1:])
        return removed

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self.recipe):
            raise ValueError(f"There is no step {index + 1}; the recipe has {len(self.recipe)} steps.")

    def recipe_view(self) -> List[Dict[str, Any]]:
        """The recipe as plain data for the UI: what each step did and what it learned."""
        return [
            {
                "index": i,
                "tool": l.tool,
                "args": l.args,
                "kind": STEP_TYPES[l.tool].kind,
                "learned": l.learned,
                "message": l.message,
            }
            for i, l in enumerate(self.recipe)
        ]

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
        return dropped

    def _split(self, test_size: float) -> int:
        from sklearn.model_selection import train_test_split

        full = self.original_df.dropna(subset=[self.target_column])
        dropped = len(self.original_df) - len(full)
        y = full[self.target_column]

        stratify = None
        if self.problem_type == "classification":
            counts = y.value_counts()
            n_test = int(np.ceil(len(full) * test_size))
            if counts.min() >= 2 and n_test >= len(counts):
                stratify = y

        train, test = train_test_split(full, test_size=test_size, random_state=42, stratify=stratify)
        self.raw_train = train.reset_index(drop=True)
        self.df = self.raw_train.copy()
        self.test_df = test.reset_index(drop=True)
        self.test_size = test_size
        return dropped

    def compute_column_evidence(self, col: str) -> Dict[str, Any]:
        """Compute statistical evidence for a single column.

        Before the test rows are locked away, only the name and type are revealed:
        statistics would let test rows influence decisions.
        """
        if col not in self.df.columns:
            return {}
        series = self.df[col]
        evidence = {"name": col, "dtype": str(series.dtype)}
        if not self.is_split:
            return evidence
        evidence |= {
            "missing_ratio": float(series.isnull().mean()),
            "unique_count": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if not clean.empty:
                from scipy import stats as sp_stats
                evidence.update({
                    "mean": round(float(clean.mean()), 4),
                    "std": round(float(clean.std()), 4),
                    "skewness": round(float(sp_stats.skew(clean)), 4),
                    "outlier_ratio": round(
                        float(len(clean[np.abs(sp_stats.zscore(clean)) > 3]) / len(clean))
                        if len(clean) > 0 else 0, 4
                    ),
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
            "test_rows": len(self.test_df) if self.is_split else 0,
            "test_size": self.test_size,
            "lock_count": self.lock_count,
            "recipe": self.recipe_view(),
            "completed_steps": self.completed_steps,
            "can_undo": self.can_undo,
            "undo_count": self.undo_count,
            "trained_models": list(self.trained_models.keys()),
            "columns": columns,
        }

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
    """Manages data sessions with SQLite persistence."""

    def __init__(self):
        self._sessions: Dict[str, DataSession] = {}
        self._db_path: Optional[str] = None

    def initialize(self):
        """Initialize the data directory and SQLite database."""
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        self._db_path = os.path.join(settings.DATA_DIR, "sessions.db")
        self._init_db()

    def _init_db(self):
        """Create the sessions metadata table."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    target_column TEXT,
                    problem_type TEXT,
                    completed_steps TEXT,
                    memory TEXT,
                    workflow_log TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            # Columns added after the first release
            existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            for column, sql_type in [("test_size", "REAL"), ("recipe", "TEXT"), ("lock_count", "INTEGER")]:
                if column not in existing:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} {sql_type}")

    def create_session(self, df: pd.DataFrame, filename: str) -> DataSession:
        """Create a new session and persist it."""
        session = DataSession(df, filename)
        self._sessions[session.session_id] = session
        self._persist_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[DataSession]:
        """Retrieve a session from memory or disk."""
        # Try memory first
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try loading from disk
        session = self._load_session(session_id)
        if session:
            self._sessions[session_id] = session
        return session

    def _path(self, session_id: str, suffix: str) -> str:
        return os.path.join(settings.DATA_DIR, f"{session_id}{suffix}.parquet")

    def _write_or_remove(self, df: Optional[pd.DataFrame], path: str) -> None:
        if df is not None:
            df.to_parquet(path, index=False)
        elif os.path.exists(path):
            os.remove(path)

    def _persist_session(self, session: DataSession):
        """Save the data needed to rebuild the session: original rows, the lock, and the recipe.

        The cleaned training rows aren't stored: they're rebuilt on load by replaying the
        recipe on the raw training rows, which also restores every sticky note.
        """
        if not self._db_path:
            return

        session.updated_at = time.time()
        session.original_df.to_parquet(self._path(session.session_id, "_original"), index=False)
        self._write_or_remove(session.raw_train, self._path(session.session_id, "_train"))
        self._write_or_remove(session.test_df, self._path(session.session_id, "_test"))

        # Save metadata to SQLite
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, filename, current_stage, target_column, problem_type,
                 completed_steps, memory, workflow_log, created_at, updated_at, test_size,
                 recipe, lock_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.filename,
                session.current_stage,
                session.target_column,
                session.problem_type,
                json.dumps(session.completed_steps),
                json.dumps(session.memory[-20:]),  # Keep last 20 messages
                json.dumps(session.workflow_log[-50:]),
                session.created_at,
                session.updated_at,
                session.test_size,
                json.dumps([line.to_json() for line in session.recipe]),
                session.lock_count,
            ))

    def _load_session(self, session_id: str) -> Optional[DataSession]:
        """Load a session from SQLite + Parquet."""
        if not self._db_path:
            return None

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,)
                ).fetchone()

            if not row:
                return None

            original_path = self._path(session_id, "_original")
            if not os.path.exists(original_path):
                return None

            session = DataSession(pd.read_parquet(original_path), row["filename"], session_id=session_id)
            session.current_stage = row["current_stage"]
            session.target_column = row["target_column"]
            session.problem_type = row["problem_type"]
            session.completed_steps = json.loads(row["completed_steps"] or "[]")
            session.memory = json.loads(row["memory"] or "[]")
            session.workflow_log = json.loads(row["workflow_log"] or "[]")
            session.created_at = row["created_at"]
            session.updated_at = row["updated_at"]
            session.lock_count = row["lock_count"] or 0

            # Rebuild the training rows: raw training rows + replayed recipe (restores sticky notes too)
            train_path, test_path = self._path(session_id, "_train"), self._path(session_id, "_test")
            if os.path.exists(train_path) and os.path.exists(test_path):
                session.raw_train = pd.read_parquet(train_path)
                session.test_df = pd.read_parquet(test_path)
                session.test_size = row["test_size"]
                lines = [Line(s["tool"], s["args"]) for s in json.loads(row["recipe"] or "[]")]
                session._set_recipe(lines)
                session.models_cleared = False

            return session
        except Exception:
            logger.exception("Could not load session %s", session_id)
            return None

    def save(self, session: DataSession):
        """Explicitly save a session."""
        self._persist_session(session)

    def cleanup_expired(self):
        """Remove sessions older than TTL."""
        cutoff = time.time() - (settings.SESSION_TTL_HOURS * 3600)
        expired = [
            sid for sid, s in self._sessions.items()
            if s.updated_at < cutoff
        ]
        for sid in expired:
            self._delete_session(sid)

    def _delete_session(self, session_id: str):
        """Remove a session from memory and disk."""
        self._sessions.pop(session_id, None)

        # Remove parquet files
        for suffix in ["", "_original", "_train", "_test"]:
            path = os.path.join(settings.DATA_DIR, f"{session_id}{suffix}.parquet")
            if os.path.exists(path):
                os.remove(path)

        # Remove from SQLite
        if self._db_path:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# Global session manager instance
session_manager = SessionManager()
