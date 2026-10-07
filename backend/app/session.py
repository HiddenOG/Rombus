import pandas as pd
import numpy as np
import uuid
import os
import json
import sqlite3
import time
from typing import Dict, Optional, Any, List
from app.config import settings


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
        self.scaler = None
        self.label_encoders: Dict[str, Any] = {}
        self.created_at = time.time()
        self.updated_at = time.time()
        # Undo/rollback: snapshot stack of DataFrame states
        self._snapshots: List[pd.DataFrame] = []
        self._max_snapshots = 20

    def save_snapshot(self):
        """Save current DataFrame state for undo."""
        self._snapshots.append(self.df.copy())
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def undo(self) -> bool:
        """Revert to the previous DataFrame state."""
        if not self._snapshots:
            return False
        self.df = self._snapshots.pop()
        self.updated_at = time.time()
        return True

    @property
    def can_undo(self) -> bool:
        return len(self._snapshots) > 0

    @property
    def undo_count(self) -> int:
        return len(self._snapshots)

    def compute_column_evidence(self, col: str) -> Dict[str, Any]:
        """Compute statistical evidence for a single column."""
        if col not in self.df.columns:
            return {}
        series = self.df[col]
        evidence = {
            "name": col,
            "dtype": str(series.dtype),
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

    def _persist_session(self, session: DataSession):
        """Save session metadata to SQLite and DataFrame to Parquet."""
        if not self._db_path:
            return

        session.updated_at = time.time()

        # Save DataFrame as parquet
        parquet_path = os.path.join(settings.DATA_DIR, f"{session.session_id}.parquet")
        session.df.to_parquet(parquet_path, index=False)

        # Save original DF
        original_path = os.path.join(settings.DATA_DIR, f"{session.session_id}_original.parquet")
        session.original_df.to_parquet(original_path, index=False)

        # Save metadata to SQLite
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, filename, current_stage, target_column, problem_type,
                 completed_steps, memory, workflow_log, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

            # Load DataFrame
            parquet_path = os.path.join(settings.DATA_DIR, f"{session_id}.parquet")
            if not os.path.exists(parquet_path):
                return None

            df = pd.read_parquet(parquet_path)
            session = DataSession(df, row["filename"], session_id=session_id)
            session.current_stage = row["current_stage"]
            session.target_column = row["target_column"]
            session.problem_type = row["problem_type"]
            session.completed_steps = json.loads(row["completed_steps"] or "[]")
            session.memory = json.loads(row["memory"] or "[]")
            session.workflow_log = json.loads(row["workflow_log"] or "[]")
            session.created_at = row["created_at"]
            session.updated_at = row["updated_at"]

            # Load original DF if available
            original_path = os.path.join(settings.DATA_DIR, f"{session_id}_original.parquet")
            if os.path.exists(original_path):
                session.original_df = pd.read_parquet(original_path)

            return session
        except Exception:
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
        for suffix in ["", "_original"]:
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
