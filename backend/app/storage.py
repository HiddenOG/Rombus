"""Where notebooks are kept: a database for the notebook records, a file store for the data.

Both are swappable:
  Database   SQLAlchemy, so DATABASE_URL can point at SQLite (default, local and private)
             or PostgreSQL (e.g. Supabase) without code changes.
  FileStore  the data tables (original upload, locked train/test rows) as Parquet files.
             LocalFileStore keeps them on disk; a cloud store (e.g. Supabase Storage) only
             needs the same three methods.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, create_engine, delete, inspect, insert, select, text, update

from app.config import settings

metadata = MetaData()

sessions_table = Table(
    "sessions",
    metadata,
    Column("session_id", String(64), primary_key=True),
    Column("filename", Text, nullable=False),
    Column("current_stage", String(32), nullable=False),
    Column("target_column", Text),
    Column("problem_type", String(32)),
    Column("completed_steps", Text),
    Column("memory", Text),
    Column("workflow_log", Text),
    Column("created_at", Float),
    Column("updated_at", Float),
    Column("test_size", Float),
    Column("recipe", Text),
    Column("lock_count", Integer),
    Column("cells", Text),
    Column("exam_count", Integer),
    Column("owner", String(64)),
)


def database_url() -> str:
    """DATABASE_URL if set, otherwise a SQLite file inside DATA_DIR."""
    return settings.DATABASE_URL or f"sqlite:///{os.path.join(settings.DATA_DIR, 'sessions.db')}"


class Database:
    """The notebook records table."""

    def __init__(self, url: str):
        self.engine = create_engine(url, future=True)
        metadata.create_all(self.engine)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Databases created by older versions lack newer columns: add them."""
        existing = {c["name"] for c in inspect(self.engine).get_columns("sessions")}
        with self.engine.begin() as conn:
            for column in sessions_table.columns:
                if column.name not in existing:
                    sql_type = column.type.compile(dialect=self.engine.dialect)
                    conn.execute(text(f"ALTER TABLE sessions ADD COLUMN {column.name} {sql_type}"))

    def save(self, row: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:  # one transaction: replace the old record
            conn.execute(delete(sessions_table).where(sessions_table.c.session_id == row["session_id"]))
            conn.execute(insert(sessions_table).values(**row))

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(select(sessions_table).where(sessions_table.c.session_id == session_id)).mappings().first()
        return dict(row) if row else None

    def recent(self, limit: int, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """Most recent notebooks of `owner`, plus unowned ones from before owners existed."""
        t = sessions_table
        query = (select(t.c.session_id, t.c.filename, t.c.target_column, t.c.current_stage, t.c.updated_at)
                 .order_by(t.c.updated_at.desc()).limit(limit))
        if owner is not None:
            query = query.where((t.c.owner == owner) | (t.c.owner.is_(None)))
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(query).mappings()]

    def older_than(self, cutoff: float) -> List[str]:
        """Anonymous notebooks unused since `cutoff`. Notebooks in an account are never expired."""
        t = sessions_table
        query = select(t.c.session_id).where(
            (t.c.updated_at < cutoff) & (t.c.owner.is_(None) | ~t.c.owner.startswith("user:")))
        with self.engine.connect() as conn:
            return [r[0] for r in conn.execute(query)]

    def transfer_owner(self, old: str, new: str) -> int:
        """Give all of `old`'s notebooks to `new`; returns how many moved."""
        with self.engine.begin() as conn:
            return conn.execute(
                update(sessions_table).where(sessions_table.c.owner == old).values(owner=new)).rowcount

    def delete(self, session_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(sessions_table).where(sessions_table.c.session_id == session_id))


class FileStore(Protocol):
    def save(self, key: str, df: pd.DataFrame) -> None: ...
    def load(self, key: str) -> Optional[pd.DataFrame]: ...
    def delete(self, key: str) -> None: ...


class LocalFileStore:
    """Data tables as Parquet files in a folder on this machine."""

    def __init__(self, folder: str):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.folder, f"{key}.parquet")

    def save(self, key: str, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        with open(self._path(key), "wb") as f:
            f.write(buffer.getvalue())

    def load(self, key: str) -> Optional[pd.DataFrame]:
        path = self._path(key)
        return pd.read_parquet(path) if os.path.exists(path) else None

    def delete(self, key: str) -> None:
        if os.path.exists(self._path(key)):
            os.remove(self._path(key))
