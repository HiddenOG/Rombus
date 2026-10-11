"""Where notebooks are kept: a database for the notebook records, a file store for the data.

Both are swappable:
  Database   SQLAlchemy, so DATABASE_URL can point at SQLite (default, local and private)
             or PostgreSQL (e.g. Supabase) without code changes.
  FileStore  the data tables (original upload, locked train/test rows) as Parquet files.
             LocalFileStore keeps them on disk; SupabaseFileStore keeps them in a private
             Supabase Storage bucket (used when SUPABASE_SECRET_KEY is set).
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import quote, unquote

import httpx
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


logger = logging.getLogger(__name__)


def database_url() -> str:
    """DATABASE_URL if set, otherwise a SQLite file inside DATA_DIR.
    Supabase hands out plain "postgresql://" URLs; we talk to them with the psycopg driver."""
    url = settings.DATABASE_URL.strip().strip("\"'")
    if url.startswith(("postgresql://", "postgres://")):
        url = "postgresql+psycopg://" + _encode_password(url.split("://", 1)[1])
    return url or f"sqlite:///{os.path.join(settings.DATA_DIR, 'sessions.db')}"


def _encode_password(rest: str) -> str:
    """In "user:password@host/db", make the password URL-safe. Characters like # / @ ? would
    otherwise be read as parts of the address (a # starts a "bookmark", cutting the rest off).
    Already-encoded passwords (%23 etc.) are left as they are."""
    credentials, at, host = rest.rpartition("@")
    if not at or ":" not in credentials:
        return rest
    user, _, password = credentials.partition(":")
    return f"{user}:{quote(unquote(password), safe='')}@{host}"


class Database:
    """The notebook records table."""

    def __init__(self, url: str):
        # pre_ping: a cloud database drops idle connections; check before use instead of failing
        self.engine = create_engine(url, future=True, pool_pre_ping=True)
        metadata.create_all(self.engine)
        self._add_missing_columns()
        self._close_public_door()

    def _close_public_door(self) -> None:
        """On Supabase, tables are reachable through its public web API with the publishable key
        that every visitor's browser has. Row Level Security with no rules shuts that door; this
        server connects as the table's owner, so it still gets in."""
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as conn:
                conn.execute(text("ALTER TABLE sessions ENABLE ROW LEVEL SECURITY"))

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
        """Most recent notebooks of `owner` (all notebooks when owner is None, for maintenance scripts).
        Unowned notebooks from before owners existed are never listed to visitors: on a shared
        server that would show them to everyone."""
        t = sessions_table
        query = (select(t.c.session_id, t.c.filename, t.c.target_column, t.c.current_stage, t.c.updated_at)
                 .order_by(t.c.updated_at.desc()).limit(limit))
        if owner is not None:
            query = query.where(t.c.owner == owner)
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


class SupabaseFileStore:
    """Data tables as Parquet files in a private Supabase Storage bucket.

    Uses the secret key, which can read and write everything: it lives only on this server,
    never in the browser. The bucket is private, so files are reachable only through this server.
    """

    def __init__(self, project_url: str, secret_key: str, bucket: str, transport: Optional[httpx.BaseTransport] = None):
        self.bucket = bucket
        # The new sb_secret_ keys go in both headers (they aren't JWTs; the gateway accepts them this way)
        self.client = httpx.Client(
            base_url=f"{project_url.rstrip('/')}/storage/v1",
            headers={"apikey": secret_key, "Authorization": f"Bearer {secret_key}"},
            timeout=60,
            transport=transport,  # tests pass a fake Supabase here
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if self.client.get(f"/bucket/{self.bucket}").status_code == 200:
            return
        r = self.client.post("/bucket", json={"id": self.bucket, "name": self.bucket, "public": False})
        if r.status_code >= 400 and "already exists" not in r.text.lower():
            raise RuntimeError(f"Couldn't create the Supabase Storage bucket '{self.bucket}': {r.status_code} {r.text[:200]}")

    def _path(self, key: str) -> str:
        return f"/object/{self.bucket}/{key}.parquet"

    def save(self, key: str, df: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        r = self.client.post(self._path(key), content=buffer.getvalue(),
                             headers={"Content-Type": "application/octet-stream", "x-upsert": "true"})
        r.raise_for_status()

    def load(self, key: str) -> Optional[pd.DataFrame]:
        r = self.client.get(self._path(key))
        if r.status_code == 200:
            return pd.read_parquet(io.BytesIO(r.content))
        if r.status_code in (400, 404) and "not" in r.text.lower():  # Storage reports "not found" either way
            return None
        r.raise_for_status()
        return None

    def delete(self, key: str) -> None:
        r = self.client.request("DELETE", f"/object/{self.bucket}", json={"prefixes": [f"{key}.parquet"]})
        if r.status_code >= 400:
            logger.warning("Couldn't delete %s from Supabase Storage: %s", key, r.status_code)


def file_store() -> FileStore:
    """Supabase Storage when a secret key is configured, otherwise files on this machine."""
    if settings.SUPABASE_URL and settings.SUPABASE_SECRET_KEY:
        return SupabaseFileStore(settings.SUPABASE_URL, settings.SUPABASE_SECRET_KEY, settings.STORAGE_BUCKET)
    return LocalFileStore(settings.DATA_DIR)
