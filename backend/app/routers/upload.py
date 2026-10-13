"""Upload router — file upload endpoint.

One file, or several stacked into one table (e.g. a CSV per stock ticker): rows of every file one
after the other, with a `source_file` column saying where each row came from. Columns missing from
a file are blank for its rows.
"""

import io
import os
from typing import List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.access import current_owner, limit
from app.config import settings
from app.session import session_manager

router = APIRouter()

ALLOWED = (".csv", ".xlsx", ".xls")
MAX_FILES = 20
SOURCE_COLUMN = "source_file"


@router.post("/upload", dependencies=[Depends(limit("upload"))])
async def upload_file(file: Optional[UploadFile] = File(None), files: Optional[List[UploadFile]] = File(None),
                      visitor: str = Depends(current_owner)):
    """Upload one CSV/Excel file, or several to stack into one table, as a new notebook."""
    uploads = [f for f in ([file] if file else []) + (files or []) if f is not None]
    if not uploads:
        raise HTTPException(status_code=400, detail="Choose a file to upload.")
    if len(uploads) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_FILES} files at once.")

    read: List[Tuple[str, bytes]] = []
    total = 0
    for upload in uploads:
        name = upload.filename or "uploaded_file.csv"
        if not name.lower().endswith(ALLOWED):
            raise HTTPException(status_code=400, detail=f"'{name}': please upload .csv, .xlsx or .xls files.")
        contents = await upload.read()
        total += len(contents)
        if total > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Files too large. Maximum in total: {settings.MAX_UPLOAD_SIZE_MB} MB.")
        read.append((name, contents))
    # Reading the tables and saving them to storage are slow: do them on a worker thread
    return await run_in_threadpool(_create_notebook, read, visitor)


SEPARATORS = (";", "\t", "|")


def _read_csv(contents: bytes) -> pd.DataFrame:
    """A CSV as people actually save them: UTF-8 (with or without BOM) or Windows/Excel encoding,
    separated by commas, or semicolons (European Excel), tabs or pipes."""
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = contents.decode("cp1252", errors="replace")  # Excel on Windows: £, é, ü...
    if "\x00" in text[:10000]:
        raise ValueError("it isn't a text file (a CSV is text)")
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] == 1:  # one column: maybe it isn't separated by commas
        header = text.split("\n", 1)[0]
        best = max(SEPARATORS, key=header.count)
        if header.count(best):
            df = pd.read_csv(io.StringIO(text), sep=best)
    return df


def _read_table(name: str, contents: bytes) -> pd.DataFrame:
    try:
        if name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = _read_csv(contents)
    except Exception as e:
        reason = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        raise HTTPException(status_code=400, detail=f"Couldn't read '{name}' as a table: {reason}")
    df.columns = [str(c) for c in df.columns]  # Excel headers like 2024 come in as numbers
    return df


def _create_notebook(read: List[Tuple[str, bytes]], visitor: str):
    tables = [(name, _read_table(name, contents)) for name, contents in read]
    if len(tables) == 1:
        name, df = tables[0]
    else:
        stems = [os.path.splitext(n)[0] for n, _ in tables]
        if any(SOURCE_COLUMN in t.columns for _, t in tables):
            raise HTTPException(status_code=400, detail=f"A file already has a '{SOURCE_COLUMN}' column; rename it first.")
        df = pd.concat([t.assign(**{SOURCE_COLUMN: s}) for s, (_, t) in zip(stems, tables)], ignore_index=True)
        name = f"{stems[0]} + {len(tables) - 1} more.csv"

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(df) > settings.MAX_ROWS or df.shape[1] > settings.MAX_COLUMNS:
        raise HTTPException(
            status_code=413,
            detail=f"That table is too big for this demo: at most {settings.MAX_ROWS:,} rows and "
                   f"{settings.MAX_COLUMNS} columns (yours has {len(df):,} rows, {df.shape[1]} columns).",
        )

    session = session_manager.create_session(df, name, owner=visitor)
    return {
        "session_id": session.session_id,
        "current_stage": session.current_stage,
        "summary": session.get_summary(),
        "files": len(tables),
    }
