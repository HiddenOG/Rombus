"""Upload router — file upload endpoint."""

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.access import limit, visitor_id
from app.config import settings
from app.session import session_manager

router = APIRouter()

ALLOWED = (".csv", ".xlsx", ".xls")


@router.post("/upload", dependencies=[Depends(limit("upload"))])
async def upload_file(file: UploadFile = File(...), visitor: str = Depends(visitor_id)):
    """Upload a CSV or Excel file and create a new notebook owned by this visitor."""
    name = file.filename or "uploaded_file.csv"
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(status_code=400, detail="Please upload a .csv, .xlsx or .xls file.")

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB} MB.")

    try:
        df = pd.read_excel(io.BytesIO(contents)) if name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't read that file as a table: {e}")

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
    }
