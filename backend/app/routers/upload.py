"""Upload router — file upload endpoint."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import pandas as pd
import io

from app.auth import verify_api_key
from app.session import session_manager
from app.config import settings

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    """Upload a CSV or Excel file and create a new session."""
    try:
        contents = await file.read()

        # Check file size
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )

        # Parse the file
        if file.filename and file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Create session
        session = session_manager.create_session(df, file.filename or "uploaded_file.csv")

        return {
            "session_id": session.session_id,
            "current_stage": session.current_stage,
            "summary": session.get_summary(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
