from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Verify the API key if authentication is enabled.
    Auth is disabled when API_SECRET_KEY is empty.
    """
    # If no secret key is configured, auth is disabled
    if not settings.API_SECRET_KEY:
        return None

    if not api_key or api_key != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )
    return api_key
