"""Accounts: moving an anonymous visitor's notebooks into their account when they sign in."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.access import signed_in_user, visitor_id
from app.session import session_manager

router = APIRouter()


@router.post("/account/claim")
async def claim_visitor_notebooks(user: Optional[str] = Depends(signed_in_user), visitor: str = Depends(visitor_id)):
    """Give the notebooks this browser made before signing in to the signed-in account.
    Needs both the login token and this browser's visitor id (which only this browser knows)."""
    if not user:
        raise HTTPException(status_code=401, detail="Sign in first.")
    return {"moved": session_manager.transfer_owner(visitor, user)}
