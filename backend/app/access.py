"""Who may use which notebook, and how much.

Two kinds of owner:
  - a signed-in user (Supabase Auth): the browser sends the login token as
    "Authorization: Bearer <token>"; the owner is "user:<their id>", the same on every device.
  - an anonymous visitor: each browser makes a random, unguessable visitor id and sends it as the
    X-Visitor-Id header. Like a cloakroom ticket, it only works in that browser.
A notebook belongs to the owner who created it; nobody else can list, open, change or delete it.
When a visitor signs in, their notebooks move to their account (see routers/account.py).
Notebooks from before owners existed are claimed by the first owner who opens them.

Login tokens are checked without any secret: Supabase signs them with a private key and publishes
the matching public key, which proves a token is genuine, unchanged and not expired.

Usage limits protect free AI allowances and the server: per owner, per network address
(a backstop if someone keeps making new visitor ids), and a daily total for the AI.
"""

import re
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

import jwt
from fastapi import Depends, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.session import DataSession, session_manager

VISITOR_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def visitor_id(x_visitor_id: Optional[str] = Header(None)) -> str:
    if not x_visitor_id or not VISITOR_ID.match(x_visitor_id):
        raise HTTPException(status_code=401, detail="Missing or invalid visitor id. Reload the page and try again.")
    return x_visitor_id


_jwks: Dict[str, jwt.PyJWKClient] = {}


def _signing_key(token: str):
    """The Supabase public key that matches this token (fetched once, then cached)."""
    base = settings.SUPABASE_URL.rstrip("/")
    if base not in _jwks:
        _jwks[base] = jwt.PyJWKClient(f"{base}/auth/v1/.well-known/jwks.json", cache_keys=True)
    return _jwks[base].get_signing_key_from_jwt(token).key


def signed_in_user(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """'user:<id>' for a valid login token, None when there's no token. A bad token is refused
    (never silently treated as anonymous, which would show the wrong notebooks)."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid sign-in. Please sign in again.")
    if not settings.SUPABASE_URL:
        raise HTTPException(status_code=401, detail="Accounts aren't set up on this server.")
    try:
        claims = jwt.decode(
            token, _signing_key(token), algorithms=["ES256", "RS256"],
            audience="authenticated", issuer=f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1",
        )
    except (jwt.InvalidTokenError, jwt.PyJWKClientError):
        raise HTTPException(status_code=401, detail="Your sign-in has expired. Please sign in again.")
    return f"user:{claims['sub']}"


def current_owner(user: Optional[str] = Depends(signed_in_user),
                  x_visitor_id: Optional[str] = Header(None)) -> str:
    """Whoever is asking: the signed-in user if there is one, otherwise the anonymous visitor."""
    return user or visitor_id(x_visitor_id)


def owned_session(session_id: str, visitor: str = Depends(current_owner)) -> DataSession:
    """The notebook, if it belongs to this visitor. Others get 'not found' (we don't reveal it exists)."""
    session = session_manager.get_session(session_id)
    if session is None or (session.owner and session.owner != visitor):
        raise HTTPException(status_code=404, detail="Notebook not found.")
    if not session.owner:  # from before owners existed: claim it
        session.owner = visitor
        session_manager.save(session)
    return session


async def load_owned(session_id: str, visitor: str) -> DataSession:
    """owned_session on a worker thread: opening a notebook that isn't in memory means downloading
    it from storage and replaying its recipe, which shouldn't hold up everyone else's requests."""
    return await run_in_threadpool(owned_session, session_id, visitor)


# ── Usage limits ─────────────────────────────────────────────────────────────────

_events: Dict[str, Deque[float]] = defaultdict(deque)


def _hit(key: str, limit: int, window_seconds: float) -> bool:
    """Record one event for `key`; False if that would exceed `limit` within the window."""
    now = time.time()
    events = _events[key]
    while events and events[0] <= now - window_seconds:
        events.popleft()
    if len(events) >= limit:
        return False
    events.append(now)
    return True


def client_address(request: Request) -> str:
    """The visitor's network address. Behind a hosting proxy (e.g. Railway) it's in X-Forwarded-For.
    Use the LAST entry: that's the one our proxy added. Earlier entries come from the visitor, who
    could write anything there to dodge the per-address limit."""
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    return forwarded[-1] if forwarded else (request.client.host if request.client else "unknown")


# kind: (setting with the allowance, window in seconds, unit, what's being counted)
LIMITS = {
    "ai": ("AI_PER_MINUTE", 60, "minute", "AI messages"),
    "upload": ("UPLOADS_PER_HOUR", 3600, "hour", "uploads"),
    "compute": ("TRAININGS_PER_HOUR", 3600, "hour", "training runs"),
}


def enforce(kind: str, request: Request, visitor: str) -> None:
    """Per-owner, per-address (3x) and, for AI, daily total limits. Raises 429 when exceeded."""
    if not settings.RATE_LIMITS:
        return
    setting, window, unit, what = LIMITS[kind]
    allowed = getattr(settings, setting)  # read on every request, so changes apply without a restart
    message = None
    if not _hit(f"{kind}:visitor:{visitor}", allowed, window):
        message = f"You're going a bit fast: at most {allowed} {what} per {unit}."
    elif not _hit(f"{kind}:ip:{client_address(request)}", allowed * 3, window):
        message = "Too many requests from your network. Please wait a moment."
    elif kind == "ai" and not _hit("ai:daily", settings.AI_DAILY_LIMIT, 86400):
        message = "The demo's AI allowance for today is used up. You can still add steps yourself with Add step."
    if message:
        raise HTTPException(status_code=429, detail=message)


def limit(kind: str):
    """The same limits as a route dependency."""
    def check(request: Request, visitor: str = Depends(current_owner)) -> None:
        enforce(kind, request, visitor)
    return check


def reset_limits() -> None:
    """For tests."""
    _events.clear()
