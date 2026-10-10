"""Who may use which notebook, and how much.

Visitors don't log in (yet): each browser makes a random, unguessable visitor id and sends it as
the X-Visitor-Id header. A notebook belongs to the visitor who created it; nobody else can list,
open, change or delete it. Notebooks from before owners existed are claimed by the first visitor
who opens them (on a laptop that's you; a fresh deployment has none).

Usage limits protect free AI allowances and the server: per visitor, per network address
(a backstop if someone keeps making new visitor ids), and a daily total for the AI.
"""

import re
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import Depends, Header, HTTPException, Request

from app.config import settings
from app.session import DataSession, session_manager

VISITOR_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def visitor_id(x_visitor_id: Optional[str] = Header(None)) -> str:
    if not x_visitor_id or not VISITOR_ID.match(x_visitor_id):
        raise HTTPException(status_code=401, detail="Missing or invalid visitor id. Reload the page and try again.")
    return x_visitor_id


def owned_session(session_id: str, visitor: str = Depends(visitor_id)) -> DataSession:
    """The notebook, if it belongs to this visitor. Others get 'not found' (we don't reveal it exists)."""
    session = session_manager.get_session(session_id)
    if session is None or (session.owner and session.owner != visitor):
        raise HTTPException(status_code=404, detail="Notebook not found.")
    if not session.owner:  # from before owners existed: claim it
        session.owner = visitor
        session_manager.save(session)
    return session


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
    # Behind a hosting proxy (e.g. Railway) the visitor's address is the first X-Forwarded-For entry
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def limit(kind: str):
    """A dependency enforcing the per-visitor, per-address (3x) and, for AI, daily total limits."""
    window = {"ai": 60, "upload": 3600}[kind]
    unit = {"ai": "minute", "upload": "hour"}[kind]

    def check(request: Request, visitor: str = Depends(visitor_id)) -> None:
        if not settings.RATE_LIMITS:
            return
        # Read the limits on every request, so changed settings apply without a restart
        per_minute = {"ai": settings.AI_PER_MINUTE, "upload": settings.UPLOADS_PER_HOUR}[kind]
        message = None
        if not _hit(f"{kind}:visitor:{visitor}", per_minute, window):
            message = f"You're going a bit fast: at most {per_minute} {'AI messages' if kind == 'ai' else 'uploads'} per {unit}."
        elif not _hit(f"{kind}:ip:{client_address(request)}", per_minute * 3, window):
            message = "Too many requests from your network. Please wait a moment."
        elif kind == "ai" and not _hit("ai:daily", settings.AI_DAILY_LIMIT, 86400):
            message = "The demo's AI allowance for today is used up. You can still add steps yourself with Add step."
        if message:
            raise HTTPException(status_code=429, detail=message)

    return check


def reset_limits() -> None:
    """For tests."""
    _events.clear()
