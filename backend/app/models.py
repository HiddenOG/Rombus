from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ChatRequest(BaseModel):
    session_id: str
    # Long enough for any real question; stops one message using up the AI allowance
    message: str = Field(min_length=1, max_length=4000)


class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any]
    explanation: str


class ChatResponse(BaseModel):
    response: str
    suggested_actions: List[ToolCall]
    current_stage: str
    # Which AI answered (provider and model), tokens used, and a note if the chain fell back
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Optional[int]]] = None
    fallback_note: Optional[str] = None
    failed: bool = False


class ActionRequest(BaseModel):
    session_id: str
    tool: str
    arguments: Dict[str, Any]


class ActionResponse(BaseModel):
    success: bool
    message: str
    evidence: Optional[Dict[str, Any]] = None
    new_state_summary: Optional[Dict[str, Any]] = None
    stage_transition: Optional[str] = None
    # The recipe line this action created, if it changed data (lets a notebook cell link to it)
    step: Optional[Dict[str, Any]] = None


class SessionInfo(BaseModel):
    session_id: str
    filename: str
    current_stage: str
    shape: List[int]
    target: Optional[str] = None
    problem_type: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    active_sessions: int
