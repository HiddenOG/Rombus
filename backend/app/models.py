from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any]
    explanation: str


class ChatResponse(BaseModel):
    response: str
    suggested_actions: List[ToolCall]
    current_stage: str


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
