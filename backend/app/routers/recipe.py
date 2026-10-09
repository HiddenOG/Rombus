"""Recipe router — the notebook's backend: tools, notebooks, and editing recipe steps."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import verify_api_key
from app.session import DataSession, session_manager
from app.tools import BEFORE_SPLIT_TOOLS, TOOL_KINDS, TOOL_SPECS

router = APIRouter()


class StepUpdate(BaseModel):
    args: Dict[str, Any]


def _session(session_id: str) -> DataSession:
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _respond(session: DataSession, message: str) -> Dict[str, Any]:
    if session.models_cleared:
        message += " Trained models were cleared because the data changed; train again to use them."
    session_manager.save(session)
    return {"success": True, "message": message, "new_state_summary": session.get_summary()}


@router.get("/tools")
async def list_tools():
    """Every tool with its settings schema, so the notebook can build edit forms."""
    return {
        "tools": [
            {**spec, "kind": TOOL_KINDS[name], "before_split": name in BEFORE_SPLIT_TOOLS}
            for name, spec in TOOL_SPECS.items()
        ]
    }


@router.get("/sessions")
async def list_sessions(_: str = Depends(verify_api_key)):
    return {"sessions": session_manager.list_sessions()}


@router.get("/session/{session_id}/recipe")
async def get_recipe(session_id: str, _: str = Depends(verify_api_key)):
    return {"recipe": _session(session_id).recipe_view()}


@router.put("/session/{session_id}/recipe/{step_id}")
async def update_step(session_id: str, step_id: str, body: StepUpdate, _: str = Depends(verify_api_key)):
    session = _session(session_id)
    try:
        line = session.update_step(step_id, body.args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Updated the step and replayed the recipe. {line.message}")


@router.delete("/session/{session_id}/recipe/{step_id}")
async def delete_step(session_id: str, step_id: str, _: str = Depends(verify_api_key)):
    session = _session(session_id)
    try:
        removed = session.delete_step(step_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Removed '{removed.tool.replace('_', ' ')}' and replayed the rest of the recipe.")
