"""Recipe router — list, edit and delete recipe steps (the notebook's cells)."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import verify_api_key
from app.session import DataSession, session_manager

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


@router.get("/session/{session_id}/recipe")
async def get_recipe(session_id: str, _: str = Depends(verify_api_key)):
    return {"recipe": _session(session_id).recipe_view()}


@router.put("/session/{session_id}/recipe/{index}")
async def update_step(session_id: str, index: int, body: StepUpdate, _: str = Depends(verify_api_key)):
    session = _session(session_id)
    try:
        line = session.update_step(index, body.args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Updated step {index + 1} and replayed the recipe. {line.message}")


@router.delete("/session/{session_id}/recipe/{index}")
async def delete_step(session_id: str, index: int, _: str = Depends(verify_api_key)):
    session = _session(session_id)
    try:
        removed = session.delete_step(index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Removed step {index + 1} ({removed.tool.replace('_', ' ')}) and replayed the rest.")
