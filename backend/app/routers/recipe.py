"""Recipe router — the notebook's backend: tools, notebooks, and editing recipe steps."""

from typing import Any, Dict, List

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


class CellsUpdate(BaseModel):
    cells: List[Dict[str, Any]]


@router.get("/session/{session_id}/cells")
async def get_cells(session_id: str, _: str = Depends(verify_api_key)):
    return {"cells": _session(session_id).cells}


@router.put("/session/{session_id}/cells")
async def save_cells(session_id: str, body: CellsUpdate, _: str = Depends(verify_api_key)):
    """The notebook's cells, saved with the notebook so any browser can reopen it."""
    session = _session(session_id)
    session.cells = body.cells
    session_manager.save(session)
    return {"success": True}


@router.delete("/session/{session_id}")
async def delete_notebook(session_id: str, _: str = Depends(verify_api_key)):
    _session(session_id)  # 404 if it doesn't exist
    session_manager.delete_session(session_id)
    return {"success": True}


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


class CodeRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


@router.post("/session/{session_id}/code")
async def show_code(session_id: str, body: CodeRequest, _: str = Depends(verify_api_key)):
    """Equivalent Python for an action or view cell (recipe steps carry their own code)."""
    from app.action_code import action_code
    return {"code": action_code(_session(session_id), body.tool, body.args)}


class CodeUpdate(BaseModel):
    code: str


@router.put("/session/{session_id}/recipe/{step_id}/code")
async def update_step_code(session_id: str, step_id: str, body: CodeUpdate, _: str = Depends(verify_api_key)):
    """Save edited code for a step. A built-in step becomes a custom code step (and remembers what it was)."""
    session = _session(session_id)
    try:
        current = session.recipe[session.index_of(step_id)]
        based_on = current.args.get("based_on") if current.tool == "custom_code" else {"tool": current.tool, "args": current.args}
        args = {"code": body.code, **({"based_on": based_on} if based_on else {})}
        line = session.replace_step(step_id, "custom_code", args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Saved your code and replayed the recipe. {line.message}")


@router.post("/session/{session_id}/recipe/{step_id}/revert")
async def revert_step(session_id: str, step_id: str, _: str = Depends(verify_api_key)):
    """Turn a custom code step back into the built-in step it started from."""
    session = _session(session_id)
    try:
        current = session.recipe[session.index_of(step_id)]
        based_on = current.args.get("based_on")
        if current.tool != "custom_code" or not based_on:
            raise ValueError("This step wasn't converted from a built-in step, so there's nothing to revert to.")
        line = session.replace_step(step_id, based_on["tool"], based_on["args"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Reverted to the built-in '{line.tool.replace('_', ' ')}' step. {line.message}")


@router.delete("/session/{session_id}/recipe/{step_id}")
async def delete_step(session_id: str, step_id: str, _: str = Depends(verify_api_key)):
    session = _session(session_id)
    try:
        removed = session.delete_step(step_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Removed '{removed.tool.replace('_', ' ')}' and replayed the rest of the recipe.")
