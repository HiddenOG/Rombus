"""Recipe router — the notebook's backend: tools, notebooks, and editing recipe steps."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.access import current_owner, owned_session
from app.config import settings
from app.session import DataSession, session_manager
from app.tools import BEFORE_SPLIT_TOOLS, TOOL_KINDS, TOOL_SPECS

router = APIRouter()


class StepUpdate(BaseModel):
    args: Dict[str, Any]



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
            if settings.ALLOW_CUSTOM_CODE or not name.startswith("custom_")
        ]
    }


@router.get("/sessions")
async def list_sessions(visitor: str = Depends(current_owner)):
    return {"sessions": session_manager.list_sessions(owner=visitor)}


class CellsUpdate(BaseModel):
    cells: List[Dict[str, Any]]


@router.get("/session/{session_id}/cells")
async def get_cells(session_id: str, visitor: str = Depends(current_owner)):
    return {"cells": owned_session(session_id, visitor).cells}


@router.put("/session/{session_id}/cells")
async def save_cells(session_id: str, body: CellsUpdate, visitor: str = Depends(current_owner)):
    """The notebook's cells, saved with the notebook so any browser can reopen it."""
    session = owned_session(session_id, visitor)
    session.cells = body.cells
    session_manager.save(session)
    return {"success": True}


@router.delete("/session/{session_id}")
async def delete_notebook(session_id: str, visitor: str = Depends(current_owner)):
    owned_session(session_id, visitor)  # 404 unless it exists and belongs to this visitor
    session_manager.delete_session(session_id)
    return {"success": True}


@router.get("/session/{session_id}/export")
async def export(session_id: str, what: str = "train", format: str = "csv", visitor: str = Depends(current_owner)):
    """Download cleaned data (what=train|all, format=csv|xlsx) or the pipeline (what=script)."""
    import re
    from fastapi.responses import Response
    from app.export import cleaned_rows, pipeline_script, to_file

    session = owned_session(session_id, visitor)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", session.filename.rsplit(".", 1)[0]) or "notebook"
    if what == "script":
        if not session.is_split:
            raise HTTPException(status_code=400, detail="Choose the target first: the pipeline starts with the lock.")
        body, media, name = pipeline_script(session).encode("utf-8"), "text/x-python", f"{stem}_pipeline.py"
    elif what in ("train", "all") and format in ("csv", "xlsx"):
        body, media = to_file(cleaned_rows(session, what), format)
        name = f"{stem}_{'train_clean' if what == 'train' else 'all_clean'}.{format}"
    else:
        raise HTTPException(status_code=400, detail="Export what=train|all|script, format=csv|xlsx.")
    return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/session/{session_id}/recipe")
async def get_recipe(session_id: str, visitor: str = Depends(current_owner)):
    return {"recipe": owned_session(session_id, visitor).recipe_view()}


@router.put("/session/{session_id}/recipe/{step_id}")
async def update_step(session_id: str, step_id: str, body: StepUpdate, visitor: str = Depends(current_owner)):
    session = owned_session(session_id, visitor)
    try:
        line = session.update_step(step_id, body.args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Updated the step and replayed the recipe. {line.message}")


class CodeRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


@router.post("/session/{session_id}/code")
async def show_code(session_id: str, body: CodeRequest, visitor: str = Depends(current_owner)):
    """Code for an action or view cell, and whether it can be edited (recipe steps carry their own code)."""
    from app.action_code import code_info
    return code_info(owned_session(session_id, visitor), body.tool, body.args)


class CodeUpdate(BaseModel):
    code: str


@router.put("/session/{session_id}/recipe/{step_id}/code")
async def update_step_code(session_id: str, step_id: str, body: CodeUpdate, visitor: str = Depends(current_owner)):
    """Save edited code for a step. A built-in step becomes a custom code step (and remembers what it was)."""
    session = owned_session(session_id, visitor)
    try:
        current = session.recipe[session.index_of(step_id)]
        based_on = current.args.get("based_on") if current.tool == "custom_code" else {"tool": current.tool, "args": current.args}
        args = {"code": body.code, **({"based_on": based_on} if based_on else {})}
        line = session.replace_step(step_id, "custom_code", args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Saved your code and replayed the recipe. {line.message}")


@router.post("/session/{session_id}/recipe/{step_id}/revert")
async def revert_step(session_id: str, step_id: str, visitor: str = Depends(current_owner)):
    """Turn a custom code step back into the built-in step it started from."""
    session = owned_session(session_id, visitor)
    try:
        current = session.recipe[session.index_of(step_id)]
        based_on = current.args.get("based_on")
        if current.tool != "custom_code" or not based_on:
            raise ValueError("This step wasn't converted from a built-in step, so there's nothing to revert to.")
        line = session.replace_step(step_id, based_on["tool"], based_on["args"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Reverted to the built-in '{line.tool.replace('_', ' ')}' step. {line.message}")


class Move(BaseModel):
    direction: int  # -1 = earlier, +1 = later


@router.post("/session/{session_id}/recipe/{step_id}/move")
async def move_step(session_id: str, step_id: str, body: Move, visitor: str = Depends(current_owner)):
    session = owned_session(session_id, visitor)
    try:
        line = session.move_step(step_id, body.direction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Moved '{line.tool.replace('_', ' ')}' {'later' if body.direction > 0 else 'earlier'} "
                             f"and replayed the recipe.")


@router.post("/session/{session_id}/restart")
async def restart(session_id: str, visitor: str = Depends(current_owner)):
    """Restart, Jupyter-style: clear trained models and the AI's memory; rebuild the data from scratch."""
    session = owned_session(session_id, visitor)
    session.restart()
    session_manager.save(session)
    return {
        "success": True,
        "message": "Restarted: trained models and the AI's conversation memory were cleared, and the training "
                   f"rows were rebuilt from scratch by replaying {len(session.recipe)} recipe steps.",
        "new_state_summary": session.get_summary(),
    }


@router.delete("/session/{session_id}/recipe/{step_id}")
async def delete_step(session_id: str, step_id: str, visitor: str = Depends(current_owner)):
    session = owned_session(session_id, visitor)
    try:
        removed = session.delete_step(step_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _respond(session, f"Removed '{removed.tool.replace('_', ' ')}' and replayed the rest of the recipe.")
