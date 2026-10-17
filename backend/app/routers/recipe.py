"""Recipe router — the notebook's backend: tools, notebooks, and editing recipe steps.

Work on a notebook (replaying the recipe, saving to storage) runs through run_exclusive:
on a worker thread, one request at a time per notebook.
"""

import re
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.access import current_owner, load_owned
from app.config import settings
from app.session import DataSession, run_exclusive, session_manager
from app.tools import BEFORE_SPLIT_TOOLS, NEEDS_TARGET, TOOL_KINDS, TOOL_SPECS

router = APIRouter()


class StepUpdate(BaseModel):
    args: Dict[str, Any]


def _respond(session: DataSession, message: str) -> Dict[str, Any]:
    if session.models_cleared:
        message += " Trained models were cleared because the data changed; train again to use them."
    session_manager.save(session)
    return {"success": True, "message": message, "new_state_summary": session.get_summary()}


async def _change(session_id: str, visitor: str, change: Callable[[DataSession], str]) -> Dict[str, Any]:
    """Apply a recipe change; a ValueError from it becomes a 400 with its (user-facing) message."""
    session = await load_owned(session_id, visitor)

    def work():
        try:
            message = change(session)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _respond(session, message)

    return await run_exclusive(session.session_id, work)


@router.get("/tools")
async def list_tools():
    """Every tool with its settings schema, so the notebook can build edit forms."""
    return {
        "tools": [
            {**spec, "kind": TOOL_KINDS[name], "before_split": name in BEFORE_SPLIT_TOOLS,
             "needs_target": name in NEEDS_TARGET}
            for name, spec in TOOL_SPECS.items()
            if settings.ALLOW_CUSTOM_CODE or not name.startswith("custom_")
        ]
    }


@router.get("/sessions")
def list_sessions(visitor: str = Depends(current_owner)):
    return {"sessions": session_manager.list_sessions(owner=visitor)}


class CellsUpdate(BaseModel):
    cells: List[Dict[str, Any]] = Field(max_length=2000)


@router.get("/session/{session_id}/cells")
async def get_cells(session_id: str, visitor: str = Depends(current_owner)):
    return {"cells": (await load_owned(session_id, visitor)).cells}


@router.put("/session/{session_id}/cells")
async def save_cells(session_id: str, body: CellsUpdate, visitor: str = Depends(current_owner)):
    """The notebook's cells, saved with the notebook so any browser can reopen it."""
    session = await load_owned(session_id, visitor)

    def work():
        session.cells = body.cells
        session_manager.save(session)
        return {"success": True}

    return await run_exclusive(session.session_id, work)


@router.delete("/session/{session_id}")
async def delete_notebook(session_id: str, visitor: str = Depends(current_owner)):
    session = await load_owned(session_id, visitor)  # 404 unless it exists and belongs to this visitor
    await run_exclusive(session.session_id, session_manager.delete_session, session.session_id)
    return {"success": True}


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", text) or "notebook"


@router.get("/session/{session_id}/export")
async def export(session_id: str, what: str = "train", format: str = "csv", algorithm: Optional[str] = None,
                 visitor: str = Depends(current_owner)):
    """Download cleaned data (what=train|all, format=csv|xlsx), the pipeline (what=script),
    or a trained model with its recipe (what=model, optional algorithm)."""
    from app.export import cleaned_rows, pipeline_script, to_file
    from app.tools.model import model_bundle

    session = await load_owned(session_id, visitor)
    stem = _safe_name(session.filename.rsplit(".", 1)[0])

    def work():
        if what == "model":
            try:
                model_name, body = model_bundle(session, algorithm)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return body, "application/octet-stream", f"{stem}_{_safe_name(model_name)}_model.pkl"
        if what == "script":
            if not session.can_explore:
                raise HTTPException(status_code=400, detail="Choose the target (or Clean only) first: the pipeline starts there.")
            return pipeline_script(session).encode("utf-8"), "text/x-python", f"{stem}_pipeline.py"
        if what in ("train", "all") and format in ("csv", "xlsx"):
            sealed_before = bool(session.sealed) and not session.is_split
            body, media = to_file(cleaned_rows(session, what), format)
            if sealed_before and not session.sealed:  # the export showed the sealed rows: remember that
                session_manager.save(session)
            return body, media, f"{stem}_{'train_clean' if what == 'train' else 'all_clean'}.{format}"
        raise HTTPException(status_code=400, detail="Export what=train|all|script|model, format=csv|xlsx.")

    body, media, name = await run_exclusive(session.session_id, work)
    return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/session/{session_id}/recipe")
async def get_recipe(session_id: str, visitor: str = Depends(current_owner)):
    return {"recipe": (await load_owned(session_id, visitor)).recipe_view()}


@router.put("/session/{session_id}/recipe/{step_id}")
async def update_step(session_id: str, step_id: str, body: StepUpdate, visitor: str = Depends(current_owner)):
    def change(session):
        line = session.update_step(step_id, body.args)
        return f"Updated the step and replayed the recipe. {line.message}"
    return await _change(session_id, visitor, change)


class CodeRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


@router.post("/session/{session_id}/code")
async def show_code(session_id: str, body: CodeRequest, visitor: str = Depends(current_owner)):
    """Code for an action or view cell, and whether it can be edited (recipe steps carry their own code)."""
    from app.action_code import code_info
    return code_info(await load_owned(session_id, visitor), body.tool, body.args)


class CodeUpdate(BaseModel):
    code: str = Field(max_length=50_000)


@router.put("/session/{session_id}/recipe/{step_id}/code")
async def update_step_code(session_id: str, step_id: str, body: CodeUpdate, visitor: str = Depends(current_owner)):
    """Save edited code for a step. A built-in step becomes a custom code step (and remembers what it was)."""
    def change(session):
        from app.sandbox import not_allowed
        reason = not_allowed(session)
        if reason:
            raise ValueError(reason)
        current = session.recipe[session.index_of(step_id)]
        based_on = current.args.get("based_on") if current.tool == "custom_code" else {"tool": current.tool, "args": current.args}
        args = {"code": body.code, **({"based_on": based_on} if based_on else {})}
        line = session.replace_step(step_id, "custom_code", args)
        return f"Saved your code and replayed the recipe. {line.message}"
    return await _change(session_id, visitor, change)


@router.post("/session/{session_id}/recipe/{step_id}/revert")
async def revert_step(session_id: str, step_id: str, visitor: str = Depends(current_owner)):
    """Turn a custom code step back into the built-in step it started from."""
    def change(session):
        current = session.recipe[session.index_of(step_id)]
        based_on = current.args.get("based_on")
        if current.tool != "custom_code" or not based_on:
            raise ValueError("This step wasn't converted from a built-in step, so there's nothing to revert to.")
        line = session.replace_step(step_id, based_on["tool"], based_on["args"])
        return f"Reverted to the built-in '{line.tool.replace('_', ' ')}' step. {line.message}"
    return await _change(session_id, visitor, change)


class Move(BaseModel):
    direction: int  # -1 = earlier, +1 = later


@router.post("/session/{session_id}/recipe/{step_id}/move")
async def move_step(session_id: str, step_id: str, body: Move, visitor: str = Depends(current_owner)):
    def change(session):
        line = session.move_step(step_id, body.direction)
        return f"Moved '{line.tool.replace('_', ' ')}' {'later' if body.direction > 0 else 'earlier'} and replayed the recipe."
    return await _change(session_id, visitor, change)


class HeaderRow(BaseModel):
    row: int = Field(ge=1, le=50)


@router.post("/session/{session_id}/header")
async def set_header_row(session_id: str, body: HeaderRow, visitor: str = Depends(current_owner)):
    """The column names are on another row of the spreadsheet (a title above the table)."""
    return await _change(session_id, visitor, lambda session: session.set_header_row(body.row))


@router.post("/session/{session_id}/restart")
async def restart(session_id: str, visitor: str = Depends(current_owner)):
    """Restart, Jupyter-style: clear trained models and the AI's memory; rebuild the data from scratch."""
    def change(session):
        session.restart()
        session.models_cleared = False  # the message below already says so
        return ("Restarted: trained models and the AI's conversation memory were cleared, and the training "
                f"rows were rebuilt from scratch by replaying {len(session.recipe)} recipe steps.")
    return await _change(session_id, visitor, change)


class StepIds(BaseModel):
    step_ids: List[str] = Field(min_length=1, max_length=500)


@router.post("/session/{session_id}/recipe/delete")
async def delete_steps(session_id: str, body: StepIds, visitor: str = Depends(current_owner)):
    """Undo several steps at once: one replay, all or nothing."""
    def change(session):
        removed = session.delete_steps(body.step_ids)
        names = ", ".join(l.tool.replace("_", " ") for l in removed)
        return f"Removed {len(removed)} step{'s' if len(removed) != 1 else ''} ({names}) and replayed the rest of the recipe."
    return await _change(session_id, visitor, change)


@router.delete("/session/{session_id}/recipe/{step_id}")
async def delete_step(session_id: str, step_id: str, visitor: str = Depends(current_owner)):
    def change(session):
        removed = session.delete_step(step_id)
        return f"Removed '{removed.tool.replace('_', ' ')}' and replayed the rest of the recipe."
    return await _change(session_id, visitor, change)
