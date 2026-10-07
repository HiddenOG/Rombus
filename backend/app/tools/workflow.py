"""Workflow tools — stage transitions, undo, session management."""

from typing import Dict, Any
from app.tools import register_tool
from app.session import DataSession
from app.workflow_context import STAGES


@register_tool("transition_stage", "Advance to the next ASEMIC stage.")
def transition_stage(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Move to the next ASEMIC workflow stage."""
    try:
        idx = STAGES.index(session.current_stage)
    except ValueError:
        session.current_stage = STAGES[0]
        return {"success": True, "message": f"Reset to {STAGES[0]} stage."}

    if idx >= len(STAGES) - 1:
        return {
            "success": True,
            "message": f"Already at the final stage: {session.current_stage}. Workflow complete!",
        }

    session.current_stage = STAGES[idx + 1]
    return {
        "success": True,
        "message": f"Advanced to {session.current_stage} stage ({idx + 2}/{len(STAGES)}).",
        "stage_transition": session.current_stage,
    }


@register_tool("undo", "Revert the last data transformation.")
def undo(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Undo the last data transformation."""
    if not session.can_undo:
        return {"success": False, "message": "Nothing to undo. No previous snapshots available."}

    success = session.undo()
    if success:
        # Remove the last completed step
        if session.completed_steps:
            removed_step = session.completed_steps.pop()
        else:
            removed_step = "unknown"

        return {
            "success": True,
            "message": f"Undone last action ({removed_step}). DataFrame reverted. {session.undo_count} undo(s) remaining.",
        }
    else:
        return {"success": False, "message": "Failed to undo."}
