"""Action router — tool execution endpoint."""

from fastapi import APIRouter, HTTPException, Depends

from app.auth import verify_api_key
from app.models import ActionRequest, ActionResponse
from app.session import session_manager
from app.tools import execute_tool

router = APIRouter()


@router.post("/action", response_model=ActionResponse)
async def execute_action(
    request: ActionRequest,
    _: str = Depends(verify_api_key),
):
    """Execute a data science tool/action."""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Execute the tool
    result = execute_tool(request.tool, session, request.arguments)

    if not result.get("success", False):
        return ActionResponse(
            success=False,
            message=result.get("message", "Action failed."),
        )

    # Update session state
    session.completed_steps.append(request.tool)
    session.workflow_log.append({
        "tool": request.tool,
        "message": result["message"],
        "stage": session.current_stage,
    })

    # Add to conversation memory for context
    session.memory.append({
        "role": "assistant",
        "content": f"Action Executed: {request.tool}. Result: {result['message']}",
    })

    # Persist session
    session_manager.save(session)

    return ActionResponse(
        success=True,
        message=result["message"],
        evidence=result.get("evidence"),
        new_state_summary=session.get_summary(),
        stage_transition=result.get("stage_transition"),
    )
