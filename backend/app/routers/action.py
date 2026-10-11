"""Action router — tool execution endpoint."""

from fastapi import APIRouter, Depends, Request

from app.access import current_owner, enforce, load_owned
from app.models import ActionRequest, ActionResponse
from app.session import DataSession, run_exclusive, session_manager
from app.tools import execute_tool

router = APIRouter()

# Tools that keep the CPU busy for seconds (or minutes): counted against TRAININGS_PER_HOUR
HEAVY_TOOLS = {"train_models", "auto_ml", "explain_model", "custom_models", "final_exam"}


@router.post("/action", response_model=ActionResponse)
async def execute_action(request: ActionRequest, http: Request, visitor: str = Depends(current_owner)):
    """Execute a data science tool/action."""
    session = await load_owned(request.session_id, visitor)
    if request.tool in HEAVY_TOOLS:
        enforce("compute", http, visitor)
    return await run_exclusive(session.session_id, _run, session, request)


def _run(session: DataSession, request: ActionRequest) -> ActionResponse:
    session.models_cleared = False
    result = execute_tool(request.tool, session, request.arguments)

    if not result.get("success", False):
        return ActionResponse(
            success=False,
            message=result.get("message", "Action failed."),
        )

    if session.models_cleared and request.tool not in ("train_test_split", "train_models", "auto_ml"):
        result["message"] += " Trained models were cleared because the data changed; train again to use them."

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
    session.memory = session.memory[-20:]
    session.workflow_log = session.workflow_log[-50:]
    session.completed_steps = session.completed_steps[-100:]

    # Persist session
    session_manager.save(session)

    step = None
    if result.get("step_id"):
        index = session.index_of(result["step_id"])
        step = session.step_view(session.recipe[index], index)

    return ActionResponse(
        success=True,
        message=result["message"],
        evidence=result.get("evidence"),
        new_state_summary=session.get_summary(),
        stage_transition=result.get("stage_transition"),
        step=step,
    )
