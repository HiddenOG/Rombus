"""Chat router — AI copilot conversation endpoint."""

from fastapi import APIRouter, HTTPException, Depends

from app.auth import verify_api_key
from app.models import ChatRequest, ChatResponse
from app.session import session_manager
from app.copilot import get_copilot_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_copilot(
    request: ChatRequest,
    _: str = Depends(verify_api_key),
):
    """Chat with the AI copilot."""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Update conversation memory
    session.memory.append({"role": "user", "content": request.message})
    if len(session.memory) > 20:
        session.memory = session.memory[-20:]

    # Get AI response
    parsed = await get_copilot_response(session, request.message)

    # Save assistant response to memory
    session.memory.append({"role": "assistant", "content": parsed["response"]})

    # Persist session
    session_manager.save(session)

    return ChatResponse(
        response=parsed["response"],
        suggested_actions=parsed.get("suggested_actions", []),
        current_stage=session.current_stage,
    )
