"""Chat router — AI copilot conversation endpoint."""

from fastapi import APIRouter, Depends

from app.access import current_owner, limit, load_owned
from app.copilot import get_copilot_response
from app.models import ChatRequest, ChatResponse
from app.session import run_exclusive, session_manager

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(limit("ai"))])
async def chat_with_copilot(request: ChatRequest, visitor: str = Depends(current_owner)):
    """Chat with the AI copilot."""
    session = await load_owned(request.session_id, visitor)

    # Update conversation memory
    session.memory.append({"role": "user", "content": request.message})
    if len(session.memory) > 20:
        session.memory = session.memory[-20:]

    # Get AI response
    parsed = await get_copilot_response(session, request.message)
    failed = bool(parsed.get("failure"))

    # Save the assistant's reply to memory, but not error notices: the next AI would read them
    # as if it had said them
    if not failed:
        session.memory.append({"role": "assistant", "content": parsed["response"]})

    # Persist session (a network call with cloud storage: off the main worker)
    await run_exclusive(session.session_id, session_manager.save, session)

    return ChatResponse(
        response=parsed["response"],
        suggested_actions=parsed.get("suggested_actions", []),
        current_stage=session.current_stage,
        provider=parsed.get("provider"),
        model=parsed.get("model"),
        usage=parsed.get("usage"),
        fallback_note=parsed.get("fallback_note"),
        failed=failed,
    )
