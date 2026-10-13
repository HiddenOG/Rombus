"""Chat router — AI copilot conversation endpoint."""

import json

from fastapi import APIRouter, Depends

from app.access import current_owner, limit, load_owned
from app.copilot import get_copilot_response
from app.copilot_prompt import memory_entry
from app.models import ChatRequest, ChatResponse
from app.session import run_exclusive, session_manager

router = APIRouter()


@router.post("/assist", dependencies=[Depends(limit("ai"))])
async def assist(request: ChatRequest, visitor: str = Depends(current_owner)):
    """The notebook assistant: answers, or writes one code cell for the user to read and run."""
    from app.assistant import assist as ask_assistant
    session = await load_owned(request.session_id, visitor)
    result = await ask_assistant(session, request.message)
    if not result.get("failed"):
        # Remembered (with the step or code it proposed), so follow-ups like "now plot it" make sense
        said = result["reply"] + (f"\n```python\n{result['code']}\n```" if result.get("code") else "")
        if result.get("step"):
            said += f"\n```step\n{json.dumps(result['step'])}\n```"
        session.memory += [{"role": "user", "content": request.message, "kind": "assist"},
                           {"role": "assistant", "content": said, "kind": "assist"}]
        session.memory = session.memory[-20:]
        await run_exclusive(session.session_id, session_manager.save, session)
    return result


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
        session.memory.append(memory_entry(parsed))

    # Persist session (a network call with cloud storage: off the main worker)
    await run_exclusive(session.session_id, session_manager.save, session)

    # A notice (e.g. "no step came with this answer") is shown, but not remembered: kept out of the
    # history, so the model never sees it as something it said and copies it
    shown = parsed["response"] + (f"\n\n{parsed['notice']}" if parsed.get("notice") else "")
    return ChatResponse(
        response=shown,
        suggested_actions=parsed.get("suggested_actions", []),
        current_stage=session.current_stage,
        provider=parsed.get("provider"),
        model=parsed.get("model"),
        usage=parsed.get("usage"),
        fallback_note=parsed.get("fallback_note"),
        failed=failed,
    )
