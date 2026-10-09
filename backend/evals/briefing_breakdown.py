"""Rough breakdown of what the copilot briefing is made of (≈ 4 characters per token).

Run from backend/:  python -m evals.briefing_breakdown
"""

import json

from app.copilot_prompt import STATIC_INSTRUCTIONS, state_snapshot
from app.ollama_client import build_tools
from evals.copilot_eval import cleaned, fresh, locked


def tokens(text: str) -> int:
    return round(len(text) / 4)


for label, setup in [("before lock", fresh), ("after lock", locked), ("ready to train", cleaned)]:
    session = setup()
    tools = build_tools(session)
    tool_tokens = {t["function"]["name"]: tokens(json.dumps(t)) for t in tools}
    print(f"\n{label}: instructions≈{tokens(STATIC_INSTRUCTIONS)}  state≈{tokens(state_snapshot(session))}  "
          f"tools≈{sum(tool_tokens.values())} ({len(tools)} tools)")
    for name, n in sorted(tool_tokens.items(), key=lambda kv: -kv[1])[:6]:
        print(f"   {name:24} ≈{n}")
