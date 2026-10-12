"""A safety net under the AI: obvious requests get their Run button even if the model forgets it.

The model usually understands "drop duplicates" or "drop em" (right after counting duplicates),
but sometimes answers in words only, and only a function call makes a Run button. When a reply
comes back with no suggestion at all, these plain rules look at the message and recent history
for one clear step. Rules, not AI: predictable, instant, and the same with every provider.

Deliberately limited:
  - only when the AI suggested nothing (it never overrides the AI's choice)
  - only steps that need no extra settings, or a column named in the message
  - never the final exam or re-locking the test rows: those must be a deliberate click
"""

import re
from typing import Any, Dict, List, Optional

from app.session import DataSession
from app.tools import available_tools

# Clear commands -> step. Each pattern needs an action word AND a subject, so a question like
# "how many duplicates are there?" doesn't turn into a step.
# Word stems, so typos and tenses still match ("droped", "removing", "duplocates")
_ACT = r"(drop\w*|remov\w*|delet\w*|get rid|clear\w*|clean\w*|eliminat\w*|dedup\w*)"
COMMANDS = [
    (re.compile(rf"\b{_ACT}.*\bdup\w*|\bdup\w*.*\b{_ACT}|\bdedup\w*"), "remove_duplicates"),
    (re.compile(r"\b(fill|impute|handle|fix|replace|treat)\b.*\b(missing|nulls?|nans?|blanks?|empty|gaps?)\b"
                r"|\b(missing|nulls?|nans?|blanks?|gaps?)\b.*\b(fill|impute|handle|fix|replace)\b"), "impute_all_missing"),
    (re.compile(r"\b(encode|encoding|one[- ]?hot|label[- ]encode|dummies)\b"), "encode_all_categorical"),
    (re.compile(r"\b(scale|scaling|normali[sz]e|standardi[sz]e)\b"), "scale_all_features"),
    (re.compile(r"\b(train|fit|build)\b.*\bmodels?\b|\bmodels?\b.*\b(train|fit)\b"), "train_models"),
    (re.compile(r"\b(auto ?ml|tune|tuning|hyper ?param)"), "auto_ml"),
]

# Short replies that mean "yes, do that"
FOLLOW_UP = re.compile(
    r"^\s*(yes|yeah|yep|yup|ok|okay|sure|please|do it|do that|go|go ahead|proceed|run it|"
    r"(drop|remove|delete|fix|clean|fill|do) (em|them|those|it|that|these)|ok do it|yes please)\b[\s.!]*",
    re.IGNORECASE)

# After this view ran, a "yes, do it" means this step
NEXT_AFTER_VIEW = {"check_duplicates": "remove_duplicates"}

# Steps the safety net may offer without settings from the user
NO_SETTINGS = {"remove_duplicates", "impute_all_missing", "encode_all_categorical", "scale_all_features",
               "train_models", "auto_ml"}

_SUGGESTED = re.compile(r"\[Suggested as Run buttons: ([a-z_, ]+)\]")
_EXECUTED = re.compile(r"^Action Executed: (\w+)\.")


def obvious_step(session: DataSession, message: str) -> Optional[Dict[str, Any]]:
    """The one clear step this message asks for, as a suggested action, or None."""
    allowed = set(available_tools(session))
    text = message.lower()

    tool = _column_removal(session, text)
    if tool:
        return tool if "remove_columns" in allowed else None

    for pattern, name in COMMANDS:
        if pattern.search(text):
            return _offer(session, name, allowed, f"You asked for this, so here it is: {_title(name)}.")

    if FOLLOW_UP.match(message) and len(message.split()) <= 6:
        name = _previous_step(session)
        if name:
            return _offer(session, name, allowed, f"Following on from the previous cell: {_title(name)}.")
    return None


def _offer(session: DataSession, name: str, allowed: set, why: str) -> Optional[Dict[str, Any]]:
    if name not in allowed or name not in NO_SETTINGS:
        return None
    if name == "remove_duplicates" and not session.df.duplicated().any():
        return None  # nothing to remove; the AI's words can say so
    return {"tool": name, "arguments": {}, "explanation": why}


def _column_removal(session: DataSession, text: str) -> Optional[Dict[str, Any]]:
    """'drop the society column', 'remove balcony and bath' -> remove_columns with those columns."""
    if not re.search(rf"\b{_ACT}", text) or re.search(r"\bdup\w*|\brows?\b", text):
        return None
    named = [c for c in session.df.columns
             if c != session.target_column and re.search(rf"(?<![\w]){re.escape(str(c).lower())}(?![\w])", text)]
    if not named:
        return None
    return {"tool": "remove_columns", "arguments": {"columns": named},
            "explanation": f"You asked to remove {', '.join(named)}."}


def _previous_step(session: DataSession) -> Optional[str]:
    """What 'yes / do it' refers to: the AI's latest suggestion, or the natural next step after the
    latest view. Looks back past a few word-only replies (the AI talking without suggesting)."""
    for entry in list(reversed(_earlier_assistant_entries(session)))[:4]:
        suggested = _SUGGESTED.search(entry)
        if suggested:
            return suggested.group(1).split(",")[0].strip()
        executed = _EXECUTED.match(entry)
        if executed:
            return NEXT_AFTER_VIEW.get(executed.group(1))  # an action ran since: only its natural next step
    return None


def _earlier_assistant_entries(session: DataSession) -> List[str]:
    return [m["content"] for m in session.memory if m.get("role") == "assistant"]


def _title(name: str) -> str:
    return name.replace("_", " ").capitalize()
