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

# Talk of blanks (typos included: "nul vales"), of dropping or filling them, and of looking at them
_MISSING = re.compile(r"\b(nul\w*|nans?|missing|blanks?|empty|n/?a)\b")
_DROP = re.compile(r"\b(drop\w*|remov\w*|delet\w*|eliminat\w*|get rid)")
_FILL = re.compile(r"\b(fill\w*|imput\w*|replac\w*)\b")
_SHOW_MISSING = re.compile(r"\b(where|visuali[sz]\w*|map|heat ?map|plot|show|see|chart)\b.*\b(nul\w*|nans?|missing|blanks?)\b"
                           r"|\b(nul\w*|missing|blanks?)\b.*\b(map|heat ?map|plot|chart|pattern)\b")
_UNIQUE = re.compile(r"\b(unique|distinct|value ?counts?|categories|array list|all the values)\b")

# Short replies that mean "yes, do that"
FOLLOW_UP = re.compile(
    r"^\s*(yes|yeah|yep|yup|ok|okay|sure|please|do it|do that|go|go ahead|proceed|run it|"
    r"(drop|remove|delete|fix|clean|fill|do) (em|them|those|it|that|these)|ok do it|yes please)\b[\s.!]*",
    re.IGNORECASE)

# After this view ran, a "yes, do it" means this step
NEXT_AFTER_VIEW = {"check_duplicates": "remove_duplicates"}

# Steps the safety net may offer without settings from the user
NO_SETTINGS = {"remove_duplicates", "drop_missing_rows", "impute_all_missing", "encode_all_categorical",
               "scale_all_features", "train_models", "auto_ml", "plot_missing", "check_duplicates"}

_SUGGESTED = re.compile(r"\[Suggested as Run buttons: ([a-z_, ]+)\]")  # how an older version wrote them
_EXECUTED = re.compile(r"^Action Executed: (\w+)\.")


def obvious_step(session: DataSession, message: str) -> Optional[Dict[str, Any]]:
    """The one clear step this message asks for, as a suggested action, or None."""
    allowed = set(available_tools(session))
    text = message.lower()

    # "drop the nulls in size": rows with blanks, NOT the size column (an earlier version got this wrong)
    if _DROP.search(text) and _MISSING.search(text) and not _FILL.search(text):
        if "drop_missing_rows" not in allowed:
            return None
        cols = _named_columns(session, text)
        where = f" in {', '.join(cols)}" if cols else ""
        return {"tool": "drop_missing_rows", "arguments": {"columns": cols} if cols else {},
                "explanation": f"You asked to drop the rows with blanks{where} (training rows only)."}

    tool = _column_removal(session, text)
    if tool:
        return tool if "remove_columns" in allowed else None

    # Looking, not changing: "where are the nulls", "unique values of size"
    if _SHOW_MISSING.search(text) and "plot_missing" in allowed:
        return {"tool": "plot_missing", "arguments": {}, "explanation": "A map of where the blanks are."}
    if _UNIQUE.search(text) and "show_unique_values" in allowed:
        cols = _named_columns(session, text) or ([session.target_column] if "target" in text else [])
        if cols:
            return {"tool": "show_unique_values", "arguments": {"column": cols[0]},
                    "explanation": f"The distinct values of {cols[0]}, with how many rows have each."}

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


# Judgement calls the AI may only suggest when the user asks for that kind of change. Each is allowed
# when the user's message contains one of its words (typos tolerated by the stems).
ON_REQUEST = {
    "remove_outliers": r"outlier|extreme|unusual|odd|weird|anomal|std|standard dev|iqr",
    "drop_rows_where": r"drop\w*|remov\w*|delet\w*|exclud\w*|filter\w*|get rid|rows? (where|with)",
    "group_rare_categories": r"rare|other|group\w*|merge|combin\w*|reduce\w*|too many|categor\w*|dimension",
    "add_column": r"new (column|feature)|feature|add\w* (a )?column|create\w*|ratio|formula|per\b|engineer\w*|combin\w*|deriv\w*",
    "change_type": r"type|convert\w*|astype|int\w*|float|numer\w*|whole number|text|string",
    "rename_column": r"renam\w*|name",
    "fill_value": r"fill\w*|replac\w*|impute|blank|null|nan|missing|zero|unknown",
}


def asked_for(tool: str, message: str, session: Optional[DataSession] = None) -> bool:
    """Whether the user asked for this kind of change (always true for ordinary steps).
    A short "yes / do it" also counts what the copilot proposed just before: agreeing is asking."""
    words = ON_REQUEST.get(tool)
    if words is None:
        return True
    request = message
    if session is not None and FOLLOW_UP.match(message) and len(message.split()) <= 6:
        earlier = [m["content"] for m in session.memory[:-1] if m.get("role") == "assistant"]
        request += " " + (earlier[-1] if earlier else "")
    return bool(re.search(words, request.lower()))


def _named_columns(session: DataSession, text: str) -> List[str]:
    return [c for c in session.df.columns
            if c != session.target_column and re.search(rf"(?<![\w]){re.escape(str(c).lower())}(?![\w])", text)]


def _column_removal(session: DataSession, text: str) -> Optional[Dict[str, Any]]:
    """'drop the society column', 'remove balcony and bath' -> remove_columns with those columns.
    Not when the message is about rows, duplicates or blanks: those mean rows, not columns."""
    if not re.search(rf"\b{_ACT}", text) or re.search(r"\bdup\w*|\brows?\b", text) or _MISSING.search(text):
        return None
    named = _named_columns(session, text)
    if not named:
        return None
    return {"tool": "remove_columns", "arguments": {"columns": named},
            "explanation": f"You asked to remove {', '.join(named)}."}


def _previous_step(session: DataSession) -> Optional[str]:
    """What 'yes / do it' refers to: the AI's latest suggestion, or the natural next step after the
    latest view. Looks back past a few word-only replies (the AI talking without suggesting)."""
    for entry in list(reversed([m for m in session.memory if m.get("role") == "assistant"]))[:4]:
        if entry.get("suggested"):
            return entry["suggested"][0]
        suggested = _SUGGESTED.search(entry["content"])  # notebooks saved by an older version
        if suggested:
            return suggested.group(1).split(",")[0].strip()
        executed = _EXECUTED.match(entry["content"])
        if executed:
            return NEXT_AFTER_VIEW.get(executed.group(1))  # an action ran since: only its natural next step
    return None


def _title(name: str) -> str:
    return name.replace("_", " ").capitalize()
