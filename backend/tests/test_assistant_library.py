"""The assistant uses the verified library first: a built-in step when one fits, code only when none does.

Every proposed step is checked before anyone sees it (exists now, settings known and valid, real
columns, readable conditions and formulas, judgement calls only when asked), with one retry."""

import asyncio
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app, assistant
from app.assistant import catalogue, check_step, library, read_reply, system_prompt
from app.config import settings
from app.session import DataSession
from app.tools import execute_tool

H = {"X-Visitor-Id": "assistant-library-visitor-01"}


@pytest.fixture
def session():
    df = pd.DataFrame({
        "location": ["Whitefield", "Indira Nagar", "Hebbal", "Yelahanka"] * 15,
        "size": ["2 BHK", "3 BHK", None, "4 Bedroom"] * 15,
        "total_sqft": [1000.0, 1500.0, 1200.0, None] * 15,
        "bath": [2.0, None, 2.0, 3.0] * 15,
        "price": [float(p) for p in range(60, 120)],
    })
    s = DataSession(df, "bengaluru.csv")
    assert execute_tool("select_target", s, {"column": "price"})["success"]
    return s


def step(tool, **arguments):
    return f"Fills the blanks.\n```step\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


def fake_ai(monkeypatch, *answers):
    sent, queue = [], list(answers)

    async def complete(system, messages):
        sent.append({"system": system, "messages": messages})
        return {"text": queue.pop(0), "provider": "Groq", "model": "m"}

    monkeypatch.setattr(assistant, "complete", complete)
    return sent


# ── The catalogue ────────────────────────────────────────────────────────────

def test_catalogue_lists_what_fits_the_notebook_now(session):
    text = catalogue(session)
    assert "impute_missing(column*, strategy: mean|median|mode)" in text
    assert "drop_rows_where(conditions* [e.g. price > 10000" in text  # the syntax example survives
    # Never offered: code (that's what cells are), re-locking, changing target or mode after the lock
    for hidden in ("custom_code", "custom_view", "train_test_split", "select_target", "clean_only", "undo"):
        assert hidden not in library(session)
    assert "train_models(algorithms: " in text and "linear" in text  # regression algorithms only
    assert "svc" not in text.split("train_models")[1].split("\n")[0]


def test_before_the_lock_only_the_start_is_offered():
    s = DataSession(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "x.csv")
    assert set(library(s)) == {"select_target", "clean_only", "preview_data"}


def test_without_code_the_prompt_says_so(session, monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", False)
    assert "Code can't run in this notebook" in system_prompt(session)
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    assert "Code can't run in this notebook" not in system_prompt(session)


# ── Checking a proposed step ─────────────────────────────────────────────────

@pytest.mark.parametrize("proposed, problem", [
    ({"tool": "impute_missing", "arguments": {"column": "baths", "strategy": "median"}}, "no column named baths"),
    ({"tool": "impute_missing", "arguments": {"column": "bath", "strategy": "average"}}, "must be one of"),
    ({"tool": "impute_missing", "arguments": {"strategy": "median"}}, "needs column"),
    ({"tool": "impute_missing", "arguments": {"column": "bath", "how": "median"}}, "has no setting how"),
    ({"tool": "fill_blanks", "arguments": {}}, "isn't a built-in"),
    ({"tool": "custom_code", "arguments": {"code": "x"}}, "isn't a built-in"),
    ({"tool": "train_test_split", "arguments": {}}, "isn't a built-in"),
    ({"tool": "drop_rows_where", "arguments": {"conditions": "sqft > 300"}}, "sqft"),
    ({"tool": "add_column", "arguments": {"name": "x", "formula": "total_sqft /"}}, ""),
    ({"tool": "plot_line", "arguments": {"x": "location", "y": ["prices"]}}, "no column named prices"),
    ("not json at all", "must be JSON"),
])
def test_bad_steps_are_caught(session, proposed, problem):
    checked, why = check_step(session, proposed, "drop rows, fill blanks, add a column feature")
    assert checked is None and problem in why


def test_the_dry_run_catches_what_only_running_would_show(session):
    """A feature built from the target, or touching the target, fails the dry run with the step's own reason."""
    leak = {"tool": "add_column", "arguments": {"name": "pps", "formula": "price / total_sqft"}}
    checked, why = check_step(session, leak, "add a price per sqft feature")
    assert checked is None and "column you're predicting" in why
    rename = {"tool": "rename_column", "arguments": {"column": "price", "new_name": "cost"}}
    assert "can't be renamed" in check_step(session, rename, "rename price to cost")[1]
    # Training-row filters may use the target: that's allowed and passes
    outliers = {"tool": "remove_outliers", "arguments": {"value": "price / total_sqft", "group_by": "location"}}
    assert check_step(session, outliers, "remove the outliers")[0] == outliers
    # And the dry run changed nothing
    assert len(session.recipe) == 0 and "pps" not in session.df.columns


def test_judgement_steps_need_to_be_asked_for(session):
    outliers = {"tool": "remove_outliers", "arguments": {"value": "price / total_sqft", "group_by": "location"}}
    assert check_step(session, outliers, "what should I do next?")[0] is None
    assert check_step(session, outliers, "remove the extreme prices per sqft in each location")[0] == outliers


def test_a_good_step_comes_back_clean(session):
    checked, why = check_step(session, {"tool": "drop_missing_rows", "arguments": {"columns": "size", "x": None}},
                              "drop rows where size is blank")
    assert why is None and checked == {"tool": "drop_missing_rows", "arguments": {"columns": ["size"]}}


def test_a_step_block_on_one_line(session):
    text = 'Shows them. ```step {"tool": "show_unique_values", "arguments": {"column": "location"}}```'
    answer = read_reply(session, text, "values of location")
    assert answer["steps"][0]["tool"] == "show_unique_values" and answer["words"] == "Shows them."


def test_several_steps_are_checked_in_order(session):
    """Yes -> Y then No -> N is two steps; a later step may use a column an earlier one makes."""
    text = ("Two steps.\n```step\n" + json.dumps({"tool": "add_column", "arguments": {"name": "sqft_per_bath", "formula": "total_sqft / bath"}})
            + "\n```\n```step\n" + json.dumps({"tool": "plot_histogram", "arguments": {"column": "sqft_per_bath"}})
            + "\n```\n```step\n" + json.dumps({"tool": "plot_histogram", "arguments": {"column": "nope"}}) + "\n```")
    answer = read_reply(session, text, "add a sqft per bath feature and show its histogram")
    assert [s["tool"] for s in answer["steps"]] == ["add_column", "plot_histogram"]
    assert "no column named nope" in answer["skipped"][0] and "sqft_per_bath" not in session.df


def test_several_steps_reach_the_notebook_and_its_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    two = "\n".join(f"```step\n{json.dumps({'tool': 'replace_values', 'arguments': {'column': 'paid', 'find': a, 'replace_with': b}})}\n```"
                    for a, b in (("Yes", "Y"), ("No", "N")))
    sent = fake_ai(monkeypatch, two, "Done.")
    with TestClient(app, headers=H) as client:
        csv = "paid,x\n" + "\n".join(f"{'Yes' if i % 2 else 'No'},{i}" for i in range(20))
        sid = client.post("/api/upload", files={"file": ("p.csv", csv.encode())}).json()["session_id"]
        client.post("/api/action", json={"session_id": sid, "tool": "clean_only", "arguments": {}})
        r = client.post("/api/assist", json={"session_id": sid, "message": "turn Yes into Y and No into N in paid"}).json()
        assert [s["arguments"]["find"] for s in r["steps"]] == ["Yes", "No"]
        assert r["reply"] == "Built-in steps: replace values, then replace values."
        client.post("/api/assist", json={"session_id": sid, "message": "thanks"})
        assert sent[1]["messages"][-2]["content"].count("```step") == 2  # both remembered


def test_clean_only_notebooks_are_told_there_are_no_models():
    s = DataSession(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}), "c.csv")
    assert execute_tool("clean_only", s, {})["success"]
    prompt = system_prompt(s)
    assert "ONLY CLEANS DATA" in prompt and "train_models" not in catalogue(s)
    assert "replace_values(" in catalogue(s)


def test_settings_beside_the_tool_name_still_count(session):
    loose = {"tool": "filter_rows", "conditions": "bath > 2"}
    assert check_step(session, loose, "show rows with more than 2 baths")[0] == \
        {"tool": "filter_rows", "arguments": {"conditions": "bath > 2"}}


def test_a_step_wins_over_code_in_the_same_answer(session):
    text = step("impute_missing", column="bath", strategy="median") + "\n```python\nresult = train\n```"
    answer = read_reply(session, text, "fill the blanks in bath with the median")
    assert answer["steps"][0]["tool"] == "impute_missing" and "code" not in answer


# ── The whole exchange ───────────────────────────────────────────────────────

def test_a_matching_request_gets_the_verified_step(session, monkeypatch):
    sent = fake_ai(monkeypatch, step("impute_missing", column="bath", strategy="median"))
    result = asyncio.run(assistant.assist(session, "fill the blanks in bath with the median"))
    assert result["step"] == {"tool": "impute_missing", "arguments": {"column": "bath", "strategy": "median"}}
    assert "code" not in result and result["reply"] == "Fills the blanks."
    assert "BUILT-INS AVAILABLE NOW" in sent[0]["system"] and len(sent) == 1
    # And it runs as proposed
    assert execute_tool(result["step"]["tool"], session, result["step"]["arguments"])["success"]


def test_a_wrong_step_is_retried_with_the_reason(session, monkeypatch):
    sent = fake_ai(monkeypatch, step("impute_missing", column="Bath", strategy="median"),
                   step("impute_missing", column="bath", strategy="median"))
    result = asyncio.run(assistant.assist(session, "fill the bath blanks with the median"))
    assert result["step"]["arguments"]["column"] == "bath" and len(sent) == 2
    assert "no column named Bath" in sent[1]["messages"][-1]["content"]


def test_the_retry_may_switch_to_code(session, monkeypatch):
    code = "TRAIN_ONLY = True\n\ndef learn(train):\n    return {}\n\ndef apply(df, learned):\n    return df"
    fake_ai(monkeypatch, step("drop_rows_where", conditions="total_sqft / bhk < 300"), f"```python\n{code}\n```")
    result = asyncio.run(assistant.assist(session, "drop rows where sqft per bhk is under 300"))
    assert result["kind"] == "change" and "step" not in result


def test_a_step_that_still_doesnt_fit_shows_no_button(session, monkeypatch):
    fake_ai(monkeypatch, step("impute_missing", column="Bath"), step("impute_missing", column="BATH"))
    result = asyncio.run(assistant.assist(session, "fill bath"))
    assert "step" not in result and "code" not in result and "doesn't fit" in result["reply"]


def test_unasked_judgement_call_becomes_words_only(session, monkeypatch):
    outliers = step("remove_outliers", value="price")
    fake_ai(monkeypatch, outliers, outliers)
    result = asyncio.run(assistant.assist(session, "what next?"))
    assert "step" not in result


BROKEN_CHANGE = "TRAIN_ONLY = True\n\ndef learn(train):\n    return train\n\ndef apply(df, learned):\n    return df"
FIXED_CHANGE = "TRAIN_ONLY = True\n\ndef learn(train):\n    return {}\n\ndef apply(df, learned):\n    return df[df['bath'] < 10]"


def test_code_that_fails_a_trial_run_is_retried_with_the_error(session, monkeypatch):
    """The eval caught learn() returning a table: the trial run finds it, and the AI fixes it."""
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    sent = fake_ai(monkeypatch, f"```python\n{BROKEN_CHANGE}\n```", f"Fixed.\n```python\n{FIXED_CHANGE}\n```")
    result = asyncio.run(assistant.assist(session, "drop rows with 10+ baths using my own rule"))
    assert result["code"] == FIXED_CHANGE and result["problem"] is None
    assert "must return a dict" in sent[1]["messages"][-1]["content"]
    assert session.recipe == []  # trial runs never touch the notebook


def test_code_still_broken_is_shown_with_its_problem(session, monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    fake_ai(monkeypatch, "```python\nresult = train['nope'].mean()\n```", "```python\nresult = train['nope2'].sum()\n```")
    result = asyncio.run(assistant.assist(session, "mean of nope"))
    assert result["code"] == "result = train['nope'].mean()" and "error when run" in result["problem"]


def test_without_code_on_the_server_only_syntax_is_checked(session, monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", False)
    fake_ai(monkeypatch, "```python\nresult = train['nope'].mean()\n```")
    assert asyncio.run(assistant.assist(session, "mean of nope"))["problem"] is None


def test_number_cleaning_says_which_words_it_dropped():
    s = DataSession(pd.DataFrame({"sqft": ["1000", "1200 - 1400", "188.89Sq. Yards", "6Acres", "900"] * 4,
                                  "price": range(20)}), "x.csv")
    execute_tool("select_target", s, {"column": "price"})
    message = execute_tool("clean_numeric_string", s, {"column": "sqft"})["message"]
    assert "'Sq. Yards'" in message and "'Acres'" in message and "aren't comparable" in message


def test_checking_an_answer_doesnt_stop_the_server(session, monkeypatch):
    """Trial runs happen on the notebook's worker thread: other requests keep being served meanwhile."""
    import time
    monkeypatch.setattr(assistant, "code_problem", lambda s, code: time.sleep(0.6))
    fake_ai(monkeypatch, "```python\nresult = train.head()\n```")

    async def ticks_while_checking():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1
        running = asyncio.create_task(ticker())
        await assistant.assist(session, "show the first rows")
        running.cancel()
        return ticks
    assert asyncio.run(ticks_while_checking()) >= 8  # ~12 in 0.6 s; a blocked loop manages 0 or 1


LOOK_THAT_ADDS = 'classes = {"Whitefield": "upper"}\ndf["class"] = df["location"].map(classes).fillna("unknown")\nresult = df.head()'
CHANGE_THAT_ADDS = ('def learn(train):\n    return {}\n\ndef apply(df, learned):\n'
                    '    df["class"] = df["location"].map({"Whitefield": "upper"}).fillna("unknown")\n    return df')


def test_a_look_cell_that_should_change_the_data_is_rewritten(session, monkeypatch):
    """Asked to divide locations into classes, the AI wrote a Look cell: its column would vanish."""
    monkeypatch.setattr(settings, "ALLOW_CUSTOM_CODE", True)
    sent = fake_ai(monkeypatch, f"```python\n{LOOK_THAT_ADDS}\n```", f"```python\n{CHANGE_THAT_ADDS}\n```")
    result = asyncio.run(assistant.assist(session, "divide these locations into lower, middle and upper class"))
    assert result["kind"] == "change" and "Write it as a CHANGE cell" in sent[1]["messages"][-1]["content"]


def test_a_look_cell_making_a_column_to_show_it_is_fine():
    from app.assistant import look_cell_meant_to_change
    chart = 'train["pps"] = train["price"] / train["total_sqft"]\ntrain["pps"].hist()'
    assert look_cell_meant_to_change(chart, "plot a histogram of price per sqft") is None
    assert look_cell_meant_to_change(LOOK_THAT_ADDS, "categorise the locations by class") is not None


def test_custom_needs_still_get_code(session, monkeypatch):
    fake_ai(monkeypatch, "A custom rule.\n```python\nresult = train.groupby('location')['price'].std()\n```")
    result = asyncio.run(assistant.assist(session, "std of price per location as a table"))
    assert result["kind"] == "look" and "step" not in result


def test_a_yes_after_a_proposal_counts_as_asking(session, monkeypatch):
    session.memory = [{"role": "user", "content": "anything unusual in price?", "kind": "assist"},
                      {"role": "assistant", "content": "You could remove the outliers in price.", "kind": "assist"}]
    fake_ai(monkeypatch, step("remove_outliers", value="price", method="iqr"))
    result = asyncio.run(assistant.assist(session, "yes do it"))
    assert result["step"]["tool"] == "remove_outliers"


def test_endpoint_remembers_the_step(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    sent = fake_ai(monkeypatch, "Locks 20%.\n```step\n{\"tool\": \"select_target\", \"arguments\": {\"column\": \"y\"}}\n```",
                   "Done before: you chose y.")
    with TestClient(app, headers=H) as client:
        csv = "x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(30))
        sid = client.post("/api/upload", files={"file": ("d.csv", csv.encode())}).json()["session_id"]
        r = client.post("/api/assist", json={"session_id": sid, "message": "predict y"}).json()
        assert r["step"] == {"tool": "select_target", "arguments": {"column": "y"}}
        client.post("/api/assist", json={"session_id": sid, "message": "what did you suggest?"})
        assert '"tool": "select_target"' in sent[1]["messages"][-2]["content"]
