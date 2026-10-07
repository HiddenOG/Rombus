"""Tests for the Gemini copilot integration (no network calls)."""

import pandas as pd
import pytest
from google.genai import types

from app.gemini import (
    MAX_SUGGESTIONS,
    build_contents,
    build_system_instruction,
    build_tool_declarations,
    parse_response,
)
from app.session import DataSession
from app.tools import TOOL_REGISTRY, categorical_columns


@pytest.fixture
def session():
    df = pd.DataFrame({"age": [20, 30, None], "city": ["a", "b", "a"]})
    return DataSession(df, "people.csv")


def _response(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


class TestDeclarations:
    def test_every_tool_is_declared(self):
        names = {d["name"] for d in build_tool_declarations()}
        assert names == set(TOOL_REGISTRY)

    def test_explanation_is_required(self):
        for decl in build_tool_declarations():
            assert "explanation" in decl["parameters_json_schema"]["required"]

    def test_config_validates_against_sdk(self, session):
        config = types.GenerateContentConfig.model_validate({
            "system_instruction": build_system_instruction(session),
            "tools": [{"function_declarations": build_tool_declarations()}],
            "automatic_function_calling": {"disable": True},
        })
        assert len(config.tools[0].function_declarations) == len(TOOL_REGISTRY)


class TestContents:
    def test_current_message_not_duplicated(self, session):
        session.memory = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "clean it"},
        ]
        contents = build_contents(session, "clean it")
        assert [c["role"] for c in contents] == ["user", "model", "user"]
        assert contents[-1]["parts"][0]["text"] == "clean it"

    def test_system_instruction_lists_columns(self, session):
        text = build_system_instruction(session)
        assert "age" in text and "city" in text


class TestParseResponse:
    def test_text_and_function_calls(self):
        resp = _response(
            types.Part(text="Fill the gaps first."),
            types.Part(function_call=types.FunctionCall(
                name="impute_missing",
                args={"column": "age", "strategy": "median", "explanation": "age has gaps"},
            )),
        )
        parsed = parse_response(resp)
        assert parsed["response"] == "Fill the gaps first."
        assert parsed["suggested_actions"] == [{
            "tool": "impute_missing",
            "arguments": {"column": "age", "strategy": "median"},
            "explanation": "age has gaps",
        }]

    def test_unknown_tool_dropped(self):
        resp = _response(types.Part(function_call=types.FunctionCall(name="rm_rf", args={})))
        assert parse_response(resp)["suggested_actions"] == []

    def test_calls_only_gets_default_text(self):
        resp = _response(types.Part(function_call=types.FunctionCall(name="undo", args={})))
        parsed = parse_response(resp)
        assert parsed["response"]
        assert parsed["suggested_actions"][0]["tool"] == "undo"

    def test_suggestions_capped(self):
        calls = [types.Part(function_call=types.FunctionCall(name="undo", args={})) for _ in range(5)]
        assert len(parse_response(_response(*calls))["suggested_actions"]) == MAX_SUGGESTIONS

    def test_thoughts_hidden(self):
        resp = _response(types.Part(text="secret reasoning", thought=True), types.Part(text="Answer"))
        assert parse_response(resp)["response"] == "Answer"

    def test_empty_response(self):
        parsed = parse_response(types.GenerateContentResponse(candidates=[]))
        assert parsed["suggested_actions"] == []


class TestCategoricalColumns:
    def test_detects_pandas3_strings_and_category(self):
        df = pd.DataFrame({
            "s": ["a", "b"],
            "c": pd.Categorical(["x", "y"]),
            "n": [1, 2],
        })
        assert categorical_columns(df) == ["s", "c"]
