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


class FakeGoogle:
    """Stands in for the Gemini client: each model either answers or fails with a status code."""

    def __init__(self, failures):
        self.failures = failures  # model name -> HTTP status it fails with
        self.calls = []
        self.aio = self
        self.models = self

    async def generate_content(self, model, **_):
        from google.genai import errors
        self.calls.append(model)
        if model in self.failures:
            code = self.failures[model]
            error_cls = errors.ServerError if code >= 500 else errors.ClientError
            raise error_cls(code, {"error": {"code": code, "message": "fake", "status": "FAKE"}})
        return _response(types.Part(text=f"answer from {model}"))


class TestBackupModels:
    @pytest.fixture(autouse=True)
    def models(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "GOOGLE_API_KEY", "test-key")
        monkeypatch.setattr(settings, "GEMINI_MODEL", "main")
        monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODELS", "backup1, backup2")

    def ask(self, monkeypatch, session, failures):
        import asyncio
        from app import gemini
        fake = FakeGoogle(failures)
        monkeypatch.setattr(gemini, "_get_client", lambda: fake)
        return asyncio.run(gemini.get_gemini_response(session, "hi")), fake.calls

    def test_main_model_answers(self, monkeypatch, session):
        out, calls = self.ask(monkeypatch, session, {})
        assert out["response"] == "answer from main"
        assert calls == ["main"]

    def test_busy_main_model_falls_back(self, monkeypatch, session):
        out, calls = self.ask(monkeypatch, session, {"main": 503})
        assert out["response"] == "answer from backup1"
        assert calls == ["main", "backup1"]

    def test_rate_limited_falls_back(self, monkeypatch, session):
        out, calls = self.ask(monkeypatch, session, {"main": 429, "backup1": 503})
        assert out["response"] == "answer from backup2"

    def test_all_busy_gives_friendly_message(self, monkeypatch, session):
        from app.gemini import BUSY_MESSAGE
        out, calls = self.ask(monkeypatch, session, {"main": 503, "backup1": 503, "backup2": 500})
        assert out["response"] == BUSY_MESSAGE
        assert calls == ["main", "backup1", "backup2"]

    def test_all_out_of_allowance_says_so(self, monkeypatch, session):
        from app.gemini import QUOTA_MESSAGE
        out, _ = self.ask(monkeypatch, session, {"main": 429, "backup1": 429, "backup2": 429})
        assert out["response"] == QUOTA_MESSAGE

    def test_bad_key_does_not_try_backups(self, monkeypatch, session):
        from app.gemini import KEY_MESSAGE
        out, calls = self.ask(monkeypatch, session, {"main": 401})
        assert out["response"] == KEY_MESSAGE
        assert calls == ["main"]

    def test_bad_request_does_not_try_backups(self, monkeypatch, session):
        from app.gemini import GENERIC_MESSAGE
        out, calls = self.ask(monkeypatch, session, {"main": 400})
        assert out["response"] == GENERIC_MESSAGE
        assert calls == ["main"]

    def test_missing_key_skips_google_entirely(self, monkeypatch, session):
        from app.config import settings
        from app.gemini import KEY_MESSAGE
        monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
        out, calls = self.ask(monkeypatch, session, {})
        assert out["response"] == KEY_MESSAGE
        assert calls == []

    def test_duplicate_models_tried_once(self, monkeypatch, session):
        from app.config import settings
        monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODELS", "main,backup1,backup1")
        _, calls = self.ask(monkeypatch, session, {"main": 503, "backup1": 503})
        assert calls == ["main", "backup1"]


class TestCategoricalColumns:
    def test_detects_pandas3_strings_and_category(self):
        df = pd.DataFrame({
            "s": ["a", "b"],
            "c": pd.Categorical(["x", "y"]),
            "n": [1, 2],
        })
        assert categorical_columns(df) == ["s", "c"]
