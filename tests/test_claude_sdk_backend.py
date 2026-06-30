"""Offline unit tests for the claude_sdk backend wiring.

These never make a live SDK/CLI call — they exercise auth resolution, provider
selection/unwrap, the MCP tool wrapping, and config plumbing. The live agentic
loop is covered by manual/eval runs that need a real subscription token.
"""
import asyncio
import dataclasses

import pytest

from onboarding_brain.config import get_settings
from onboarding_brain.providers import get_provider  # noqa: F401
from onboarding_brain.providers.claude_agent_sdk_provider import (
    ClaudeAgentSDKProvider,
    as_sdk_provider,
    resolve_subscription_auth,
)


def _sdk_settings():
    return dataclasses.replace(get_settings(), backend="claude_sdk")


# ── config plumbing ──────────────────────────────────────────────────────────

def test_model_used_reports_claude_sdk():
    s = _sdk_settings()
    assert s.model_used.startswith("claude_sdk/")


def test_backend_registered_in_factory():
    # The factory's unknown-backend error now lists claude_sdk as a valid option.
    from onboarding_brain.providers import _make
    with pytest.raises(ValueError) as exc:
        _make("nope", _sdk_settings())
    assert "claude_sdk" in str(exc.value)


# ── auth resolution ──────────────────────────────────────────────────────────

def test_auth_rejects_api_key_without_optin(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.delenv("ONBOARDING_CLAUDE_SDK_ALLOW_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as exc:
        resolve_subscription_auth(_sdk_settings())
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_auth_token_passed_through(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok-123")
    env = resolve_subscription_auth(dataclasses.replace(_sdk_settings(), claude_sdk_oauth_token="tok-123"))
    assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "tok-123"


# ── provider unwrap ──────────────────────────────────────────────────────────

def test_as_sdk_provider_direct_and_wrapped(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok-123")
    s = dataclasses.replace(_sdk_settings(), claude_sdk_oauth_token="tok-123")
    prov = ClaudeAgentSDKProvider(s)
    assert as_sdk_provider(prov) is prov

    # Still discoverable if ever wrapped by something exposing `.primary`.
    class _Wrapper:
        primary = prov
    assert as_sdk_provider(_Wrapper()) is prov
    # An unrelated object unwraps to None.
    assert as_sdk_provider(object()) is None


# ── MCP tool wrapping ────────────────────────────────────────────────────────

class _FakeExecutor:
    def __init__(self):
        self.calls = []
        self.used_paths = set()
        self.call_log = []

    def execute(self, name, inp):
        self.calls.append((name, inp))
        return f"ran {name}"


def test_make_sdk_tools_covers_all_definitions_and_routes():
    from onboarding_brain.kt.agent_sdk import _make_sdk_tools
    from onboarding_brain.kt.tools import TOOL_DEFINITIONS

    ex = _FakeExecutor()
    events = []
    sdk_tools = _make_sdk_tools(ex, events.append)

    names = {t.name for t in sdk_tools}
    assert names == {d["name"] for d in TOOL_DEFINITIONS}

    # Each tool keeps the original rich JSON schema (with descriptions/required).
    by_name = {t.name: t for t in sdk_tools}
    src = {d["name"]: d for d in TOOL_DEFINITIONS}
    for n, t in by_name.items():
        assert t.input_schema == src[n]["input_schema"]

    # Invoking a handler routes to executor.execute and emits the UI events.
    search = by_name["search_code"]
    result = asyncio.run(search.handler({"query": "auth"}))
    assert result["content"][0]["text"] == "ran search_code"
    assert ex.calls == [("search_code", {"query": "auth"})]
    assert [e["type"] for e in events] == ["tool_call", "tool_result"]
    assert events[1]["ok"] is True
