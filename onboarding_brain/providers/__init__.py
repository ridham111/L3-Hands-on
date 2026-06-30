"""LLM provider layer — single backend: the Claude Agent SDK.

This project is a PURE AGENT. There is exactly one backend (`claude_sdk`): the
agent harness from `claude-agent-sdk`, which owns the tool-use loop. The app only
supplies the 9 code-aware tools as an in-process MCP server. There is deliberately
no alternative-LLM multiplexing or fallback chain — the value is the agent itself.
"""
from __future__ import annotations

from ..config import Settings, get_settings
from .base import LLMError, LLMProvider, LLMResult


def _make(backend: str, settings: Settings) -> LLMProvider:
    if backend == "claude_sdk":
        from .claude_agent_sdk_provider import ClaudeAgentSDKProvider
        return ClaudeAgentSDKProvider(settings)
    raise ValueError(
        f"Unknown backend: {backend!r}. This is a pure Claude Agent SDK app — the "
        f"only valid backend is 'claude_sdk'."
    )


def get_provider(settings: Settings | None = None, backend: str | None = None) -> LLMProvider:
    """Build the agent-SDK provider. `backend` is accepted for call-site
    compatibility but only `claude_sdk` exists; anything else falls back to it."""
    settings = settings or get_settings()
    return _make("claude_sdk", settings)


__all__ = ["LLMProvider", "LLMError", "get_provider"]
