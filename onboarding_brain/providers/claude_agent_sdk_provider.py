"""Claude Agent SDK backend — drives the bundled Claude Code CLI via claude-agent-sdk.

Unlike `claude_provider` (which calls the raw Anthropic /v1/messages API), this
backend runs Anthropic's *agent harness*: the SDK spawns the Claude Code CLI as a
subprocess and that CLI owns the tool-use loop. We authenticate with a Claude
**Pro/Max subscription over OAuth** — no billed API key.

Two roles:
  • `_complete()` — a single-shot, tool-less completion used by every helper
    (briefing, install guide, tour, walkthrough, chat condense/RAG). Implements the
    LLMProvider contract so those call sites work unchanged.
  • The agentic chat path is NOT here — because the SDK owns the loop, it can't
    expose a per-turn `complete_turn()`. That path lives in `kt/agent_sdk.py`,
    which registers the 9 KT tools as an in-process MCP server and runs the loop.

Async→sync bridge: the SDK is async-only; the rest of the app is sync (and runs
inside FastAPI's threadpool). We run each coroutine to completion in a dedicated
worker thread with its own event loop, so this is safe whether or not the caller
already has a running loop.
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from .base import LLMProvider, LLMResult


def _run_sync(coro_factory: Callable[[], Any]) -> Any:
    """Run an async coroutine to completion from sync code.

    Always executes in a fresh thread with a private event loop so it works even
    when the calling thread already has a running loop. `coro_factory` is a
    zero-arg callable that builds the coroutine *inside* the worker thread.
    """
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 — re-raised on the calling thread
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def resolve_subscription_auth(settings: Settings) -> dict[str, str]:
    """Return env overrides forcing subscription OAuth, or raise with guidance.

    The CLI's credential precedence puts ANTHROPIC_API_KEY *above* OAuth, so if a
    key is present it would silently bill the API. We refuse to start in that
    state unless the user explicitly opts in via ONBOARDING_CLAUDE_SDK_ALLOW_API_KEY.
    """
    token = settings.claude_sdk_oauth_token or os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "")
    creds_exist = (Path.home() / ".claude" / ".credentials.json").exists()
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    allow_api_key = os.getenv("ONBOARDING_CLAUDE_SDK_ALLOW_API_KEY", "").lower() in (
        "1", "true", "yes", "on")

    if has_api_key and not allow_api_key:
        raise RuntimeError(
            "claude_sdk backend: ANTHROPIC_API_KEY is set, which overrides subscription "
            "OAuth and would BILL the API. Unset it to use your Claude Pro/Max "
            "subscription, or set ONBOARDING_CLAUDE_SDK_ALLOW_API_KEY=1 to use the key "
            "on purpose."
        )
    if not token and not creds_exist and not (has_api_key and allow_api_key):
        raise RuntimeError(
            "claude_sdk backend: no subscription credentials found. Run `claude setup-token` "
            "and set CLAUDE_CODE_OAUTH_TOKEN, or log in once with `claude`."
        )

    env: dict[str, str] = {}
    if token:
        # Pass the long-lived token through to the CLI subprocess explicitly.
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


def as_sdk_provider(provider: Any) -> "ClaudeAgentSDKProvider | None":
    """Return the ClaudeAgentSDKProvider if `provider` is one (or wraps one as the
    primary of a FallbackProvider), else None. Lets callers route the agentic loop
    to the SDK even when a single-shot fallback backend is configured behind it."""
    if isinstance(provider, ClaudeAgentSDKProvider):
        return provider
    primary = getattr(provider, "primary", None)
    if isinstance(primary, ClaudeAgentSDKProvider):
        return primary
    return None


class ClaudeAgentSDKProvider(LLMProvider):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "`claude-agent-sdk` not installed. pip install claude-agent-sdk"
            ) from exc

        # Fail fast at construction if auth is misconfigured (mirrors ClaudeProvider).
        self._auth_env = resolve_subscription_auth(settings)
        self._model = settings.claude_sdk_model  # "" => CLI subscription default

    @property
    def name(self) -> str:
        return f"claude_sdk/{self._model or 'subscription-default'}"

    def base_options(self, **overrides: Any):
        """ClaudeAgentOptions shared by both the single-shot and agentic paths.

        Locks the harness down to an offline, grounded posture: no project/user
        settings are loaded, and (unless overridden) no tools are permitted.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        opts: dict[str, Any] = dict(
            model=self._model or None,
            permission_mode="bypassPermissions",  # only our own read-only tools run
            setting_sources=None,                  # ignore ambient CLAUDE.md / settings
            tools=[],                              # disable ALL built-in tools (Read/Bash/
                                                   # ToolSearch/…) so only our MCP tools exist
                                                   # — keeps the agent grounded and avoids the
                                                   # tool-search deferral indirection
            allowed_tools=[],                      # default: nothing allowed (single-shot)
            env=dict(self._auth_env),
        )
        opts.update(overrides)
        return ClaudeAgentOptions(**opts)

    def _complete(self, system: str, user: str) -> LLMResult:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            query,
        )

        async def _run() -> str:
            options = self.base_options(
                system_prompt=system,
                max_turns=1,
            )
            final: str | None = None
            text_parts: list[str] = []
            async for msg in query(prompt=user, options=options):
                if isinstance(msg, AssistantMessage):
                    if msg.error:
                        raise RuntimeError(f"claude_sdk error: {msg.error}")
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                elif isinstance(msg, ResultMessage):
                    if msg.is_error:
                        raise RuntimeError(
                            f"claude_sdk result error: {msg.errors or msg.subtype}"
                        )
                    final = msg.result
            return (final or "".join(text_parts)).strip()

        text = _run_sync(_run)
        return LLMResult(text=text, model=self.name)
