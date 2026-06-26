"""Model-agnostic LLM provider layer (groq | openrouter | claude).

Set ONBOARDING_LLM_FALLBACK_BACKEND to chain two providers: every completion
tries the primary first and falls back to the backup on any failure (each
provider's own model-level fallback still applies inside it).
"""
from __future__ import annotations

from ..config import Settings, get_settings
from .base import LLMError, LLMProvider, LLMResult


def _make(backend: str, settings: Settings) -> LLMProvider:
    if backend == "groq":
        from .groq_provider import GroqProvider
        return GroqProvider(settings)
    if backend == "openrouter":
        from .openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(settings)
    if backend == "claude":
        from .claude_provider import ClaudeProvider
        return ClaudeProvider(settings)
    raise ValueError(f"Unknown backend: {backend!r}. Valid options: claude | groq | openrouter")


class FallbackProvider(LLMProvider):
    """Primary provider with a different-backend backup behind it."""

    def __init__(self, settings: Settings, primary: LLMProvider, backup: LLMProvider):
        super().__init__(settings)
        self.primary = primary
        self.backup = backup

    @property
    def name(self) -> str:
        return f"{self.primary.name} (backup: {self.backup.name})"

    def _complete(self, system: str, user: str) -> LLMResult:
        try:
            return self.primary._complete(system, user)
        except Exception:
            return self.backup._complete(system, user)

    def complete_turn(self, system: str, messages: list, tools: list):
        """Delegate agentic tool-use turns to the primary if it supports them."""
        if hasattr(self.primary, "complete_turn"):
            return self.primary.complete_turn(system, messages, tools)
        raise LLMError(f"Primary provider {self.primary.name!r} does not support tool use")


def get_provider(settings: Settings | None = None, backend: str | None = None) -> LLMProvider:
    """Build the LLM provider. Pass `backend` to force a specific one (e.g. the
    walkthrough uses OpenRouter while chat uses Groq); otherwise use the default
    backend + its configured fallback chain."""
    settings = settings or get_settings()
    if backend and backend != settings.backend:
        try:
            return _make(backend, settings)
        except Exception:
            return _make(settings.backend, settings)
    primary = _make(settings.backend, settings)
    fb = settings.fallback_backend
    if fb and fb != settings.backend:
        try:
            return FallbackProvider(settings, primary, _make(fb, settings))
        except RuntimeError:
            pass  # backup misconfigured (e.g. missing key) -> primary only
    return primary


__all__ = ["LLMProvider", "LLMError", "FallbackProvider", "get_provider"]
