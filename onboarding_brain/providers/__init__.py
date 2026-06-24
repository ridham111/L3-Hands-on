"""Model-agnostic LLM provider layer (mock | groq | openrouter | ollama).

Set ONBOARDING_LLM_FALLBACK_BACKEND to chain two providers: every completion
tries the primary first and falls back to the backup on any failure (each
provider's own model-level fallback still applies inside it).
"""
from __future__ import annotations

from ..config import Settings, get_settings
from .base import LLMError, LLMProvider, LLMResult
from .mock_provider import MockProvider


def _make(backend: str, settings: Settings) -> LLMProvider:
    if backend == "groq":
        from .groq_provider import GroqProvider

        return GroqProvider(settings)
    if backend == "openrouter":
        from .openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(settings)
    if backend == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(settings)
    return MockProvider(settings)


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


def get_provider(settings: Settings | None = None, backend: str | None = None) -> LLMProvider:
    """Build the LLM provider. Pass `backend` to force a specific one (e.g. the
    walkthrough uses OpenRouter while chat uses Groq); otherwise use the default
    backend + its configured fallback chain. Falls back to the default backend,
    then mock, if the requested one can't be constructed (missing key/package)."""
    settings = settings or get_settings()
    if backend and backend != settings.backend:
        try:
            return _make(backend, settings)
        except Exception:
            try:
                return _make(settings.backend, settings)
            except Exception:
                return MockProvider(settings)
    primary = _make(settings.backend, settings)
    fb = settings.fallback_backend
    if settings.backend != "mock" and fb and fb not in (settings.backend, "mock"):
        try:
            return FallbackProvider(settings, primary, _make(fb, settings))
        except RuntimeError:
            pass  # backup misconfigured (e.g. missing key) -> primary only
    return primary


__all__ = ["LLMProvider", "LLMError", "MockProvider", "FallbackProvider", "get_provider"]
