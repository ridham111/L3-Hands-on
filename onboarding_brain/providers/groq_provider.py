"""Groq backend (free tier, OpenAI-compatible). https://console.groq.com/keys

Free-tier models have separate per-model capacity limits (tokens/minute and
tokens/day), so when the primary model hits 429/413 we retry once on the
fallback model instead of degrading to "couldn't find this in the indexed
code". Configure with GROQ_MODEL / GROQ_FALLBACK_MODEL.
"""
from __future__ import annotations

from ..config import Settings
from .base import LLMProvider, LLMResult


def _should_fallback(exc: Exception) -> bool:
    """Switch to the fallback model on capacity limits AND on model-availability
    errors — so a decommissioned/unknown primary model ID degrades gracefully
    to the fallback instead of failing the whole request."""
    if getattr(exc, "status_code", None) in (400, 404, 413, 429):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "rate_limit", "too large", "429", "404", "400",
        "decommission", "does not exist", "not found", "unavailable", "invalid model",
    ))


class GroqProvider(LLMProvider):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.groq_api_key:
            raise RuntimeError(
                "ONBOARDING_LLM_BACKEND=groq but GROQ_API_KEY is empty. "
                "Get a free key at https://console.groq.com/keys or use the mock backend."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("`groq` not installed. pip install groq") from exc
        self._client = Groq(api_key=settings.groq_api_key, timeout=settings.request_timeout_s)

    @property
    def name(self) -> str:
        return f"groq/{self.settings.groq_model}"

    def _call(self, model: str, system: str, user: str) -> LLMResult:
        resp = self._client.chat.completions.create(
            model=model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return LLMResult(text=resp.choices[0].message.content or "", model=f"groq/{model}")

    def _complete(self, system: str, user: str) -> LLMResult:
        try:
            return self._call(self.settings.groq_model, system, user)
        except Exception as exc:
            fallback = self.settings.groq_fallback_model
            if fallback and fallback != self.settings.groq_model and _should_fallback(exc):
                return self._call(fallback, system, user)
            raise
