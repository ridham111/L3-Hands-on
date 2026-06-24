"""OpenRouter backend (OpenAI-compatible gateway to many models, several free).
https://openrouter.ai/keys

Notes for free models: per-day request caps apply, and large reasoning models
can be slow — raise ONBOARDING_REQUEST_TIMEOUT_S if needed. On capacity errors
(402/429/5xx) we retry once on OPENROUTER_FALLBACK_MODEL when configured.
"""
from __future__ import annotations

import httpx

from ..config import Settings
from .base import LLMProvider, LLMResult

# capacity AND model-availability hints — both warrant trying the fallback model
_CAPACITY_HINTS = ("rate", "limit", "quota", "capacity", "overloaded", "temporarily",
                   "empty completion", "not found", "unavailable", "no endpoints",
                   "decommission", "does not exist", "invalid model")


def _is_capacity_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(f" {code}" in msg or f":{code}" in msg for code in ("400", "402", "404", "408", "429", "502", "503")):
        return True
    return any(h in msg for h in _CAPACITY_HINTS)


class OpenRouterProvider(LLMProvider):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "ONBOARDING_LLM_BACKEND=openrouter but OPENROUTER_API_KEY is empty. "
                "Get a key at https://openrouter.ai/keys or use another backend."
            )
        self._client = httpx.Client(
            base_url=settings.openrouter_base,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "X-Title": "KT Brain",
            },
            timeout=settings.request_timeout_s,
        )

    @property
    def name(self) -> str:
        return f"openrouter/{self.settings.openrouter_model}"

    def _call(self, model: str, system: str, user: str) -> LLMResult:
        resp = self._client.post("/chat/completions", json={
            "model": model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        })
        if resp.status_code != 200:
            raise RuntimeError(f"openrouter {model} HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"openrouter {model} error: {str(data['error'])[:300]}")
        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        # reasoning models may put the final text in `content` and their chain
        # in `reasoning`; fall back if content comes back empty
        text = (msg.get("content") or msg.get("reasoning") or "").strip()
        if not text:
            raise RuntimeError(f"openrouter {model} returned an empty completion")
        return LLMResult(text=text, model=f"openrouter/{model}")

    def _complete(self, system: str, user: str) -> LLMResult:
        try:
            return self._call(self.settings.openrouter_model, system, user)
        except Exception as exc:
            fallback = self.settings.openrouter_fallback_model
            if fallback and fallback != self.settings.openrouter_model and _is_capacity_error(exc):
                return self._call(fallback, system, user)
            raise
