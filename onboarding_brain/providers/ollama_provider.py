"""Ollama backend (local, no key). Run `ollama serve` first."""
from __future__ import annotations

import httpx

from ..config import Settings
from .base import LLMProvider, LLMResult


class OllamaProvider(LLMProvider):
    @property
    def name(self) -> str:
        return f"ollama/{self.settings.ollama_model}"

    def _complete(self, system: str, user: str) -> LLMResult:
        url = self.settings.ollama_host.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self.settings.ollama_model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "response_format": {"type": "json_object"},  # OpenAI-compat JSON mode
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        r = httpx.post(url, json=payload, timeout=self.settings.request_timeout_s)
        r.raise_for_status()
        return LLMResult(text=r.json()["choices"][0]["message"]["content"] or "", model=self.name)
