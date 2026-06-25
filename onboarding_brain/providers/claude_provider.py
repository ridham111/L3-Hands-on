"""Anthropic Claude backend — authenticates via `claude auth` OAuth (no API key needed)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import Settings
from .base import LLMProvider, LLMResult


def _claude_auth_token() -> str | None:
    """Read the OAuth access token stored by `claude auth` on disk.

    Claude Code stores credentials at ~/.claude/.credentials.json.
    The Python SDK doesn't read this file automatically, so we do it here.
    Returns None if the file doesn't exist or can't be parsed.
    """
    creds_path = Path.home() / ".claude" / ".credentials.json"
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        return data["claudeAiOauth"]["accessToken"]
    except Exception:
        return None


class ClaudeProvider(LLMProvider):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("`anthropic` not installed. pip install anthropic") from exc

        # Credential resolution order:
        # 1. ANTHROPIC_API_KEY env var (traditional API key)
        # 2. ANTHROPIC_AUTH_TOKEN env var (OAuth token set manually)
        # 3. ~/.claude/.credentials.json written by `claude auth`
        api_key = os.getenv("ANTHROPIC_API_KEY")
        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN") or _claude_auth_token()

        if api_key:
            self._client = anthropic.Anthropic(api_key=api_key)
        elif auth_token:
            self._client = anthropic.Anthropic(auth_token=auth_token)
        else:
            raise RuntimeError(
                "Claude: no credentials found. Run `claude auth` or set ANTHROPIC_API_KEY."
            )

        self._model = settings.claude_model

    @property
    def name(self) -> str:
        return f"claude/{self._model}"

    def _complete(self, system: str, user: str) -> LLMResult:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self.settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
        text = next(
            (b.text for b in message.content if b.type == "text"),
            "",
        )
        return LLMResult(text=text, model=f"claude/{self._model}")
