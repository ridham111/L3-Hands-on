"""LLM provider interface + retry/timeout."""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id: str
    name: str
    input: dict


@dataclass
class TurnResult:
    """Result of one LLM turn in the agentic loop."""
    stop_reason: str                        # "end_turn" | "tool_use" | "max_tokens"
    text: str                               # final text when stop_reason == "end_turn"
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: Any = None                 # provider-native blocks re-sent verbatim


class LLMProvider(abc.ABC):
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def _complete(self, system: str, user: str) -> LLMResult: ...

    def complete(self, system: str, user: str) -> LLMResult:
        attempts = max(1, self.settings.max_retries + 1)
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._complete(system, user)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < attempts - 1:
                    time.sleep(min(2.0, 0.5 * (2 ** attempt)))
        raise LLMError(f"{self.name} failed after {attempts} attempt(s): {last}") from last
