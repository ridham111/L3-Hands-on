"""Deterministic, offline provider — EVAL/TEST USE ONLY.

This is a test double, NOT a user-facing backend. The only way it enters the
system is via `install_stub()`, which monkeypatches `get_provider` in the modules
that build providers. The one real backend stays `claude_sdk`.

Why it exists: the regression gate must be HERMETIC and REPEATABLE — same input,
same output, no network, no API key, no cost. A real LLM phrases answers
differently every run, so exact-match eval checks can only be satisfied by a
deterministic generator. It produces grounded chat/walkthrough output purely from
the retrieved context (citing only files that exist), passing grounding checks by
construction.
"""
from __future__ import annotations

import json
import re

from onboarding_brain.providers.base import LLMProvider, LLMResult

def _stub_chat(user: str) -> dict:
    """Deterministic chat answer: cite exactly the files present in the retrieved
    CONTEXT (as a real grounded model would), so source/grounding checks are
    stable. Headers look like `[path:line-line · lang]`."""
    paths: list[str] = []
    for p in re.findall(r"\[([^\]]+?):\d+-\d+", user):
        if p not in paths:
            paths.append(p)
    if not paths:
        return {"answer": "I couldn't find this in the indexed code.",
                "general_note": "", "used_sources": [], "confidence": 0.3}
    refs = ", ".join(f"`{p}`" for p in paths[:4])
    return {
        "answer": f"Based on the retrieved code in {refs}, here is how it works: "
                  f"the relevant logic lives in these files and they cover the question.",
        "general_note": "",
        "used_sources": paths,
        "confidence": 0.9,
    }


def _stub_walkthrough(user: str) -> dict:
    """Deterministic walkthrough section: grounded in the section's files, with a
    couple of takeaways — exercises the takeaways path too."""
    files = re.findall(r"\[([^\]]+?) ·", user)
    files = list(dict.fromkeys(files))
    if not files:
        return {"explanation": "This part isn't present in the indexed code.", "takeaways": []}
    refs = ", ".join(f"`{p}`" for p in files[:5])
    return {
        "explanation": f"This section is built from {refs}. These files work together to "
                       f"implement the area described, and reading them in order shows the flow.",
        "takeaways": [f"Start with `{files[0]}` for this area.",
                      "These files are grounded in the real codebase."],
    }


class StubProvider(LLMProvider):
    """Deterministic, multi-purpose test double. Dispatches by prompt shape:
    chat -> grounded answer citing context files; walkthrough -> section body +
    takeaways; else -> empty (condense falls back to chat.py's offline condenser).
    It is not the SDK agent provider, so chat falls through to the deterministic RAG
    pipeline (the offline test harness) rather than the live Claude Agent SDK loop."""

    @property
    def name(self) -> str:
        return "stub/deterministic-v1"

    def _complete(self, system: str, user: str) -> LLMResult:
        if "CONTEXT (retrieved snippets)" in user:
            payload = _stub_chat(user)
        elif "Walkthrough section" in user:
            payload = _stub_walkthrough(user)
        else:
            # e.g. the condense prompt — carries no "question" key, so chat.py
            # falls back to its deterministic offline condenser.
            payload = {}
        return LLMResult(text=json.dumps(payload, ensure_ascii=False), model=self.name)


def install_stub(settings, *, patch_source: bool = True) -> StubProvider:
    """Monkeypatch `get_provider` in the agent-flow modules so the whole system
    uses the deterministic StubProvider. Returns the instance. EVAL/TEST USE
    ONLY — this keeps the stub out of the user-facing backend dispatch entirely.

    patch_source=False leaves `onboarding_brain.providers.get_provider` itself
    untouched, so tests that call it directly still see the real factory, while the
    agent flows (which bind get_provider at import) still get the stub."""
    stub = StubProvider(settings)

    def _factory(*_a, **_k):
        return stub

    import onboarding_brain.onboarding as _onboarding
    import onboarding_brain.kt.chat as _chat
    import onboarding_brain.kt.agent as _agent
    import onboarding_brain.kt.walkthrough as _walk
    import onboarding_brain.kt.tour as _tour

    targets = [_onboarding, _chat, _agent, _walk, _tour]
    if patch_source:
        import onboarding_brain.providers as _providers
        targets.append(_providers)

    for mod in targets:
        if hasattr(mod, "get_provider"):
            mod.get_provider = _factory
    return stub
