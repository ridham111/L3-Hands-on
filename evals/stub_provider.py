"""Deterministic, offline provider — EVAL/TEST USE ONLY.

This is a test double, NOT a user-facing backend. It is never registered in
`onboarding_brain.providers._make()`; the only way it enters the system is via
`install_stub()`, which monkeypatches `get_provider` in the modules that build
providers. User-facing backends stay exactly `claude | groq | openrouter`.

Why it exists: the regression gate must be HERMETIC and REPEATABLE — same input,
same output, no network, no API key, no cost. A real LLM phrases setup steps,
overviews, etc. differently every run, so exact-match eval checks can only be
satisfied by a deterministic generator. This provider produces a grounded
briefing purely from the gathered repo context: it only cites sources that
exist, says "not found in repo" when data is absent, and strips embedded
instructions (prompt-injection defense), so it passes the grounding checks by
construction.
"""
from __future__ import annotations

import json
import re

from onboarding_brain.prompts import extract_context
from onboarding_brain.providers.base import LLMProvider, LLMResult

NOT_FOUND = "not found in repo"

# config file -> deterministic run steps (must match what the eval checks assert)
_SETUP_RULES = [
    ("angular.json", ["Install dependencies: npm install", "Run locally: npx ng serve (or npm start)"]),
    ("package.json", ["Install dependencies: npm install", "Start the app: npm start (see package.json scripts)"]),
    ("requirements.txt", ["Create a virtualenv", "Install dependencies: pip install -r requirements.txt"]),
    ("pyproject.toml", ["Install the package: pip install . (or poetry install)"]),
    ("docker-compose.yml", ["Start everything: docker compose up"]),
    ("Dockerfile", ["Build the image: docker build -t app .", "Run it: docker run app"]),
    ("Makefile", ["Run the documented make target: make (see Makefile)"]),
    ("go.mod", ["Build: go build ./...", "Run: go run ."]),
    ("Cargo.toml", ["Build & run: cargo run"]),
    ("pom.xml", ["Build: mvn install", "Run: mvn spring-boot:run (or per pom.xml)"]),
]

_FRAMEWORK_HINTS = {
    "angular.json": ("Angular", "A web UI framework; the app's frontend is built with it."),
    "fastapi": ("FastAPI", "A Python web framework used to expose HTTP APIs."),
    "django": ("Django", "A Python web framework."),
    "react": ("React", "A JavaScript UI library."),
    "ngrx": ("NgRx", "Redux-style state management for Angular."),
    "uvicorn": ("Uvicorn", "The server that runs the Python web app."),
    "pytest": ("pytest", "The test runner used in this repo."),
}

_INJECTION = re.compile(
    r"ignore\b[\w\s]{0,30}instructions|disregard|output\b[\w\s]{0,20}secret|"
    r"reply only|system\s*:|jailbreak|prompt\b[\w\s]{0,15}(ignore|reveal)",
    re.IGNORECASE,
)


def _first_sentences(text: str, n: int = 2) -> str:
    clean = re.sub(r"[#>*`]|\!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)", " ", text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    parts = re.split(r"(?<=[.!?])\s+", clean)
    # drop sentences that look like embedded instructions (prompt-injection defense)
    parts = [p for p in parts if not _INJECTION.search(p)]
    return " ".join(parts[:n]).strip()[:600]


def _stub_briefing(ctx: dict) -> dict:
    readme = ctx.get("readme") or {}
    configs = ctx.get("config_files") or []
    config_names = {c.get("file") for c in configs}
    config_blob = " ".join((c.get("content") or "").lower() for c in configs)
    git_log = ctx.get("git_log") or []
    ownership = ctx.get("ownership") or []
    top_dirs = ctx.get("top_level_dirs") or []

    if readme.get("content"):
        overview = {"answer": _first_sentences(readme["content"]), "sources": [readme["file"]]}
    else:
        overview = {"answer": NOT_FOUND, "sources": []}

    folder_map = [
        {"folder": d, "purpose": "Top-level directory (purpose not documented in repo).",
         "sources": ["file tree"]}
        for d in top_dirs[:12]
    ]

    steps, srcs = [], []
    for fname, fsteps in _SETUP_RULES:
        if fname in config_names:
            steps.extend(fsteps)
            srcs.append(fname)
    setup_steps = {"steps": steps, "sources": srcs} if steps else {"steps": [NOT_FOUND], "sources": []}

    if git_log:
        subjects = [ln.split("|")[-1] for ln in git_log[:8]]
        recent_work = {"answer": "Recent commits: " + "; ".join(subjects), "sources": ["git log"]}
    else:
        recent_work = {"answer": NOT_FOUND, "sources": []}

    owners = [
        {"area": o["area"], "owner": ", ".join(o.get("top_authors", [])) or NOT_FOUND,
         "sources": ["git history"]}
        for o in ownership
    ]

    glossary = []
    for key, (term, meaning) in _FRAMEWORK_HINTS.items():
        if key in config_names or key in config_blob:
            src = key if key in config_names else (readme.get("file") or "config_files")
            glossary.append({"term": term, "meaning": meaning,
                             "sources": [src if src in (ctx.get("available_sources") or []) else "file tree"]})

    key_features = [
        {"feature": d.rstrip("/").split("/")[-1],
         "detail": "Feature area inferred from the folder name.", "sources": ["file tree"]}
        for d in (ctx.get("dir_map") or []) if d.count("/") == 2
    ][:8]

    return {
        "overview": overview,
        "key_features": key_features,
        "folder_map": folder_map,
        "setup_steps": setup_steps,
        "recent_work": recent_work,
        "owners": owners,
        "glossary": glossary,
    }


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
    takeaways; condense -> standalone query echo; else -> briefing JSON. Has no
    `complete_turn`, so chat routes to the RAG pipeline (deterministic) rather
    than the agentic loop."""

    @property
    def name(self) -> str:
        return "stub/deterministic-v1"

    def _complete(self, system: str, user: str) -> LLMResult:
        if "CONTEXT (retrieved snippets)" in user:
            payload = _stub_chat(user)
        elif "Walkthrough section" in user:
            payload = _stub_walkthrough(user)
        else:
            # Briefing shape. For the condense prompt this carries no "question"
            # key, so chat.py falls back to its deterministic offline condenser
            # (which folds history in) — exactly the proven behavior we want.
            payload = _stub_briefing(extract_context(user))
        return LLMResult(text=json.dumps(payload, ensure_ascii=False), model=self.name)


def install_stub(settings, *, patch_source: bool = True) -> StubProvider:
    """Monkeypatch `get_provider` in the agent-flow modules so the whole system
    uses the deterministic StubProvider. Returns the instance. EVAL/TEST USE
    ONLY — this keeps the stub out of the user-facing backend dispatch entirely.

    patch_source=False leaves `onboarding_brain.providers.get_provider` itself
    untouched, so tests that call it directly to assert real provider selection
    (FallbackProvider / OpenRouterProvider) still see the real factory, while the
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
