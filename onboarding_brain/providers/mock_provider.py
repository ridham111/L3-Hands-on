"""Deterministic, offline provider.

Produces a grounded onboarding briefing purely from the gathered repo context —
no API key, no network. By construction it only cites sources that exist and
says "not found in repo" when data is absent, so it always passes the
source-grounding check. Ideal as the eval/regression baseline.
"""
from __future__ import annotations

import json
import re

from ..config import Settings
from ..prompts import extract_context
from .base import LLMProvider, LLMResult

NOT_FOUND = "not found in repo"

# config file -> (steps, source)
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
    # strip markdown headings/badges/links
    clean = re.sub(r"[#>*`]|\!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)", " ", text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    parts = re.split(r"(?<=[.!?])\s+", clean)
    # drop sentences that look like embedded instructions (prompt-injection defense)
    parts = [p for p in parts if not _INJECTION.search(p)]
    out = " ".join(parts[:n]).strip()
    return out[:600]


def _mock_briefing(ctx: dict) -> dict:
    readme = ctx.get("readme") or {}
    configs = ctx.get("config_files") or []
    config_names = {c.get("file") for c in configs}
    config_blob = " ".join((c.get("content") or "").lower() for c in configs)
    git_log = ctx.get("git_log") or []
    ownership = ctx.get("ownership") or []
    top_dirs = ctx.get("top_level_dirs") or []

    # 1. overview
    if readme.get("content"):
        overview = {"answer": _first_sentences(readme["content"]), "sources": [readme["file"]]}
    else:
        overview = {"answer": NOT_FOUND, "sources": []}

    # 2. folder map
    folder_map = [
        {"folder": d, "purpose": "Top-level directory (purpose not documented in repo).",
         "sources": ["file tree"]}
        for d in top_dirs[:12]
    ]

    # 3. setup steps
    steps, srcs = [], []
    for fname, fsteps in _SETUP_RULES:
        if fname in config_names:
            steps.extend(fsteps)
            srcs.append(fname)
    setup_steps = {"steps": steps, "sources": srcs} if steps else {"steps": [NOT_FOUND], "sources": []}

    # 4. recent work
    if git_log:
        subjects = [ln.split("|")[-1] for ln in git_log[:8]]
        recent_work = {"answer": "Recent commits: " + "; ".join(subjects), "sources": ["git log"]}
    else:
        recent_work = {"answer": NOT_FOUND, "sources": []}

    # 5. owners
    owners = [
        {"area": o["area"], "owner": ", ".join(o.get("top_authors", [])) or NOT_FOUND, "sources": ["git history"]}
        for o in ownership
    ]

    # 6. glossary
    glossary = []
    for key, (term, meaning) in _FRAMEWORK_HINTS.items():
        if key in config_names or key in config_blob:
            src = key if key in config_names else (readme.get("file") or "config_files")
            glossary.append({"term": term, "meaning": meaning, "sources": [src if src in (ctx.get("available_sources") or []) else "file tree"]})

    # key features: offline approximation from second-level directory names
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


class MockProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "mock/deterministic-v1"

    def _complete(self, system: str, user: str) -> LLMResult:
        ctx = extract_context(user)
        return LLMResult(text=json.dumps(_mock_briefing(ctx), ensure_ascii=False), model=self.name)
