"""Agent registry — the sub-agent system.

The same codebase brain (index + retrieval + grounding + provider layer) powers
specialized agents. Each agent is a small spec: an id, metadata, and a `run`
handler. The API lists the registry and dispatches to the matching handler, so new
agents need no new endpoints.

Today: an installation guide. The pattern scales to code-review, bug-triage, etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import Settings, get_settings
from .onboarding import _check_allowed
from .repo_reader import gather_repo_context

# Deterministic setup steps keyed by the config file present in the repo.
_SETUP_RULES = [
    ("angular.json", ["Install dependencies: npm install", "Run locally: npx ng serve (or npm start)"]),
    ("package.json", ["Install dependencies: npm install", "Start the app: npm start (see package.json scripts)"]),
    ("requirements.txt", ["Create a virtualenv", "Install dependencies: pip install -r requirements.txt"]),
    ("pyproject.toml", ["Install the package: pip install . (or poetry install)"]),
    ("docker-compose.yml", ["Start everything: docker compose up"]),
    ("dockerfile", ["Build the image: docker build -t app .", "Run it: docker run app"]),
    ("makefile", ["Run the documented make target: make (see Makefile)"]),
    ("go.mod", ["Build: go build ./...", "Run: go run ."]),
    ("cargo.toml", ["Build & run: cargo run"]),
    ("pom.xml", ["Build: mvn install", "Run: mvn spring-boot:run (or per pom.xml)"]),
]

# Toolchain prerequisites a newcomer must install first, by detected stack signal.
_PREREQ = [
    (r"\bangular\b", "Node.js 18 LTS and npm (Angular CLI: `npm i -g @angular/cli`)"),
    (r"\breact|next\b", "Node.js 18+ and npm or yarn"),
    (r"\bvue\b", "Node.js 18+ and npm"),
    (r"\bfastapi|uvicorn|flask|django\b", "Python 3.9+ with pip and a virtualenv"),
    (r"\bspring|maven|gradle\b", "JDK 17+ and Maven/Gradle"),
    (r"\bgo\b|go\.mod", "Go 1.20+"),
    (r"\bcargo|rust\b", "Rust toolchain (rustup)"),
    (r"\bdocker|compose\b", "Docker Desktop (or Docker Engine + compose)"),
    (r"\bpostgres|mysql|mongo|redis\b", "the database server it connects to, running locally or in Docker"),
]


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    category: str
    description: str
    run: Callable[[dict, Settings], dict]


def _installation_run(req: dict, settings: Settings) -> dict:
    """Setup steps + toolchain prerequisites, derived directly from the repo's own
    config files (package.json, requirements.txt, Dockerfile, …) — no LLM call."""
    repo_path = req["repo_path"]
    _check_allowed(repo_path, settings)
    ctx = gather_repo_context(repo_path)
    if ctx.get("error"):
        raise ValueError(ctx["error"])

    configs = ctx.get("config_files") or []
    config_names = {(c.get("file") or "").rsplit("/", 1)[-1].lower() for c in configs}
    readme = (ctx.get("readme") or {}).get("content", "") or ""
    # scan config contents + README + the config FILE NAMES (an empty angular.json
    # still signals Angular) for stack signals
    blob = (" ".join((c.get("content") or "") for c in configs) + " " + readme
            + " " + " ".join(config_names)).lower()

    steps, sources = [], []
    for fname, fsteps in _SETUP_RULES:
        if fname in config_names:
            steps.extend(fsteps)
            sources.append(fname)

    seen, prereq = set(), []
    for pat, need in _PREREQ:
        if re.search(pat, blob) and need not in seen:
            seen.add(need)
            prereq.append(need)
    if not prereq:
        prereq = ["Check the README/config files for the required runtime and version."]

    return {
        "agent_id": "installation-guide",
        "name": "Installation Guide",
        "repo_path": ctx.get("repo_path", repo_path),
        "prerequisites": prereq,
        "setup_steps": steps or ["not found in repo"],
        "sources": sources,
        "validation_status": "passed",
    }


AGENTS: dict[str, AgentSpec] = {
    s.id: s for s in [
        AgentSpec("installation-guide", "Installation Guide", "Enablement",
                  "Exact local-setup steps from the real config files, plus the toolchain "
                  "prerequisites (runtimes + versions) you must install first.",
                  _installation_run),
    ]
}


def list_agents(settings: Optional[Settings] = None) -> list[dict[str, Any]]:
    s = settings or get_settings()
    return [{"agent_id": a.id, "name": a.name, "category": a.category,
             "description": a.description, "model_used": s.model_used} for a in AGENTS.values()]


def run_agent(agent_id: str, req: dict, *, settings: Optional[Settings] = None) -> dict:
    spec = AGENTS.get(agent_id)
    if spec is None:
        raise KeyError(agent_id)
    return spec.run(req, settings or get_settings())
