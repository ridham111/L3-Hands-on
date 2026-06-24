"""Agent registry — the sub-agent system.

The same codebase brain (index + retrieval + grounding + provider layer) powers
many specialized agents. Each agent is a small spec: an id, metadata, and a
`run` handler. Add a new agent = add one entry here. The API lists the registry
and dispatches to the matching handler, so new agents need no new endpoints.

Today: the onboarding/KT briefing and an installation guide. The pattern scales
to code-review, bug-triage, test-writing, etc. — each reusing the shared
repo intelligence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import Settings, get_settings
from .contract import OnboardingRequest
from .onboarding import generate_briefing


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    category: str
    description: str
    run: Callable[[dict, Settings], dict]


# --- prerequisites a newcomer needs, inferred from the stack the briefing found.
# Deterministic + offline: maps a detected framework/tool to its real toolchain.
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


def _installation_run(req: dict, settings: Settings) -> dict:
    """Reuse the grounded briefing (setup steps come straight from real config
    files), then add the toolchain prerequisites a newcomer must install first —
    the gap a bare 'npm install' list leaves out."""
    brief = generate_briefing(OnboardingRequest(repo_path=req["repo_path"]), settings=settings)
    steps = brief.setup_steps.steps
    sources = brief.setup_steps.sources
    # scan the briefing's own grounded text for stack signals
    hay = " ".join([brief.overview.answer, " ".join(steps),
                    " ".join(g.term + " " + g.meaning for g in brief.glossary)]).lower()
    seen, prereq = set(), []
    for pat, need in _PREREQ:
        if re.search(pat, hay) and need not in seen:
            seen.add(need)
            prereq.append(need)
    if not prereq:
        prereq = ["Check the README/config files for the required runtime and version."]
    return {
        "agent_id": "installation-guide",
        "name": "Installation Guide",
        "repo_path": brief.trace.repo_path,
        "prerequisites": prereq,
        "setup_steps": steps or ["not found in repo"],
        "sources": sources,
        "overview": brief.overview.model_dump(),
        "validation_status": brief.validation_status,
        "trace": brief.trace.model_dump(),
    }


def _onboarding_run(req: dict, settings: Settings) -> dict:
    return generate_briefing(
        OnboardingRequest(repo_path=req["repo_path"]), settings=settings
    ).model_dump(mode="json")


AGENTS: dict[str, AgentSpec] = {
    s.id: s for s in [
        AgentSpec("onboarding-brain", "Onboarding Brain", "Enablement",
                  "Grounded Day-1 briefing for a repo: what it does, layout, run steps, owners.",
                  _onboarding_run),
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
