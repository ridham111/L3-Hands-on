"""Citation grounding for Onboarding Brain.

The trust rule: every source the agent cites must be a real artifact in the
repo (a scanned file, a known config/README, a directory, or an allowed meta
source like "git log"/"file tree"). Citations that don't resolve are flagged —
this catches an LLM inventing a file name to look authoritative.
"""
from __future__ import annotations

from typing import Any

_META_SOURCES = {"git log", "git history", "git blame", "file tree", "config_files", "not found in repo"}


def _resolves(source: str, available: set[str], files_lower: list[str]) -> bool:
    s = source.strip()
    if not s or s.lower() in _META_SOURCES:
        return True
    if s in available:
        return True
    sl = s.lower().rstrip("/")
    # directory citation: matches a prefix of any scanned file path
    return any(f == sl or f.startswith(sl + "/") for f in files_lower)


def check_sources(response_dict: dict, available_sources: list[str]) -> dict[str, Any]:
    available = set(available_sources or [])
    files_lower = [a.lower() for a in available]

    cited: list[str] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "sources" and isinstance(v, list):
                    cited.extend(str(x) for x in v)
                else:
                    collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)

    collect(response_dict)

    unresolved = sorted({c for c in cited if not _resolves(c, available, files_lower)})
    total = len(cited)
    status = "passed"
    if unresolved:
        status = "warning"  # cited a source we can't find — surfaced, not fatal
    coverage = round(1 - len(unresolved) / total, 2) if total else 1.0
    return {
        "validation_status": status,
        "cited_sources": total,
        "unresolved_sources": unresolved,
        "coverage": coverage,
    }
