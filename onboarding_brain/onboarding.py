"""Repo-access guard shared by the ingest path and the API.

`ONBOARDING_ALLOWED_ROOTS` fences which directories may be read; `_check_allowed`
enforces it before any repo is cloned/indexed, and `RepoAccessError` is raised on
a violation so callers can return a clean 4xx.
"""
from __future__ import annotations

from pathlib import Path

from .config import Settings


class RepoAccessError(ValueError):
    pass


def _check_allowed(repo_path: str, settings: Settings) -> None:
    if not settings.allowed_roots:
        return
    resolved = Path(repo_path).expanduser().resolve()
    for root in settings.allowed_roots:
        try:
            resolved.relative_to(Path(root).expanduser().resolve())
            return
        except ValueError:
            continue
    raise RepoAccessError(f"repo_path is outside ONBOARDING_ALLOWED_ROOTS: {resolved}")
