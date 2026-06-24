"""Repository introspection — the Onboarding Brain's tools.

Gathers grounded facts from a local repo (no LLM): file tree, README, config
files, recent git history, and per-area ownership from git. Everything returned
is a real artifact the agent can cite; the agent is forbidden to claim anything
not present here. Bounded: skips heavy dirs, caps tree size and per-file bytes,
times out git calls — safe and fast on arbitrary local paths.
"""
from __future__ import annotations

import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Optional

_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build",
    ".next", ".angular", "coverage", ".idea", ".vscode", "target", ".pytest_cache",
    ".mypy_cache", "out", "bin", "obj", ".gradle", "vendor",
}
_README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md")
_CONFIG_NAMES = (
    "package.json", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml", "Makefile",
    ".env.example", "angular.json", "pom.xml", "build.gradle", "go.mod",
    "Cargo.toml", "Gemfile", "composer.json", "tsconfig.json", "vite.config.ts",
    "next.config.js", "tox.ini",
)

_MAX_TREE_ENTRIES = 400   # lines shown in the file tree (LLM context budget)
_MAX_FILES = 5000         # citation universe — files the agent may cite
_MAX_FILE_BYTES = 6000
_GIT_TIMEOUT = 15

# KT Brain installs — exclude from briefings when one sits inside the repo
# being briefed (it's tooling, not the team's project). Detected by marker
# (any copy, anywhere) plus this install's own path.
_SELF_ROOT = Path(__file__).resolve().parents[1]


def _is_tool_install(p: Path) -> bool:
    return (p / "onboarding_brain" / "__init__.py").is_file()


def _git(repo: Path, args: list[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _read_text(path: Path) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return data[:_MAX_FILE_BYTES] + ("\n…[truncated]" if len(data) > _MAX_FILE_BYTES else "")


def _build_tree(repo: Path) -> tuple[str, list[str], list[str], list[str]]:
    """The displayed tree is capped at _MAX_TREE_ENTRIES lines, but `files`
    (the citation universe) keeps collecting up to _MAX_FILES so real files in
    large repos aren't flagged as invented citations. `dirs` is a compact
    directory-only map (depth <= 3) — module/feature names survive truncation
    even when the file tree doesn't."""
    lines: list[str] = []
    files: list[str] = []
    dirs: list[str] = []
    top_dirs: list[str] = []
    shown = 0
    for current, dirnames, filenames in os.walk(repo):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
            and (Path(current) / d).resolve() != _SELF_ROOT
            and not _is_tool_install(Path(current) / d)
        )
        rel = Path(current).relative_to(repo)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth == 0:
            top_dirs = list(dirnames)
        elif depth <= 3 and len(dirs) < 400:
            dirs.append(str(rel).replace("\\", "/") + "/")
        display = depth <= 3 and shown < _MAX_TREE_ENTRIES
        indent = "  " * depth
        if str(rel) != "." and display:
            lines.append(f"{indent}{rel.name}/")
        for fn in sorted(filenames):
            relfile = (str(rel / fn) if str(rel) != "." else fn).replace("\\", "/")
            files.append(relfile)
            if display and shown < _MAX_TREE_ENTRIES:
                lines.append(f"{indent}  {fn}")
                shown += 1
                if shown == _MAX_TREE_ENTRIES:
                    lines.append("  …[tree truncated]")
            if len(files) >= _MAX_FILES:
                return "\n".join(lines), top_dirs, files, dirs
    return "\n".join(lines), top_dirs, files, dirs


def _ownership(repo: Path, top_dirs: list[str]) -> list[dict[str, Any]]:
    owners: list[dict[str, Any]] = []
    for d in top_dirs[:8]:
        log = _git(repo, ["log", "--no-merges", "--pretty=%an", "-n", "120", "--", d])
        if not log:
            continue
        authors = [a.strip() for a in log.splitlines() if a.strip()]
        if not authors:
            continue
        top = Counter(authors).most_common(2)
        owners.append({"area": d, "top_authors": [f"{name} ({n})" for name, n in top]})
    return owners


def gather_repo_context(repo_path: str) -> dict[str, Any]:
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        return {"error": f"not a directory: {repo}", "repo_path": str(repo)}

    tree, top_dirs, files, dir_map = _build_tree(repo)

    readme = None
    for name in _README_NAMES:
        p = repo / name
        if p.is_file():
            readme = {"file": name, "content": _read_text(p)}
            break

    configs = []
    for name in _CONFIG_NAMES:
        p = repo / name
        if p.is_file():
            configs.append({"file": name, "content": _read_text(p)})

    is_git = (repo / ".git").exists() or _git(repo, ["rev-parse", "--is-inside-work-tree"]) is not None
    git_log_raw = _git(repo, ["log", "-n", "30", "--no-merges", "--date=short",
                              "--pretty=format:%h|%an|%ad|%s"]) if is_git else None
    git_log = [ln for ln in (git_log_raw or "").splitlines() if ln.strip()]
    ownership = _ownership(repo, top_dirs) if is_git else []

    available = sorted(set(files)
                       | ({readme["file"]} if readme else set())
                       | {c["file"] for c in configs})
    if is_git:
        available += ["git log", "git history", "git blame"]
    available += ["file tree"]

    return {
        "repo_path": str(repo),
        "is_git_repo": is_git,
        "file_tree": tree,
        "dir_map": dir_map,
        "top_level_dirs": top_dirs,
        "readme": readme,
        "config_files": configs,
        "git_log": git_log,
        "ownership": ownership,
        "available_sources": available,
        "file_count": len(files),
    }
