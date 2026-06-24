"""Business-context harvesting — bridge the gap between code NAMES and business
MEANING using signals that live in the repo but outside the code itself.

Two offline sources (no LLM, no network):
  1. i18n labels  — UI translation files map cryptic keys to human words
                    ("RANK_BY_STORE" -> "Rank by Store"). A file that USES a key
                    gets the human label folded into its searchable text, so a
                    question in user vocabulary finds the right component.
  2. commit log   — commit subjects say WHY code changed ("fix lasso selection
                    for multi-banner stores"). Each file's recent subjects are
                    folded into its searchable text.

Enrichment only touches `index_text` (what's embedded/searched), never `text`
(what's shown), so citations and snippets stay exact.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

_I18N_HINT = re.compile(r"(^|/)(i18n|locale[s]?|lang|translations?)(/|$)", re.IGNORECASE)
_LOCALE_FILE = re.compile(r"^(en|en[-_][a-z]{2}|messages|translation)\.json$", re.IGNORECASE)
_KEYISH = re.compile(r"[A-Za-z0-9_.]{4,}")
_GIT_TIMEOUT = 20


def _flatten(obj: Any, out: dict[str, str]) -> None:
    """Collect leaf string values from nested i18n JSON, keyed by leaf token."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                label = v.strip()
                # keep human-looking labels only (has letters, not a single word of code)
                if label and re.search(r"[A-Za-z]", label) and len(label) <= 120:
                    out[str(k)] = label
            else:
                _flatten(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten(v, out)


def build_i18n_labels(repo: Path) -> dict[str, str]:
    """Map i18n KEY -> human label, scanning translation JSON files in the repo.
    Bounded: only English/default locale files, capped total keys."""
    labels: dict[str, str] = {}
    for current, dirnames, filenames in __import__("os").walk(repo):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in
                       ("node_modules", ".git", ".venv", "dist", "build", "__pycache__")]
        cur = Path(current)
        in_i18n = bool(_I18N_HINT.search(str(cur).replace("\\", "/")))
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            if not (in_i18n or _LOCALE_FILE.match(fn)):
                continue
            try:
                data = json.loads((cur / fn).read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue
            _flatten(data, labels)
            if len(labels) > 5000:
                return labels
    return labels


def build_commit_map(repo: Path, *, max_commits: int = 400, per_file: int = 3) -> dict[str, list[str]]:
    """Map repo-relative path -> recent commit subjects (the 'why'). One git call."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", "-n", str(max_commits), "--no-merges",
             "--pretty=format:@@%s", "--name-only"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
        if out.returncode != 0:
            return {}
    except (OSError, subprocess.SubprocessError):
        return {}

    commits: dict[str, list[str]] = {}
    subject = ""
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            subject = line[2:].strip()
        elif line.strip() and subject:
            path = line.strip().replace("\\", "/")
            lst = commits.setdefault(path, [])
            if subject not in lst and len(lst) < per_file:
                lst.append(subject)
    return commits


def build_commit_chunks(repo: Path, *, max_commits: int = 300) -> list[dict]:
    """Index each git commit as a standalone searchable chunk.

    Unlike build_commit_map (which folds commit subjects into code chunks),
    this makes the commits themselves queryable — so Cortex can answer
    "what changed in auth last week?" or "what did Alice commit?" directly.
    Each chunk's text shows the commit in human-readable form.
    Returns [] if repo has no git history or git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log",
             f"-n{max_commits}", "--no-merges",
             "--pretty=format:@@SPLIT@@%H|%ai|%an|%ae|%s", "--name-only"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return []
    except (OSError, subprocess.SubprocessError):
        return []

    # Parse: @@SPLIT@@ hash|date|author_name|author_email|subject
    # followed by lines of filenames changed
    entries: list[dict] = []
    current: dict | None = None
    for line in out.stdout.splitlines():
        if line.startswith("@@SPLIT@@"):
            if current:
                entries.append(current)
            parts = line[9:].split("|", 4)
            if len(parts) < 5:
                continue
            h, date, name, email, subject = parts
            current = {
                "hash": h[:8], "date": date[:10], "author": name.strip(),
                "email": email.strip(), "subject": subject.strip(), "files": [],
            }
        elif line.strip() and current:
            current["files"].append(line.strip().replace("\\", "/"))
    if current:
        entries.append(current)

    chunks: list[dict] = []
    for i, c in enumerate(entries):
        files_preview = ", ".join(c["files"][:8])
        if len(c["files"]) > 8:
            files_preview += f" (+{len(c['files']) - 8} more)"
        # human-readable text shown as a source snippet in the UI
        text = (
            f"[COMMIT {c['hash']}] {c['date']} · {c['author']} <{c['email']}>\n"
            f"Message: {c['subject']}\n"
            f"Files changed ({len(c['files'])}): {files_preview or '—'}"
        )
        # index_text expands camelCase identifiers for better matching
        index_extra = " ".join(c["files"])  # all paths searchable
        chunks.append({
            # STABLE key per commit (was 'git-history#{i}', which renumbered on
            # every new commit and forced all ~N commit chunks to re-embed on
            # re-sync). Hash-keyed → a new commit adds ONE chunk, the rest reuse.
            "id": f"git-history#{c['hash']}",
            "text": text,
            "index_text": text + "\n" + index_extra,
            "metadata": {
                "path": "git-history",
                "language": "git",
                "line_start": i,
                "line_end": i,
                "symbol": f"{c['hash']} — {c['subject'][:60]}",
                "chunk_index": i,
            },
        })
    return chunks


def enrich_chunks(chunks: list[dict], labels: dict[str, str],
                  commits: Optional[dict] = None) -> int:
    """Fold matched i18n labels into each chunk's index_text. Returns how many
    chunks were enriched.

    NOTE: commit subjects are deliberately NOT folded into code chunks anymore.
    Doing so changed a chunk's index_text every time its file was touched by a
    recent commit, which invalidated the dense-embedding reuse cache and forced
    a full re-embed on every re-sync. Commit history is still fully searchable
    via the dedicated git-history chunks (build_commit_chunks). The `commits`
    arg is kept for backward compatibility but ignored. i18n labels are stable,
    so they stay."""
    enriched = 0
    keyset = set(labels)
    if not keyset:
        return 0
    for c in chunks:
        # which i18n keys does this chunk reference? (token-level, bounded)
        toks = set(_KEYISH.findall(c.get("text", "")))
        hit = [labels[k] for k in (toks & keyset)]
        if hit:
            c["index_text"] = (c.get("index_text") or c["text"]) + "\n" + \
                "UI labels: " + "; ".join(sorted(set(hit))[:8])
            enriched += 1
    return enriched
