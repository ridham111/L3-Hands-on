"""Walk a repo and split text/code files into retrievable chunks.

Structure-aware: chunks open at function/class/heading boundaries whenever
possible, so a retrieved chunk is a coherent unit (a whole function, a whole
README section) and citations land on real symbols instead of arbitrary
character windows. Oversized units fall back to line-aligned windows with
overlap. Bounded: skips heavy/binary dirs and files, caps file size and count.

Each chunk carries:
  text       — what the user/LLM sees (verbatim file content)
  index_text — what gets embedded/vectorized (path + text, so file and folder
               names contribute to retrieval: "payments" finds payments.py)
  metadata   — path, language, symbol (enclosing def/class/heading), line range
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterator

_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build",
    ".next", ".angular", "coverage", ".idea", ".vscode", "target", ".pytest_cache",
    ".mypy_cache", "out", "bin", "obj", ".gradle", "vendor", ".kt_index",
}
_TEXT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs", ".c", ".h",
    ".cpp", ".cs", ".php", ".scala", ".kt", ".swift", ".sh", ".sql", ".html", ".css",
    ".scss", ".vue", ".md", ".rst", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini",
    ".cfg", ".xml", ".gradle", ".dockerfile", ".env",
}
_ALWAYS = {"Dockerfile", "Makefile", "README", "requirements.txt", "package.json", ".env.example"}
# Generated / noise files that pollute retrieval — never index these.
_SKIP_FILES = {"trace.json", "results.json", "eval_trace.jsonl", "package-lock.json",
               "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock"}
_SKIP_SUFFIX = (".min.js", ".min.css", ".map", ".lock", ".log")
_LANG = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript/React",
    ".java": "Java", ".go": "Go", ".rb": "Ruby", ".rs": "Rust", ".cs": "C#",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".sql": "SQL", ".md": "Markdown",
    ".json": "JSON", ".yml": "YAML", ".yaml": "YAML",
}

# --- boundary detection: lines that START a new logical unit ----------------
_PY_BOUNDARY = re.compile(r"^\s{0,8}(@\w|async\s+def\s|def\s|class\s)")
_C_LIKE_BOUNDARY = re.compile(
    r"^\s{0,4}(?:export\s+|default\s+|public\s+|private\s+|protected\s+|internal\s+"
    r"|static\s+|abstract\s+|final\s+|async\s+|pub\s+)*"
    r"(?:function[\s*(]|class\s|interface\s|enum\s|struct\s|trait\s|impl[\s<]|fn\s|func\s"
    r"|module\s|namespace\s|type\s+\w|(?:const|let|var)\s+\w+\s*=)"
)
_RUBY_BOUNDARY = re.compile(r"^\s{0,4}(def\s|class\s|module\s)")
_MD_BOUNDARY = re.compile(r"^#{1,6}\s")
_BOUNDARIES: dict[str, re.Pattern] = {
    ".py": _PY_BOUNDARY,
    ".rb": _RUBY_BOUNDARY,
    ".md": _MD_BOUNDARY,
}
for _ext in (".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".h", ".cpp",
             ".cs", ".php", ".scala", ".kt", ".swift", ".vue"):
    _BOUNDARIES[_ext] = _C_LIKE_BOUNDARY


def _is_text(path: Path) -> bool:
    if path.name in _SKIP_FILES or path.name.lower().endswith(_SKIP_SUFFIX):
        return False
    if path.name in _ALWAYS:
        return True
    return path.suffix.lower() in _TEXT_EXT


def _read(path: Path, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            data = path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
        else:
            data = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    if "\x00" in data[:1000]:  # crude binary guard
        return None
    return data


# KT Brain installs. When a copy of the tool lives INSIDE the repo being
# indexed (common while evaluating it), its prompts/tests/evals contain the
# exact question phrases users ask and hijack retrieval — never index it.
# Detected by marker (any copy, anywhere) plus this install's own path.
# Indexing the tool's folder directly (repo == tool root) still works.
_SELF_ROOT = Path(__file__).resolve().parents[2]


def _is_tool_install(p: Path) -> bool:
    return (p / "onboarding_brain" / "__init__.py").is_file()


def _prune(current: str, dirnames: list[str]) -> list[str]:
    keep = []
    for d in sorted(dirnames):
        if d in _SKIP_DIRS or d.startswith("."):
            continue
        sub = Path(current) / d
        if sub.resolve() == _SELF_ROOT or _is_tool_install(sub):
            continue
        keep.append(d)
    return keep


def iter_chunks(repo: Path, *, chunk_chars: int, overlap: int, max_files: int, max_bytes: int) -> Iterator[dict[str, Any]]:
    seen = 0
    for current, dirnames, filenames in os.walk(repo):
        dirnames[:] = _prune(current, dirnames)
        for fn in sorted(filenames):
            p = Path(current) / fn
            if not _is_text(p):
                continue
            if seen >= max_files:
                return
            text = _read(p, max_bytes)
            if not text or not text.strip():
                continue
            seen += 1
            rel = str(p.relative_to(repo)).replace("\\", "/")
            ext = p.suffix.lower()
            lang = _LANG.get(ext, ext.lstrip(".") or "text")
            yield from _slice(rel, lang, ext, text, chunk_chars, overlap)


# --- structure-aware slicing -------------------------------------------------
def _slice(path: str, lang: str, ext: str, text: str, chunk_chars: int, overlap: int) -> Iterator[dict[str, Any]]:
    overlap = min(overlap, chunk_chars // 3)  # guarantee forward progress
    pattern = _BOUNDARIES.get(ext)

    # (line_number, piece) items; monster lines are pre-split so no single item
    # can exceed the chunk budget.
    items: list[tuple[int, str]] = []
    for lineno, ln in enumerate(text.splitlines(keepends=True), start=1):
        while len(ln) > chunk_chars:
            items.append((lineno, ln[:chunk_chars]))
            ln = ln[chunk_chars:]
        items.append((lineno, ln))

    # Segment starts: structure boundaries when the language has them,
    # otherwise paragraph starts (first non-blank line after a blank one).
    starts = [0]
    for i in range(1, len(items)):
        ln = items[i][1]
        if pattern is not None:
            if pattern.match(ln):
                starts.append(i)
        elif ln.strip() and not items[i - 1][1].strip():
            starts.append(i)

    segments = [items[starts[i]:(starts[i + 1] if i + 1 < len(starts) else len(items))]
                for i in range(len(starts))]

    # Greedy packing: whole segments per chunk; a segment bigger than the
    # budget is windowed line-aligned with overlap.
    packed: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur and "".join(t for _, t in cur).strip():
            packed.append(cur)
        cur, cur_len = [], 0

    for seg in segments:
        seg_len = sum(len(t) for _, t in seg)
        if cur_len and cur_len + seg_len > chunk_chars:
            flush()
        if seg_len <= chunk_chars:
            cur.extend(seg)
            cur_len += seg_len
            continue
        flush()
        win: list[tuple[int, str]] = []
        wlen = 0
        for item in seg:
            win.append(item)
            wlen += len(item[1])
            if wlen >= chunk_chars:
                packed.append(win)
                tail: list[tuple[int, str]] = []
                tlen = 0
                for back in reversed(win):
                    if tlen + len(back[1]) > overlap:
                        break
                    tail.insert(0, back)
                    tlen += len(back[1])
                win, wlen = tail, tlen
        # leftover stays current so following small segments can join it
        cur, cur_len = win, wlen

    flush()

    for idx, chunk_items in enumerate(packed):
        body = "".join(t for _, t in chunk_items)
        symbol = _symbol_of(chunk_items, pattern)
        yield {
            "id": f"{path}#{idx}",
            "text": body,
            "index_text": f"{path}\n{body}",
            "metadata": {"path": path, "language": lang, "chunk_index": idx,
                         "symbol": symbol,
                         "line_start": chunk_items[0][0], "line_end": chunk_items[-1][0]},
        }


def _symbol_of(chunk_items: list[tuple[int, str]], pattern: re.Pattern | None) -> str:
    """Human-readable anchor for the chunk: its first boundary line if any,
    else its first non-blank line."""
    first_nonblank = ""
    for _, t in chunk_items:
        s = t.strip()
        if s and not first_nonblank:
            first_nonblank = s
        if pattern is not None and pattern.match(t) and not s.startswith("@"):
            return s[:100]
    return first_nonblank[:100]
