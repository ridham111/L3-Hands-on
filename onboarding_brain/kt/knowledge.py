"""Knowledge elicitation — close the naming-vs-meaning gap by ASKING the human.

Code alone can't reveal intent that was never written down. After ingest we
find the files the system is least able to explain on its own — central
(referenced/large) yet cryptically named and undocumented — and turn them into
clarifying questions ("What is X for?"). The user's answers are stored as
annotations, indexed as high-priority retrievable chunks, and injected into
future answers. Tribal knowledge captured once, reused forever.

All offline: gap detection is heuristic (no LLM); answers are plain text.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

from ..config import Settings, get_settings
from .store import get_store, slugify

# words that signal a file is documented/clear enough to skip
_DOC_HINT = re.compile(r'"""|/\*\*|^\s*//|^\s*#|@description|@purpose', re.MULTILINE)
# generic/boilerplate names not worth asking about
_GENERIC = re.compile(r"^(index|main|app|utils?|helpers?|constants?|types?|models?|"
                      r"config|setup|styles?|test|spec)$", re.IGNORECASE)
# business logic lives in code, not styles/markup/data/i18n — only ask about these
_LOGIC_EXT = {".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".java", ".rb", ".rs",
              ".cs", ".kt", ".scala", ".php", ".swift", ".c", ".cpp", ".vue"}


def _is_logic_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    if any(name.endswith(s) for s in (".spec.ts", ".test.ts", ".spec.js", ".test.js",
                                      ".module.ts", ".enum.ts", ".d.ts")):
        return False  # tests/modules/enums/type-defs aren't business logic to explain
    import os
    return os.path.splitext(name)[1] in _LOGIC_EXT


def _annot_path(store, namespace: str):
    return store.ns_dir(slugify(namespace)) / "annotations.json"


def load_annotations(namespace: str, *, settings: Optional[Settings] = None) -> list[dict]:
    store = get_store(settings or get_settings())
    p = _annot_path(store, namespace)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []
    except (OSError, json.JSONDecodeError):
        return []


def save_annotation(namespace: str, file: str, answer: str, *,
                    symbol: str = "", settings: Optional[Settings] = None) -> list[dict]:
    """Record (or update) the human's explanation of a file/symbol."""
    store = get_store(settings or get_settings())
    items = load_annotations(namespace, settings=settings)
    answer = (answer or "").strip()
    items = [a for a in items if not (a.get("file") == file and a.get("symbol", "") == symbol)]
    if answer:
        items.append({"file": file, "symbol": symbol, "answer": answer[:1500], "ts": time.time()})
    p = _annot_path(store, namespace)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return items


def _cryptic(name: str) -> bool:
    """A filename whose purpose isn't obvious from the name."""
    stem = name.rsplit("/", 1)[-1].split(".")[0]
    words = re.split(r"[-_]|(?<=[a-z])(?=[A-Z])", stem)
    words = [w for w in words if w]
    if not words or _GENERIC.match(stem):
        return False
    # cryptic = has an abbreviation-like token (short, no vowels) or is very terse
    return any(len(w) <= 4 and not re.search(r"[aeiou]", w, re.I) for w in words) or len(stem) <= 5


def detect_gaps(namespace: str, *, limit: int = 8, settings: Optional[Settings] = None) -> list[dict]:
    """Rank files the system can least explain on its own and frame each as a
    clarifying question. Central (many chunks / widely imported) + cryptic +
    undocumented score highest. Already-annotated files are skipped."""
    store = get_store(settings or get_settings())
    ns = slugify(namespace)
    docs = store._load_chunk_docs(ns)
    if not docs:
        return []
    answered = {a["file"] for a in load_annotations(namespace, settings=settings)}

    # centrality: how many chunks per file, and how often the file's stem is
    # imported/referenced across the corpus
    by_file: dict[str, list[dict]] = {}
    for c in docs:
        by_file.setdefault(c.get("metadata", {}).get("path", ""), []).append(c)
    blob = "\n".join(c.get("text", "") for c in docs[:4000])

    scored = []
    for path, chunks in by_file.items():
        if not path or path in answered or path == "project-briefing":
            continue
        if not _is_logic_file(path):
            continue
        name = path.rsplit("/", 1)[-1]
        stem = name.split(".")[0]
        documented = any(_DOC_HINT.search(c.get("text", "")) for c in chunks)
        refs = blob.count(stem)  # rough usage count
        score = 0.0
        if _cryptic(name):
            score += 3
        if not documented:
            score += 1.5
        score += min(refs, 12) * 0.4          # central files matter more
        score += min(len(chunks), 6) * 0.3    # bigger files matter more
        if score < 3:
            continue
        symbol = str(chunks[0].get("metadata", {}).get("symbol", ""))
        scored.append((score, path, symbol))

    scored.sort(reverse=True)
    out = []
    for score, path, symbol in scored[:limit]:
        name = path.rsplit("/", 1)[-1]
        out.append({
            "file": path,
            "symbol": symbol,
            "question": f"What is `{name}` for — what business problem does it solve, "
                        f"and when is it used?",
        })
    return out


# directory segments that describe scaffolding, not features
_GENERIC_DIR = {
    "src", "app", "lib", "main", "core", "common", "shared", "components", "component",
    "services", "service", "models", "model", "utils", "util", "helpers", "helper",
    "assets", "public", "static", "styles", "scss", "css", "test", "tests", "spec",
    "e2e", "dist", "build", "node_modules", "vendor", "config", "constants", "types",
    "interfaces", "enums", "pipes", "directives", "guards", "interceptors", "modules",
    "pages", "views", "containers", "features", "modules", "api", "store", "state",
}


def _humanize(seg: str) -> str:
    return re.sub(r"[-_]+", " ", seg).strip()


def feature_surface(namespace: str, *, limit: int = 40, settings: Optional[Settings] = None) -> Optional[dict]:
    """A comprehensive, real feature map built from the directory structure —
    every meaningful folder is a feature area. Injected as CONTEXT for broad
    "what does this do / main features" questions so the model enumerates the
    full surface (rank-by-store, asset-review, pnc, …) instead of leaning on a
    short briefing and missing half of them. Real repo data, not a canned answer."""
    store = get_store(settings or get_settings())
    paths = store.known_paths(slugify(namespace))
    if not paths:
        return None
    seg_files: dict[str, set] = {}
    for p in paths:
        for seg in p.split("/")[:-1]:        # directory segments only
            s = seg.strip()
            low = s.lower()
            if not s or low in _GENERIC_DIR or len(s) < 3 or s.startswith("."):
                continue
            seg_files.setdefault(s, set()).add(p)
    if not seg_files:
        return None
    # distinctive feature folders rank by how much code they contain
    ranked = sorted(seg_files.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    lines = [f"- {_humanize(seg)} ({len(files)} files)" for seg, files in ranked[:limit]]
    text = ("FEATURE MAP — directories in this codebase, each a feature area "
            "(use these to enumerate the project's real features):\n" + "\n".join(lines))
    return {
        "id": "feature-map#0", "score": 1.0, "text": text,
        "metadata": {"path": "feature-map", "language": "", "symbol": "feature map",
                     "line_start": 0, "line_end": 0},
    }


def annotation_chunks(namespace: str, *, settings: Optional[Settings] = None) -> list[dict]:
    """Stored answers, shaped as retrievable chunks so they surface in search and
    get injected into answers (the curated glossary)."""
    out = []
    for a in load_annotations(namespace, settings=settings):
        label = a.get("symbol") or a.get("file", "")
        text = (f"TEAM KNOWLEDGE about {a.get('file','')}"
                + (f" ({a['symbol']})" if a.get("symbol") else "")
                + f":\n{a.get('answer','')}")
        out.append({
            "id": f"annotation::{a.get('file','')}::{a.get('symbol','')}",
            "score": 1.0, "text": text,
            "index_text": f"{a.get('file','')} {label}\n{a.get('answer','')}",
            "metadata": {"path": a.get("file", ""), "language": "",
                         "symbol": "team note: " + label, "line_start": 0, "line_end": 0},
        })
    return out
