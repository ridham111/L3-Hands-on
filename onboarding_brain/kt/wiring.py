"""Feature wiring — derive how the files behind an answer connect, so the UI can
DRAW the feature as an ordered flow instead of only describing it.

Given a set of source files, we parse their import/require statements, build the
import graph, pick the MAIN/entry file with language-aware signals (works for any
stack, not just front-end), and order the files into a top-to-bottom FLOW
(main → what it imports → next …). Each file is tagged with a role for colour.
Pure static analysis — no LLM, offline.
"""
from __future__ import annotations

import re
from collections import Counter, deque
from typing import Optional

from ..config import Settings, get_settings
from .store import get_store, slugify

# import/require targets — TS/JS, Python, Java/Kotlin, Go-ish
_IMPORT = re.compile(
    r"""(?:import\s+.*?from\s+['"]([^'"]+)['"])      # JS/TS: import x from 'y'
        |(?:require\(\s*['"]([^'"]+)['"]\s*\))         # JS: require('y')
        |(?:from\s+([\w.]+)\s+import\b)                # py: from a.b import
        |(?:^\s*import\s+([\w.]+))                      # py/java: import a.b
    """,
    re.VERBOSE | re.MULTILINE,
)

_CODE_EXTS = {"ts", "js", "tsx", "jsx", "mjs", "cjs", "py", "go", "java", "kt",
              "rb", "rs", "vue", "scala", "php", "swift", "cs"}
# filenames that conventionally mark an app's entry/bootstrap, any stack
_ENTRY_BASENAMES = {"main", "__main__", "manage", "app", "server", "asgi", "wsgi",
                    "run", "cli", "index", "bootstrap", "program", "application"}
_ROLE_RANK = {"module": 0, "component": 1, "binding": 1, "service": 2, "code": 2,
              "model": 3, "template": 4, "style": 4, "doc": 5}


def _role(path: str) -> str:
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    if name.endswith((".html", ".vue")):
        return "template"
    if name.endswith((".css", ".scss", ".sass", ".less")):
        return "style"
    if ".service." in name or name.endswith("service.ts") or "service" in name:
        return "service"
    if ".component." in name or "component" in name:
        return "component"
    if any(k in name for k in (".guard.", ".interceptor.", ".resolver.", ".pipe.", ".directive.")):
        return "binding"
    if any(k in name for k in (".model.", ".enum.", ".interface.", ".dto.", "type", ".schema.")):
        return "model"
    if ".module." in name or name.endswith("module.ts"):
        return "module"
    if name in ("readme.md", "readme.rst", "readme.txt", "readme"):
        return "doc"
    return "code"


def _basekey(path: str) -> str:
    return path.rsplit("/", 1)[-1].split(".")[0].lower()


def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _import_basekey(target: str) -> str:
    """Resolve an import target to a file basekey. Handles BOTH relative paths
    (JS/TS/Go: './x', '../a/b') AND dotted modules (Python/Java/Kotlin:
    'a.b.c', '..config', 'com.x.UserService') — taking the LAST segment, which
    is the file name, rather than the first (which is the top package)."""
    t = (target or "").strip().strip('"\'').strip()
    if not t:
        return ""
    if "/" in t:
        seg = t.rstrip("/").rsplit("/", 1)[-1]
        return seg.split(".")[0].lower()
    t = t.lstrip(".")  # relative python: ..config -> config
    if "." in t:
        last = t.rsplit(".", 1)[-1].lower()
        if last in _CODE_EXTS:  # was a filename main.ts -> stem 'main'
            return t.rsplit(".", 1)[0].rsplit(".", 1)[-1].lower()
        return last
    return t.lower()


def entry_score(path: str, text: str) -> int:
    """Language-aware likelihood that *path* is the app's entry/bootstrap file.
    0 = no signal. Used to pick the 'main file' the flow starts from — works for
    any stack (Python, Node/TS, Angular, Go, Java/Kotlin, …), not just front-end."""
    base = _basekey(path)
    low = path.lower()
    s = 0
    if base in _ENTRY_BASENAMES:
        s += 3
    if "if __name__" in text or low.endswith("__main__.py"):
        s += 4
    if "bootstrapapplication" in text.lower() or "bootstrapmodule" in text.lower():
        s += 4
    if "package main" in text and "func main(" in text:
        s += 4
    if "public static void main" in text or "fun main(" in text:
        s += 4
    if "@springbootapplication" in text.lower():
        s += 4
    if any(k in text for k in ("FastAPI(", "Flask(", "express(", "createServer(", "Application(")):
        s += 2
    if "/src/" in "/" + low:
        s += 1
    return s


def build_wiring(namespace: str, paths: list[str], *, settings: Optional[Settings] = None) -> Optional[dict]:
    """Map how files connect AND the order to read them, so the UI draws a
    top-to-bottom flow. Edge kinds, strongest first:
      import  — file A imports file B (solid, the real call graph)
      folder  — A and B live in the same directory (dashed, co-located)
      related — fallback link to the main file so nothing floats (dotted)
    Nodes carry `depth` (0 = the main/entry file, growing as you move down the
    import chain) and an `entry` flag, so the renderer can lay them out as an
    ordered vertical flow instead of a role grid."""
    store = get_store(settings or get_settings())
    ns = slugify(namespace)
    real = [p for p in dict.fromkeys(paths)
            if p and ("/" in p or "." in p)
            and p not in ("project-briefing", "feature-map", "git-history")][:10]
    if len(real) < 2:
        return None

    docs = store._load_chunk_docs(ns)
    text_by_path: dict[str, str] = {}
    for c in docs:
        p = c.get("metadata", {}).get("path", "")
        if p in real:
            text_by_path[p] = text_by_path.get(p, "") + "\n" + c.get("text", "")

    # basekey -> path, preferring a non-test, shorter path on collision
    basekey_to_path: dict[str, str] = {}
    for p in sorted(real, key=lambda p: (len(p), p)):
        basekey_to_path.setdefault(_basekey(p), p)

    edges: list[dict] = []
    seen: set = set()
    connected: set = set()
    out_adj: dict[str, list[str]] = {p: [] for p in real}
    indeg: Counter = Counter()

    def add(a: str, b: str, kind: str) -> None:
        key = tuple(sorted((a, b)))
        if a != b and key not in seen:
            seen.add(key)
            edges.append({"from": a, "to": b, "kind": kind})
            connected.add(a)
            connected.add(b)

    # 1) import edges — the real dependency graph (directed for flow ordering)
    for src in real:
        for m in _IMPORT.findall(text_by_path.get(src, "")):
            for t in m:
                if not t:
                    continue
                dst = basekey_to_path.get(_import_basekey(t))
                if dst and dst != src:
                    add(src, dst, "import")
                    if dst not in out_adj[src]:
                        out_adj[src].append(dst)
                        indeg[dst] += 1

    # 2) folder edges — connect co-located files imports didn't already link
    by_dir: dict[str, list[str]] = {}
    for p in real:
        by_dir.setdefault(_dirname(p), []).append(p)
    for sibs in by_dir.values():
        for i in range(1, len(sibs)):
            if sibs[i] not in connected or sibs[i - 1] not in connected:
                add(sibs[i - 1], sibs[i], "folder")

    # pick the MAIN file: strongest entry signal, then most net-outgoing imports
    # (a coordinator that pulls others in), then fewest incomers, then shortest.
    def out_deg(p: str) -> int:
        return len(out_adj.get(p, []))

    entry = max(real, key=lambda p: (entry_score(p, text_by_path.get(p, "")),
                                     out_deg(p) - indeg.get(p, 0),
                                     -indeg.get(p, 0), -len(p)))

    # BFS over import edges from the main file → flow depth (0 = main, deeper = later)
    depth = {entry: 0}
    q = deque([entry])
    while q:
        cur = q.popleft()
        for dst in sorted(out_adj.get(cur, []), key=lambda p: (_ROLE_RANK.get(_role(p), 2), p)):
            if dst not in depth:
                depth[dst] = depth[cur] + 1
                q.append(dst)
    # files not reached by imports go after the deepest reached level, by role
    maxd = max(depth.values()) if depth else 0
    for p in real:
        depth.setdefault(p, maxd + 1)

    # 3) hub fallback — link any still-floating file to the main file
    for p in real:
        if p not in connected and p != entry:
            add(entry, p, "related")

    order = sorted(real, key=lambda p: (depth[p], _ROLE_RANK.get(_role(p), 2), p))
    nodes = [{"id": p, "label": p.rsplit("/", 1)[-1], "role": _role(p),
              "dir": _dirname(p), "path": p, "depth": depth[p], "entry": p == entry}
             for p in order]
    return {"nodes": nodes, "edges": edges, "entry": entry}
