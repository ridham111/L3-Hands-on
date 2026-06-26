"""Guided Codebase Tour — a flow-ordered learning path, bootstrap-first.

A newcomer's hardest problem isn't answering one question (chat does that) or
reading a flat list of parts (the briefing does that) — it's knowing WHERE the
app starts and HOW the pieces connect. So the tour:

  1. finds the BOOTSTRAP / entry point (main, __main__, server, manage, asgi,
     package.json start, Angular main.ts, Go main.go, …);
  2. groups every real source file into FLOW chapters by its architectural role
     — Bootstrap -> Configuration -> Routing/Features -> Services -> Data —
     which is robust even when imports are hard to resolve;
  3. walks the IMPORT graph outward from the entry point to ORDER stops within
     chapters and to draw how each stop connects to the next.

All offline, no LLM. Reuses the wiring import parser, the chunk store, and the
knowledge helpers.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, deque
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings
from ..trace import logger
from .knowledge import load_annotations
from .store import _NOISE_PATH, get_store, slugify
from .wiring import _IMPORT, _basekey, _dirname, _import_basekey, _role, build_wiring, entry_score

# real source we tour through (includes Angular .module.ts, which the KB's
# _is_logic_file drops — the app wiring lives there)
_LOGIC_EXT = {".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".java", ".rb", ".rs",
              ".cs", ".kt", ".scala", ".php", ".swift", ".c", ".cpp", ".vue"}

_FLOW = ["Bootstrap & entry", "Configuration & wiring", "Routing & features",
         "Services & logic", "Data & models"]
_CHAPTER_WHY = {
    "Bootstrap & entry": "Where the app starts — the entry point and what it boots first.",
    "Configuration & wiring": "How the app is configured and assembled before features run.",
    "Routing & features": "The request/feature surface — routes, controllers, components, pages.",
    "Services & logic": "The business logic the features call into — where the real work happens.",
    "Data & models": "The data shapes everything above depends on — read these last.",
}
_ROLE_RANK = {"module": 0, "component": 1, "binding": 1, "service": 2, "code": 2,
              "model": 3, "template": 4, "style": 4, "doc": 5}
_CFG = re.compile(r"config|setting|env|constant|inject|container|provider|registry", re.I)
_ROUTE = re.compile(r"rout|controller|\bview|page|handler|endpoint|\burls?\b", re.I)
_DATA = re.compile(r"model|schema|entity|\bdto\b|repositor|\bdao\b|migration|\benum", re.I)


def _tour_logic_file(path: str) -> bool:
    if _NOISE_PATH.search(path):  # tests / specs / e2e / mocks / fixtures
        return False
    return os.path.splitext(path.lower())[1] in _LOGIC_EXT


def _text_by_path(docs: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in docs:
        p = c.get("metadata", {}).get("path", "")
        if p:
            out[p] = out.get(p, "") + "\n" + c.get("text", "")
    return out


def _build_graph(paths: list[str], text_by_path: dict[str, str]):
    """Directed import graph over indexed source files. Returns (out_adj, indeg)."""
    basekey_to_paths: dict[str, list[str]] = {}
    for p in paths:
        basekey_to_paths.setdefault(_basekey(p), []).append(p)

    def resolve(target: str, src: str) -> Optional[str]:
        key = _import_basekey(target)
        cands = basekey_to_paths.get(key)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        sdir = _dirname(src)
        # prefer same directory, then longest shared path prefix, then shortest path
        return sorted(cands, key=lambda p: (_dirname(p) != sdir,
                                            -_shared_prefix(p, src), len(p), p))[0]

    out_adj: dict[str, list[str]] = {p: [] for p in paths}
    indeg: Counter = Counter()
    for src in paths:
        for m in _IMPORT.findall(text_by_path.get(src, "")):
            for raw in m:
                if not raw:
                    continue
                dst = resolve(raw, src)
                if dst and dst != src and dst not in out_adj[src]:
                    out_adj[src].append(dst)
                    indeg[dst] += 1
    return out_adj, indeg


def _shared_prefix(a: str, b: str) -> int:
    ap, bp = a.split("/"), b.split("/")
    n = 0
    for x, y in zip(ap, bp):
        if x != y:
            break
        n += 1
    return n


def _load_briefing(store, ns: str) -> Optional[dict]:
    p = store.ns_dir(ns) / "briefing.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _detect_entry(paths: list[str], text_by_path: dict[str, str],
                  briefing: Optional[dict], indeg: Counter) -> list[str]:
    """Find the bootstrap entry point(s). Strong code signals first, then
    package.json, then briefing setup sources, then the most-imported file."""
    scored: list[tuple[int, int, str]] = []
    for p in paths:
        s = entry_score(p, text_by_path.get(p, ""))  # shared, language-aware
        if s:
            scored.append((s, -len(p), p))
    if scored:
        scored.sort(reverse=True)
        entries = [scored[0][2]]
        for s, _, p in scored[1:]:      # keep a 2nd, equally-strong, different entry
            if s >= scored[0][0] - 1 and _basekey(p) != _basekey(entries[0]):
                entries.append(p)
                break
        return entries

    # package.json main / scripts.start (reassemble ALL chunks before parsing)
    pkg = next((p for p in text_by_path if p.rsplit("/", 1)[-1].lower() == "package.json"), None)
    if pkg:
        try:
            data = json.loads(text_by_path[pkg])
            run = (data.get("scripts", {}) or {}).get("start", "") or \
                  (data.get("scripts", {}) or {}).get("dev", "")
            target = data.get("main") or data.get("module") or \
                next((w for w in run.split() if "." in w and "/" in w), "")
            if target:
                key = _import_basekey(target)
                hit = next((p for p in paths if _basekey(p) == key), None)
                if hit:
                    return [hit]
        except Exception:
            pass

    # briefing setup-step sources (real indexed paths only)
    if briefing:
        pset = set(paths)
        for c in [str(s) for s in ((briefing.get("setup_steps") or {}).get("sources") or [])]:
            if c in pset:
                return [c]

    # fallback: most-imported file (in-degree), else first by path
    if indeg:
        return [max(paths, key=lambda p: (indeg.get(p, 0), -len(p)))]
    return [sorted(paths)[0]] if paths else []


def _chapter_of(path: str, role: str, is_entry: bool) -> str:
    if is_entry:
        return _FLOW[0]
    low = path.lower()
    if role == "model" or _DATA.search(low):
        return _FLOW[4]
    if role == "module" or _CFG.search(low):
        return _FLOW[1]
    if role in ("component", "binding") or _ROUTE.search(low):
        return _FLOW[2]
    return _FLOW[3]  # services & logic — the default bucket


def _why(path: str, depth: int, indeg: int, is_entry: bool) -> str:
    if is_entry:
        return "the app's entry point — start reading here"
    if depth < 90:
        hop = "directly imported by the entry point" if depth == 1 else f"reached in {depth} hops from the entry point"
        return hop + (f"; used by {indeg} other file(s)" if indeg else "")
    if indeg:
        return f"used by {indeg} other file(s) in the project"
    return "part of the codebase reachable from this area"


def _annotation_for(annotations: list[dict], path: str) -> str:
    for a in annotations:
        if a.get("file") == path and a.get("answer"):
            return str(a["answer"])[:400]
    return ""


def _doc_only_tour(store, ns: str, docs: list[dict]) -> dict:
    """Repos with no source files (docs/config only) still get a tour."""
    readmes = [c for c in docs if c.get("metadata", {}).get("path", "").lower().startswith("readme")]
    src = readmes[0] if readmes else (docs[0] if docs else None)
    stops = []
    if src:
        m = src.get("metadata", {})
        stops.append({"path": m.get("path", ""), "symbol": m.get("symbol", ""),
                      "language": m.get("language", ""), "line_start": int(m.get("line_start", 0)),
                      "line_end": int(m.get("line_end", 0)), "excerpt": (src.get("text", "") or "")[:600],
                      "depth": 0, "imports": [], "reason": "project overview", "note": "", "is_entry": True})
    return {"namespace": ns, "overview": "", "entry_point": stops[0]["path"] if stops else None,
            "entry_points": [s["path"] for s in stops], "total_stops": len(stops),
            "chapters": [{"title": "Overview", "why": "Start here.", "stops": stops}] if stops else [],
            "wiring": None}


def build_tour(namespace: str, *, max_stops: int = 12, max_per_chapter: int = 4,
               narrate: bool = False, settings: Optional[Settings] = None) -> dict:
    settings = settings or get_settings()
    store = get_store(settings)
    ns = slugify(namespace)
    if not store.exists(ns):
        raise ValueError(f"namespace not indexed: {ns}. Ingest the repo first.")
    docs = store._load_chunk_docs(ns)
    if not docs:
        raise ValueError(f"no indexed content for namespace: {ns}")

    text_by_path = _text_by_path(docs)
    paths = [p for p in text_by_path if _tour_logic_file(p)]
    if not paths:
        return _doc_only_tour(store, ns, docs)

    out_adj, indeg = _build_graph(paths, text_by_path)
    briefing = _load_briefing(store, ns)
    annotations = load_annotations(namespace, settings=settings)
    entries = _detect_entry(paths, text_by_path, briefing, indeg)
    entry_set = set(entries)

    # BFS outward from entry points; depth = how far from bootstrap (chapter hint
    # + within-chapter ordering). Unreached files keep depth 99 but are still
    # placed by ROLE, so a weak import graph never produces a garbage tour.
    MAX_VISIT = 250
    depth = {e: 0 for e in entries}
    q = deque(entries)
    while q and len(depth) < MAX_VISIT:
        cur = q.popleft()
        for dst in sorted(out_adj.get(cur, []), key=lambda p: (_ROLE_RANK.get(_role(p), 2), -indeg.get(p, 0), p)):
            if dst not in depth:
                depth[dst] = depth[cur] + 1
                q.append(dst)

    # bucket every file into a fixed flow chapter by role/path (robust), then
    # order within a chapter by depth -> role -> in-degree -> path
    buckets: dict[str, list[str]] = {t: [] for t in _FLOW}
    for p in paths:
        buckets[_chapter_of(p, _role(p), p in entry_set)].append(p)
    for t in _FLOW:
        buckets[t].sort(key=lambda p: (depth.get(p, 99), _ROLE_RANK.get(_role(p), 2),
                                       -indeg.get(p, 0), p))

    out_chapters: list[dict] = []
    all_paths: list[str] = []
    budget = max_stops
    for title in _FLOW:
        if budget <= 0:
            break
        stops = []
        for p in buckets[title][:min(max_per_chapter, budget)]:
            fc = store.first_chunk(ns, p)
            if not fc:
                continue
            m = fc.get("metadata", {})
            d = depth.get(p, 99)
            stops.append({
                "path": p, "symbol": m.get("symbol", ""), "language": m.get("language", ""),
                "line_start": int(m.get("line_start", 0)), "line_end": int(m.get("line_end", 0)),
                "excerpt": (fc.get("text", "") or "")[:600], "depth": d,
                "imports": out_adj.get(p, [])[:6],
                "reason": _why(p, d, indeg.get(p, 0), p in entry_set),
                "insight": "",
                "note": _annotation_for(annotations, p), "is_entry": p in entry_set,
            })
            all_paths.append(p)
        if stops:
            out_chapters.append({"title": title, "why": _CHAPTER_WHY[title], "stops": stops})
            budget -= len(stops)

    wiring = None
    try:
        wiring = build_wiring(ns, all_paths, settings=settings)
    except Exception:
        wiring = None

    overview = ""
    if briefing:
        overview = str((briefing.get("overview") or {}).get("answer", ""))[:600]

    result = {
        "namespace": ns, "overview": overview,
        "entry_point": entries[0] if entries else None, "entry_points": entries,
        "total_stops": len(all_paths), "chapters": out_chapters, "wiring": wiring,
        "narrated": False,
    }

    # Optional LLM narration: one batched call adds a plain-English "what & why"
    # to every stop. Best-effort — a failure leaves the structural tour intact.
    if narrate:
        result["narrated"] = _narrate_tour(result, text_by_path, settings)

    return result


# ---- LLM narration (one batched call adds a one-liner per stop) ------------

def _parse_insight_json(raw: str) -> dict:
    """Robustly pull the {path: insight} object out of an LLM response."""
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S | re.I).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t).rstrip("`").strip()
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    return obj if isinstance(obj, dict) else {}


def _narrate_tour(result: dict, text_by_path: dict, settings: Settings) -> bool:
    """Add a one-line insight to each tour stop via a single LLM call.
    Best-effort: returns True only if at least one stop was narrated; any
    failure leaves the structural tour untouched."""
    stops = [s for ch in result.get("chapters", []) for s in ch.get("stops", [])]
    if not stops:
        return False
    try:
        from ..prompts import TOUR_NARRATE_SYSTEM_PROMPT, build_tour_narrate_prompt
        from ..providers import get_provider
        provider = get_provider(settings)
        if not hasattr(provider, "complete"):
            return False
        try:  # framework vocabulary helps the narration; lazy import avoids a cycle
            from .walkthrough import detect_stack
            stack = ", ".join(detect_stack(list(text_by_path.keys()), text_by_path))
        except Exception:
            stack = ""
        prompt = build_tour_narrate_prompt(stops, stack)
        raw = provider.complete(TOUR_NARRATE_SYSTEM_PROMPT, prompt).text
        mapping = _parse_insight_json(raw)
        if not mapping:
            return False
        narrated = 0
        for s in stops:
            ins = mapping.get(s["path"]) or mapping.get(s["path"].rsplit("/", 1)[-1])
            if isinstance(ins, str) and ins.strip():
                s["insight"] = ins.strip()[:240]
                narrated += 1
        return narrated > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("tour_narrate_failed namespace=%s error=%s",
                       result.get("namespace"), str(exc)[:120])
        return False


# ---- caching (tour.json) — narration is expensive, so cache it once -------

def tour_cache_path(store, ns: str) -> Path:
    return store.ns_dir(ns) / "tour.json"


def load_cached_tour(namespace: str, *, settings: Optional[Settings] = None) -> Optional[dict]:
    settings = settings or get_settings()
    store = get_store(settings)
    p = tour_cache_path(store, slugify(namespace))
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_cached_tour(namespace: str, doc: dict, *, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    store = get_store(settings)
    try:
        tour_cache_path(store, slugify(namespace)).write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
