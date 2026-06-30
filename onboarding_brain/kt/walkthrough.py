"""Project Walkthrough — a long-form, framework-aware, plain-English deep dive.

The guided tour says "read these files, in this order." The walkthrough goes
further: it explains the WHOLE project end-to-end the way a senior engineer would
onboard a new teammate — what it is, the stack, how it boots, how routing works,
the feature areas, the business logic, the data, the shared building blocks, how
it all connects, and how to run it.

It detects the framework so the sections use the right vocabulary (Angular
modules/components/pipes, Django apps/views, Express routes/middleware, …) but
keeps a generic backbone so it works for any stack. Each section is narrated by
the LLM grounded ONLY in the real files for that section (offline backends get a
structural fallback). Cached to ns_dir/walkthrough.json; runs in the background.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings
from ..prompts import WALKTHROUGH_SYSTEM_PROMPT, build_walkthrough_prompt
from ..providers import get_provider
from ..trace import logger
from .knowledge import _GENERIC_DIR
from .store import get_store, slugify
from .tour import _tour_logic_file
from .wiring import _role, build_wiring, entry_score

# ---- framework / stack detection ------------------------------------------
_STACK_SIGNALS = [
    ("Angular", lambda n, b: "angular.json" in n or any(p.endswith((".component.ts", ".module.ts")) for p in n) or "@angular/core" in b),
    ("Next.js", lambda n, b: "next.config.js" in n or "next.config.ts" in n or '"next"' in b),
    ("React", lambda n, b: '"react"' in b or "from 'react'" in b or 'from "react"' in b),
    ("Vue", lambda n, b: any(p.endswith(".vue") for p in n) or '"vue"' in b),
    ("NestJS", lambda n, b: "@nestjs" in b or any(p.endswith(".controller.ts") for p in n)),
    ("Express", lambda n, b: '"express"' in b or "require('express')" in b or "from 'express'" in b),
    ("Django", lambda n, b: "manage.py" in n or "django" in b),
    ("FastAPI", lambda n, b: "fastapi(" in b or "from fastapi" in b),
    ("Flask", lambda n, b: "flask(" in b or "from flask" in b),
    ("Spring Boot", lambda n, b: "@springbootapplication" in b or "springframework" in b),
    ("Rails", lambda n, b: "rails" in b and "gemfile" in n),
    ("Go", lambda n, b: "go.mod" in n),
    (".NET", lambda n, b: any(p.endswith(".csproj") for p in n)),
]
_LANG_BY_EXT = {".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript",
                ".jsx": "JavaScript", ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".rb": "Ruby",
                ".rs": "Rust", ".cs": "C#", ".php": "PHP", ".swift": "Swift", ".vue": "Vue"}

_ROUTE_RE = re.compile(r"rout|controller|\bview|\bpage\b|handler|endpoint|\burls?\b|navigation|app-routing", re.I)
_DATA_RE = re.compile(r"model|schema|entity|\bdto\b|repositor|\bdao\b|migration|prisma|\benum", re.I)
_SUPPORT_RE = re.compile(r"util|helper|\bpipe\b|guard|interceptor|middleware|decorator|\bhooks?\b|constant|config|shared", re.I)
_CONFIG_NAMES = {"package.json", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
                 "dockerfile", "docker-compose.yml", "docker-compose.yaml", "makefile", "angular.json",
                 "pom.xml", "build.gradle", "go.mod", "cargo.toml", "gemfile", "composer.json",
                 "tsconfig.json", ".env.example"}


def detect_stack(paths: list[str], text_by_path: dict[str, str]) -> list[str]:
    names = [p.rsplit("/", 1)[-1].lower() for p in paths]
    nameset = set(names) | {p.lower() for p in paths}
    blob = " ".join(text_by_path.get(p, "")[:3000].lower()
                    for p in paths if p.rsplit("/", 1)[-1].lower() in _CONFIG_NAMES)
    blob += " ".join(text_by_path.get(p, "")[:600].lower() for p in paths[:200])
    found = [name for name, test in _STACK_SIGNALS if test(nameset, blob)]
    # add the dominant language
    langs: dict[str, int] = {}
    for p in paths:
        ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p else ""
        if ext in _LANG_BY_EXT:
            langs[_LANG_BY_EXT[ext]] = langs.get(_LANG_BY_EXT[ext], 0) + 1
    if langs:
        top_lang = max(langs, key=langs.get)
        if top_lang not in found:
            found.append(top_lang)
    return found[:4]


def _feature_folders(logic: list[str]) -> list[tuple[str, list[str]]]:
    """Group files by their most distinctive directory (the feature area)."""
    groups: dict[str, list[str]] = {}
    for p in logic:
        for seg in p.split("/")[:-1]:
            low = seg.strip().lower()
            if not seg or low in _GENERIC_DIR or len(seg) < 3 or seg.startswith("."):
                continue
            groups.setdefault(seg, []).append(p)
            break
    return sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _excerpt(p: str, first_text: dict, lang: dict) -> dict:
    return {"path": p, "language": lang.get(p, ""), "text": first_text.get(p, "")}


def _extract_body(raw: str) -> str:
    """The LLM runs in JSON mode and returns {"explanation": "...markdown..."}.
    Pull the markdown out robustly (strip <think>, code fences, odd key names)."""
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
    if isinstance(obj, dict):
        for k in ("explanation", "body", "walkthrough", "section", "text", "markdown", "content"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in obj.values():  # any string value as a fallback
            if isinstance(v, str) and v.strip():
                return v.strip()
    elif isinstance(obj, str) and obj.strip():
        return obj.strip()
    return t


def _extract_takeaways(raw: str) -> list[str]:
    """Pull the `takeaways` bullet list out of the LLM's JSON response.
    Returns at most 3 short strings; empty list if none were produced."""
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
    if not isinstance(obj, dict):
        return []
    raw_list = obj.get("takeaways") or obj.get("key_points") or obj.get("summary")
    if not isinstance(raw_list, list):
        return []
    out = [str(x).strip() for x in raw_list if str(x).strip()]
    return out[:3]


def build_walkthrough(namespace: str, *, settings: Optional[Settings] = None) -> dict:
    settings = settings or get_settings()
    store = get_store(settings)
    ns = slugify(namespace)
    if not store.exists(ns):
        raise ValueError(f"namespace not indexed: {ns}. Ingest the repo first.")
    docs = store._load_chunk_docs(ns)
    if not docs:
        raise ValueError(f"no indexed content for namespace: {ns}")

    text_by_path: dict[str, str] = {}
    first_text: dict[str, str] = {}
    lang: dict[str, str] = {}
    for c in docs:
        m = c.get("metadata", {})
        p = m.get("path", "")
        if not p or p in ("project-briefing", "feature-map", "git-history"):
            continue
        text_by_path[p] = text_by_path.get(p, "") + "\n" + c.get("text", "")
        if p not in first_text:
            first_text[p] = c.get("text", "")
            lang[p] = m.get("language", "")
    paths = list(first_text.keys())
    logic = [p for p in paths if _tour_logic_file(p)]
    stack = detect_stack(paths, text_by_path)

    # the entry/main file via the verified language-aware scorer (over ALL files)
    entry = (max(logic, key=lambda p: (entry_score(p, text_by_path.get(p, "")), -len(p)))
             if logic else None)

    readme = next((p for p in paths if p.rsplit("/", 1)[-1].lower().startswith("readme")), None)
    configs = [p for p in paths if p.rsplit("/", 1)[-1].lower() in _CONFIG_NAMES]
    routing = [p for p in logic if _ROUTE_RE.search(p) or _role(p) in ("component", "binding")][:8]
    data = [p for p in logic if _DATA_RE.search(p) or _role(p) == "model"][:8]
    support = [p for p in logic if _SUPPORT_RE.search(p) or _role(p) == "binding"][:8]
    services = [p for p in logic if _role(p) in ("service", "code")
                and p not in routing and p not in data and p not in support][:8]
    ui = [p for p in logic if _role(p) in ("component", "template")][:8]
    feature_groups = _feature_folders(logic)

    # a flow-relevant wiring diagram for the walkthrough (entry → routing → services → data)
    flow_set = ex_paths_unique(([entry] if entry else []) + routing[:3] + services[:3] + data[:2])
    wiring = None
    try:
        wiring = build_wiring(ns, flow_set or logic, settings=settings)
    except Exception:
        wiring = None

    def ex(plist):
        return [_excerpt(p, first_text, lang) for p in plist]

    # --- section plan (generic backbone; framework vocabulary in the instructions) ---
    is_frontend = any(s in stack for s in ("Angular", "React", "Vue", "Next.js"))
    overview_files = [p for p in ([readme] if readme else []) + ([entry] if entry else []) if p]
    feature_files = [g[1][0] for g in feature_groups[:8]]  # one representative file per feature area
    connect_files = [p for p in ([entry] if entry else []) + routing[:2] + services[:2] + data[:1] if p]

    plan = [
        ("overview", "What this project is", overview_files + feature_files[:4],
         "Explain in 2-4 short paragraphs what this project is, who it's for, and the big picture of what it does. Use the README and the top feature folders."),
        ("stack", "Tech stack & how it's built", ex_paths_unique(configs + ([readme] if readme else [])),
         f"Describe the languages, frameworks, and build tooling (detected: {', '.join(stack) or 'general'}). Explain what each main config file sets up and how the project is built/installed."),
        ("entry", "How the app starts up", ex_paths_unique(([entry] if entry else []) + routing[:2]),
         "Explain the entry/bootstrap file: what runs first when the app starts, what it wires up (config, routes, server, root module), and the order things initialize."),
        ("routing", "Routing & navigation", routing,
         ("How URLs/screens map to code: routes, route guards, and lazy-loaded modules." if is_frontend
          else "How incoming requests map to code: routes/endpoints, controllers/handlers, and middleware.")),
        ("features", "Main features & modules", feature_files,
         "Walk through the main feature areas/modules (one per folder shown). For each, say what it's responsible for and name its key file(s). This is the product map."),
        ("logic", "Business logic & services", services,
         "Explain the core business logic: the important services/operations, what they actually do, and how the routing/feature layer calls into them. This is where the real work happens."),
        ("data", "Data models & storage", data,
         "Describe the data shapes: models/schemas/DTOs/entities and how data is stored or persisted. Note the key fields and relationships."),
        ("support", "Shared building blocks", support,
         "Explain the reusable pieces: utilities, helpers, pipes, guards, interceptors, middleware, and shared config — what each is for and where it's used."),
        ("connect", "How it all fits together", connect_files,
         "Tie it together: trace one realistic flow end-to-end (e.g. a request or user action) through entry → routing → service → data → response, naming the files at each hop."),
    ]
    if is_frontend and ui:
        plan.insert(5, ("ui", "UI components", ex(ui),
                        "Explain the main UI components/screens: what each renders, the key components, and how they connect to services for data."))

    # normalize file lists to excerpt dicts
    norm_plan = []
    for key, title, files, instr in plan:
        files = files if files and isinstance(files[0], dict) else ex([f for f in files if f])
        norm_plan.append((key, title, files, instr))

    project = ns
    # single backend — the long-form walkthrough runs on the same Claude Agent SDK
    provider = get_provider(settings)

    def make_section(job):
        key, title, files, instr = job
        paths_for = [f["path"] for f in files][:8]
        if not files:
            return {"key": key, "title": title,
                    "body": "_This part isn't present in the indexed code._",
                    "takeaways": [], "files": []}
        takeaways: list[str] = []
        try:
            prompt = build_walkthrough_prompt(project, ", ".join(stack), title, instr, files,
                                              budget_chars=settings.chat_context_budget_chars)
            raw = provider.complete(WALKTHROUGH_SYSTEM_PROMPT, prompt).text
            body = _extract_body(raw)
            takeaways = _extract_takeaways(raw)
            if not body:
                body = _structural_body(title, files)
        except Exception as exc:
            logger.warning("walkthrough_section_failed key=%s error=%s", key, str(exc)[:120])
            body = _structural_body(title, files)
        return {"key": key, "title": title, "body": body,
                "takeaways": takeaways, "files": paths_for}

    with ThreadPoolExecutor(max_workers=4) as exe:
        sections = list(exe.map(make_section, norm_plan))

    return {
        "namespace": ns, "title": f"Project walkthrough — {ns}",
        "stack": stack, "sections": [s for s in sections if s["body"]],
        "wiring": wiring,
        "generated_with": provider.name,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def ex_paths_unique(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(p for p in paths if p))


def _structural_body(title: str, files: list[dict]) -> str:
    """Fallback when no narrative was produced: a grounded, file-by-file outline.
    No 'set a backend' hint here — that would repeat on every section; the UI
    shows a single offline banner instead (see renderWalkthrough)."""
    lines = [f"Key files in this area ({len(files)}):", ""]
    for f in files[:8]:
        first = next((ln.strip() for ln in (f.get("text") or "").splitlines() if ln.strip()), "")
        lines.append(f"- `{f['path']}` — {first[:120] or 'source file'}")
    return "\n".join(lines)


# ---- background generation (cached to walkthrough.json), mirrors the briefing ----
_WALK_JOBS: dict[str, threading.Thread] = {}
_WALK_JOBS_LOCK = threading.Lock()


def walkthrough_path(store, ns: str) -> Path:
    return store.ns_dir(ns) / "walkthrough.json"


def load_cached_walkthrough(namespace: str, *, settings: Optional[Settings] = None) -> Optional[dict]:
    settings = settings or get_settings()
    store = get_store(settings)
    p = walkthrough_path(store, slugify(namespace))
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def walkthrough_running(namespace: str) -> bool:
    ns = slugify(namespace)
    with _WALK_JOBS_LOCK:
        t = _WALK_JOBS.get(ns)
        return bool(t and t.is_alive())


def fire_walkthrough_background(namespace: str, *, settings: Optional[Settings] = None) -> bool:
    """Start generating the walkthrough on a daemon thread. Returns False if one
    is already running for this namespace."""
    settings = settings or get_settings()
    ns = slugify(namespace)
    store = get_store(settings)
    with _WALK_JOBS_LOCK:
        existing = _WALK_JOBS.get(ns)
        if existing and existing.is_alive():
            return False

    def _run() -> None:
        try:
            doc = build_walkthrough(ns, settings=settings)
            walkthrough_path(store, ns).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            logger.info("walkthrough_done namespace=%s sections=%d", ns, len(doc.get("sections", [])))
        except Exception:
            logger.exception("walkthrough_failed namespace=%s", ns)
        finally:
            with _WALK_JOBS_LOCK:
                _WALK_JOBS.pop(ns, None)

    t = threading.Thread(target=_run, daemon=True, name=f"walkthrough-{ns}")
    with _WALK_JOBS_LOCK:
        _WALK_JOBS[ns] = t
    t.start()
    return True
