"""Level-3 AI Agent: LLM-orchestrated tool-use loop for codebase Q&A.

Architecture
────────────
The LLM receives tool definitions and a question. It decides:
  • which tools to call (search_code / read_file / find_files / grep_code / …)
  • in what order
  • when it has enough evidence to write the final answer
  • when to retry with a different query if results are poor

Python is a pure executor — it runs exactly what the LLM requests and returns
results verbatim. No hardcoded retrieval logic, no forced pipelines.

Level-3 criteria satisfied
───────────────────────────
  ✓ LLM understands the goal and plans how to achieve it
  ✓ LLM dynamically determines next steps via tool calls
  ✓ LLM selects tools based on task context
  ✓ LLM evaluates intermediate results, decides if more info is needed
  ✓ LLM recovers: retries with different queries when first results are poor
  ✓ LLM orchestrates the entire execution — Python only executes its requests
"""
from __future__ import annotations

import re
from typing import Optional

from ..config import Settings, get_settings
from ..contract import AskRequest, AskResponse, Source, Trace
from ..providers import get_provider
from ..trace import append_trace, log_event, new_trace_id, timed
from .chat_store import get_chat_store
from .store import get_store, slugify
from .tools import TOOL_DEFINITIONS, ToolExecutor

# ─────────────────────────────────────────────────────────────────────────────
# Agent system prompt — goal-oriented, tool-first
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """\
You are KT Brain, an expert codebase agent. Engineers ask you questions about a
software repository and you answer by ACTUALLY READING THE CODE — not from
memory, assumptions, or general knowledge about frameworks.

TOOLS YOU HAVE:
  search_code        — semantic search by concept, feature, or keyword
  read_file          — read a complete file (accepts partial path like "auth.ts")
  find_files         — list files matching a name pattern ("routes", "auth", "*.ts")
  get_file_structure — see every indexed file in the project
  grep_code          — find exact strings, function names, or imports
  list_symbols       — list all functions/classes/types defined in a file (fast — no full read)
  get_dependencies   — parse package.json/requirements.txt/go.mod — what libraries are used
  call_graph         — show where a function is defined, where it's called from, and what it calls
  run_grep_ast       — find all structural nodes by type: class, function, interface,
                       component, decorator, route, export

HOW TO WORK:
1. Analyse the question — what specific things do you need to find?
2. search_code first: use the key concept or feature name as the query.
3. If results reference other files or functions, read_file them.
4. Use find_files + read_file when you know a filename but search missed it.
5. Use grep_code to find exact function names, imports, or configuration keys.
6. Use list_symbols when you need a file's API surface quickly — faster than read_file
   when you only need to know what's defined, not the full body.
7. Use get_dependencies for questions about installed libraries, versions, or tech stack.
8. Use call_graph to trace a function's callers and callees — ideal for impact analysis
   ("what breaks if I change X?") and understanding entry points.
9. Use run_grep_ast to enumerate all classes, routes, components, or decorators in the
   repo — ideal for architecture questions ("what API routes exist?", "what services?").
10. Keep gathering evidence until you are CONFIDENT in the answer.
11. Write the final answer citing exact file paths and line numbers
    (e.g. "In src/auth/middleware.ts at line 42, the guard checks...").

TOOL SELECTION GUIDE:
  "what does X do?"                 → search_code, then read_file
  "where is X defined?"             → grep_code or call_graph
  "who calls X?" / "what calls X?"  → call_graph
  "what classes/routes exist?"      → run_grep_ast
  "what's exported from this file?" → list_symbols (avoid read_file for this)
  "what libraries/packages are used?" → get_dependencies
  "show me all API endpoints"       → run_grep_ast(node_type="route")
  "is X imported anywhere?"         → grep_code

RECOVERY — when search returns poor results:
  • Rephrase the query with different terms
  • Use get_file_structure to find the right area, then read_file
  • Use grep_code with the exact function or variable name
  • Use run_grep_ast if you're looking for a structural pattern

RULES:
  • Every repo-specific claim (what this code does, file names, APIs, behavior)
    must come from code you read — never from general knowledge.
  • Be concrete: name functions, classes, variables, exact file paths.
  • Cite sources inline: "In `src/main.ts`..." not vague statements.
  • If the codebase genuinely contains nothing relevant after a thorough search,
    you MUST BEGIN your reply with this exact line, on its own:
        I couldn't find this in the indexed code.
    Then, below it, briefly say what you searched for and (optionally) suggest
    what the user might mean. Do NOT cite files as if they were evidence for a
    thing that isn't there — a "not found" answer has no sources.

GENERAL NOTE — separating your knowledge from the repo's:
  • Keep the main answer strictly grounded in the code you read.
  • When practical engineering knowledge helps — toolchain prerequisites,
    version compatibility, what a command does, common pitfalls — put it in a
    final section that begins on its own line with EXACTLY this label:
        **General note (not from the repo):**
    followed by your guidance. This tells the reader it's your expertise, not a
    repo fact, so they can trust the grounded part absolutely.
  • For setup / run / install / dependency questions this General note is
    REQUIRED: state the runtime versions, global CLIs, and OS gotchas you know
    (e.g. which Node.js versions the framework version in package.json supports).
  • For "what / how / where" code questions, only add it when it genuinely
    helps — otherwise omit it entirely. Never pad. Never mix general knowledge
    into the grounded answer above the label.

OUTPUT STYLE:
  • Answer directly and concisely. Do not dump every detail you found.
  • Lead with the key insight, not a preamble.
  • Only list items (tables, bullet lists) if the question explicitly asks for
    a breakdown or the list is genuinely the clearest way to answer.
  • Skip obvious things — if 14 modules are imported, say "14 feature modules
    (auth, map, dashboard, etc.)" not a full table unless asked.
  • A short closing summary (1–2 sentences) is fine when it adds clarity.
    Do not end with a summary that just repeats what you already said.
  • Aim for the length a senior engineer would write in a Slack message:
    enough to be precise, short enough to be read in 30 seconds.
"""

NOT_FOUND = "I couldn't find this in the indexed code."
MAX_ITERATIONS = 12
_MAX_LLM_RETRIES = 2  # per-iteration provider retries before giving up
AGENT_ID = "kt-agent-v1"
_SKIP_PATHS = {"project-briefing", "feature-map", "git-history"}

_CONVERSATIONAL_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|howdy|greetings|sup|what'?s\s+up|yo+|good\s+(morning|afternoon|evening|day)|"
    r"thanks?|thank\s+you|thx|ty|cheers|ok+|okay|cool|great|awesome|nice|good|got\s+it|understood|"
    r"sure|sounds\s+good|perfect|excellent|alright|bye|goodbye|see\s+you|cya|later|"
    r"how\s+are\s+you|what'?s\s+up|who\s+are\s+you|what\s+can\s+you\s+do|help)\W*$",
    re.IGNORECASE,
)

_CONVERSATIONAL_REPLY = (
    "Hey! I'm KT Brain — I help engineers get up to speed on this codebase.\n\n"
    "Ask me anything about the repo: how a feature works, where something is defined, "
    "how the app boots, what a file does, or how two components connect. I'll read the actual code and answer."
)

_TOOL_ARG_KEY: dict[str, str] = {
    "search_code": "query",
    "read_file": "path",
    "find_files": "pattern",
    "grep_code": "pattern",
    "list_symbols": "path",
    "get_dependencies": "filter",
    "call_graph": "function_name",
    "run_grep_ast": "node_type",
}


def _tool_arg(name: str, inp: dict) -> str:
    key = _TOOL_ARG_KEY.get(name)
    return str(inp.get(key, ""))[:80] if key else ""



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def assemble_agent_response(
    request: AskRequest,
    settings: Settings,
    store,
    namespace: str,
    executor: ToolExecutor,
    final_answer: str,
    *,
    trace_id: str,
    errors: list[str],
    iterations_used: int,
    duration_ms: int,
) -> AskResponse:
    """Build the grounded AskResponse from the executor's accumulated state.

    Used by `agent_ask_sdk` (the Claude Agent SDK loop) to turn the tools' recorded
    `used_paths` / `call_log` into sources, wiring, trace, and chat-history. Kept in
    this module because it owns the agent's shared constants and tool metadata.
    """
    is_not_found = _looks_like_not_found(final_answer)

    # Sources = files the answer actually CITES (intersected with what the agent
    # read), not every file a search happened to surface. This is what makes the
    # "verified sources" honest:
    #   • not-found answer  -> no sources, ungrounded (don't imply we found it)
    #   • normal answer     -> only the files the answer names
    #   • answer cites none -> fall back to browsed files (legacy behaviour)
    if is_not_found:
        sources: list[Source] = []
        grounded = False
    else:
        cited = _cited_paths(final_answer, executor.used_paths)
        sources = _sources_from_executor(executor, store, namespace,
                                         only=cited if cited else None)
        grounded = bool(sources)

    used_for_wiring = {s.path for s in sources}
    wiring = None
    if not is_not_found and used_for_wiring:
        real_paths = [p for p in used_for_wiring if p not in _SKIP_PATHS]
        if real_paths:
            try:
                from .wiring import build_wiring
                wiring = build_wiring(namespace, real_paths, settings=settings)
            except Exception:
                pass

    trace = Trace(
        trace_id=trace_id,
        agent_id=AGENT_ID,
        model_used=settings.model_used,
        duration_ms=duration_ms,
        strategy="agent",
        errors=errors,
        grounding={
            "namespace": namespace,
            "tool_calls": len(executor.call_log),
            "files_accessed": len(executor.used_paths),
            "iterations": iterations_used,
            "call_log": executor.call_log,
        },
    )

    append_trace({
        "trace_id": trace_id, "event": "agent_ask",
        "namespace": namespace, "question": request.question[:200],
        "tool_calls": len(executor.call_log),
        "files_accessed": len(executor.used_paths),
        "iterations": iterations_used,
        "duration_ms": duration_ms,
    })
    log_event("agent_ask_end", trace_id)

    # Persist to chat history (wiring + sources saved so session restore works)
    try:
        _hist_extra: dict = {}
        if wiring:
            _hist_extra["wiring"] = wiring.model_dump(mode="json") if hasattr(wiring, "model_dump") else wiring
        src_compact = [
            {"path": s.path, "line_start": s.line_start, "line_end": s.line_end,
             "score": round(s.score, 3), "used": s.used,
             "symbol": (s.symbol or "")[:60], "snippet": (s.snippet or "")[:280]}
            for s in sources if s.path not in _SKIP_PATHS
        ][:14]
        if src_compact:
            _hist_extra["sources"] = src_compact
        get_chat_store(settings).append(
            namespace, request.question, final_answer, extra=_hist_extra or None
        )
    except Exception:
        pass

    return AskResponse(
        answer=final_answer,
        sources=sources,
        grounded=grounded,
        validation_status="warning" if errors else "passed",
        wiring=wiring,
        trace=trace,
    )

# Phrases that signal the agent concluded the thing doesn't exist in the repo.
# Anchored to the answer's opening so a passing "there is no X, instead Y" mid-
# answer doesn't trip it. Primary detection is the NOT_FOUND sentinel (the system
# prompt asks the agent to lead with it); this is the resilient fallback for when
# the model phrases the negative in its own words.
_REPO = r"(?:code\s?base|repo(?:sitory)?|project|app(?:lication)?|source\s+code|code)"
_NOT_FOUND_RE = re.compile(
    r"i\s+(?:couldn'?t|could\s+not|did\s*n'?t|was\s+un(?:able)?\s*to)\s+find"
    r"|there\s+(?:is|are|'s)\s+no\b[^.]{0,90}?\banywhere\b"
    r"|there\s+(?:is|are|'s)\s+no\b[^.]{0,90}?\bin\s+(?:this|the|the\s+entire)\s+" + _REPO + r"\b"
    r"|\b(?:does|do)\s*n'?t\s+exist\b"
    r"|\b(?:does|do)\s*n'?t\s+appear\b[^.]{0,40}?\b(?:anywhere|in\s+(?:this|the)\s+" + _REPO + r")\b"
    r"|\bno\s+(?:such|concept\s+of|notion\s+of|mention\s+of|trace\s+of)\b"
    r"|\b(?:isn'?t|aren'?t)\s+(?:a\s+thing|anywhere)\b"
    r"|\bnot\s+a\s+thing\b",
    re.IGNORECASE,
)


def _looks_like_not_found(answer: str) -> bool:
    """True if the answer is essentially 'this isn't in the repo'."""
    head = (answer or "").strip()[:300]
    if head.lower().startswith("i couldn't find"):
        return True
    return bool(_NOT_FOUND_RE.search(head))


def _cited_paths(answer: str, candidates: set[str]) -> set[str]:
    """Of the files the agent browsed, return only those it actually names in the
    answer. Stops 'search noise' (files a query surfaced but the answer never uses)
    from being presented as verified sources."""
    text = (answer or "").lower()
    hits = set()
    for p in candidates:
        pl = p.lower()
        base = pl.rsplit("/", 1)[-1]
        if pl in text or (base and base in text):
            hits.add(p)
    return hits


def _sources_from_executor(executor: ToolExecutor, store, namespace: str,
                           only: set[str] | None = None) -> list[Source]:
    """Build Source list from the files the agent accessed via tools.

    `only` restricts to a subset (e.g. just the files the answer actually cited)."""
    paths = executor.used_paths if only is None else (executor.used_paths & only)
    sources = []
    for path in sorted(paths):
        if path in _SKIP_PATHS:
            continue
        chunk = store.first_chunk(namespace, path)
        if not chunk:
            continue
        m = chunk.get("metadata", {})
        sources.append(Source(
            path=path,
            language=m.get("language", ""),
            line_start=int(m.get("line_start", 0)),
            line_end=int(m.get("line_end", 0)),
            score=1.0,
            snippet=(chunk.get("text", "") or "")[:400],
            symbol=str(m.get("symbol", "")),
            used=True,
        ))
    return sources
