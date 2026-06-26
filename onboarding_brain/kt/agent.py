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

HOW TO WORK:
1. Analyse the question — what specific things do you need to find?
2. search_code first: use the key concept or feature name as the query.
3. If results reference other files or functions, read_file them.
4. Use find_files + read_file when you know a filename but search missed it.
5. Use grep_code to find exact function names, imports, or configuration keys.
6. Keep gathering evidence until you are CONFIDENT in the answer.
7. Write the final answer citing exact file paths and line numbers
   (e.g. "In src/auth/middleware.ts at line 42, the guard checks...").

RECOVERY — when search returns poor results:
  • Rephrase the query with different terms
  • Use get_file_structure to find the right area, then read_file
  • Use grep_code with the exact function or variable name

RULES:
  • Every claim must come from code you read — never from general knowledge.
  • Be concrete: name functions, classes, variables, exact file paths.
  • Cite sources inline: "In `src/main.ts`..." not vague statements.
  • If the codebase genuinely contains nothing relevant after thorough search,
    say so honestly: "I couldn't find this in the indexed code." and briefly
    describe what you searched for.

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
}


def _tool_arg(name: str, inp: dict) -> str:
    key = _TOOL_ARG_KEY.get(name)
    return str(inp.get(key, ""))[:80] if key else ""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def agent_ask(
    request: AskRequest,
    *,
    settings: Optional[Settings] = None,
    provider=None,
    event_callback=None,
) -> AskResponse:
    """Run the agentic tool-use loop and return a grounded AskResponse.

    The LLM controls the entire flow. Python executes tool calls and assembles
    the final response object — it makes no content decisions.
    """
    settings = settings or get_settings()
    store = get_store(settings)
    trace_id = new_trace_id()
    errors: list[str] = []
    namespace = slugify(request.namespace)
    log_event("agent_ask_start", trace_id)

    if not store.exists(namespace):
        raise ValueError(f"namespace not indexed: {namespace}. Ingest the repo first.")

    if provider is None:
        provider = get_provider(settings)

    executor = ToolExecutor(namespace=namespace, store=store)
    cb = event_callback if callable(event_callback) else (lambda _e: None)

    # ── Short-circuit purely conversational messages ──────────────────────────
    if _CONVERSATIONAL_RE.match(request.question.strip()):
        cb({"type": "composing"})
        return AskResponse(
            answer=_CONVERSATIONAL_REPLY,
            sources=[],
            grounded=False,
            validation_status="passed",
            wiring=None,
            trace=Trace(
                trace_id=new_trace_id(),
                agent_id=AGENT_ID,
                model_used=settings.model_used,
                duration_ms=0,
                strategy="conversational",
                errors=[],
                grounding={"namespace": namespace, "tool_calls": 0,
                           "files_accessed": 0, "iterations": 0, "call_log": []},
            ),
        )

    # ── Build initial conversation ────────────────────────────────────────────
    history = [h.model_dump() for h in request.history]
    convo = ""
    for turn in history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        convo += f"{role}: {turn.get('content', '')[:600]}\n"

    user_content = (f"Previous conversation:\n{convo}\n" if convo else "") + request.question
    messages: list[dict] = [{"role": "user", "content": user_content}]

    # ── Agentic tool-use loop ─────────────────────────────────────────────────
    final_answer = NOT_FOUND
    iterations_used = 0

    with timed() as t:
        for iteration in range(MAX_ITERATIONS):
            iterations_used = iteration + 1
            cb({"type": "iteration", "n": iterations_used, "max": MAX_ITERATIONS})
            if iteration == 0:
                cb({"type": "thinking"})
            try:
                result = provider.complete_turn(
                    AGENT_SYSTEM_PROMPT,
                    messages,
                    TOOL_DEFINITIONS,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"llm_error:{exc}")
                cb({"type": "error_iter", "message": str(exc)})
                break

            if result.stop_reason == "end_turn":
                cb({"type": "composing"})
                final_answer = result.text.strip() or NOT_FOUND
                break

            if result.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": result.raw_content})

                tool_results = []
                for tc in result.tool_calls:
                    cb({"type": "tool_call", "tool": tc.name, "arg": _tool_arg(tc.name, tc.input)})
                    output = executor.execute(tc.name, tc.input)
                    ok = bool(output.strip()) and not output.startswith("[error")
                    cb({"type": "tool_result", "tool": tc.name, "ok": ok,
                        "chars": len(output)})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": output,
                    })
                messages.append({"role": "user", "content": tool_results})

            elif result.stop_reason == "max_tokens":
                if result.text.strip():
                    final_answer = result.text.strip()
                errors.append("max_tokens_reached")
                break
        else:
            # Exhausted MAX_ITERATIONS — use whatever the last assistant text was
            errors.append("max_iterations_reached")
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    for block in (msg.get("content") or []):
                        if hasattr(block, "type") and block.type == "text" and block.text.strip():
                            final_answer = block.text.strip()
                            break
                    break

    # ── Assemble response ─────────────────────────────────────────────────────
    is_not_found = final_answer.lower().startswith("i couldn't find")
    grounded = bool(executor.used_paths) or is_not_found

    # Don't surface sources/wiring for not-found answers: the agent scanned
    # files while searching but none are relevant — showing them misleads.
    sources = [] if is_not_found else _sources_from_executor(executor, store, namespace)

    wiring = None
    if not is_not_found and executor.used_paths:
        real_paths = [p for p in executor.used_paths if p not in _SKIP_PATHS]
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
        duration_ms=t["ms"],
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
        "duration_ms": t["ms"],
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sources_from_executor(executor: ToolExecutor, store, namespace: str) -> list[Source]:
    """Build Source list from every file the agent accessed via tools."""
    sources = []
    for path in sorted(executor.used_paths):
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
