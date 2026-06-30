"""Level-3 agent, Claude Agent SDK edition.

Same contract as `agent.py` (grounded, tool-using codebase Q&A) but the tool-use
loop is owned by Anthropic's agent harness instead of being hand-rolled here.

How it differs from `agent.py`
──────────────────────────────
  • The 9 KT tools are exposed to Claude as an in-process MCP server (built from
    the SAME `TOOL_DEFINITIONS` + `ToolExecutor`, so retrieval behaviour is
    identical). Each tool handler just calls `executor.execute(...)`.
  • A single `query()` call runs the WHOLE loop: Claude decides which tools to
    call, the SDK invokes our handlers, feeds results back, and iterates until it
    writes the final answer. We only consume the message stream to (a) drive the
    same UI events the old loop emitted and (b) capture the final text.
  • Grounding/sources/wiring/trace are assembled by the shared
    `assemble_agent_response()` from `agent.py`, reading the same
    `executor.used_paths` / `call_log` the handlers populate.

The harness is locked down (no project settings, only our read-only MCP tools)
so Cortex's core guarantee holds: answers come only from the indexed code.
"""
from __future__ import annotations

from typing import Optional

from ..config import Settings, get_settings
from ..contract import AskRequest, AskResponse, Trace
from ..providers import get_provider
from ..providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider, _run_sync
from ..trace import log_event, new_trace_id, timed
from .agent import (
    AGENT_ID,
    AGENT_SYSTEM_PROMPT,
    MAX_ITERATIONS,
    NOT_FOUND,
    _CONVERSATIONAL_REPLY,
    _CONVERSATIONAL_RE,
    _tool_arg,
    assemble_agent_response,
)
from .store import get_store, slugify
from .tools import TOOL_DEFINITIONS, ToolExecutor

_MCP_SERVER = "kt"
# The system prompt names tools bare (search_code, …); under MCP they are exposed
# as mcp__kt__search_code. Bridge the two so the model isn't confused by the names.
_SDK_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT + (
    "\n\nNOTE: Your tools are provided by the \"kt\" server. Their full names are "
    "mcp__kt__<tool> (e.g. mcp__kt__search_code), but they behave exactly as "
    "described above. You have NO other tools — do not attempt to read the "
    "filesystem or run commands; use only the kt tools to inspect the code."
)


def _make_sdk_tools(executor: ToolExecutor, cb):
    """Wrap each KT tool as an in-process MCP tool backed by the executor.

    Emitting tool_call/tool_result here (rather than from the message stream)
    keeps a single source of truth and lets us map the mcp__kt__* names back to
    the bare tool name the UI expects.
    """
    from claude_agent_sdk import tool

    def _make_handler(tool_name: str):
        async def _handler(args: dict):
            cb({"type": "tool_call", "tool": tool_name, "arg": _tool_arg(tool_name, args)})
            output = executor.execute(tool_name, dict(args))
            ok = bool(output.strip()) and not output.lower().startswith(("tool error", "[error"))
            cb({"type": "tool_result", "tool": tool_name, "ok": ok, "chars": len(output)})
            return {"content": [{"type": "text", "text": output}]}
        return _handler

    return [
        tool(d["name"], d["description"], d["input_schema"])(_make_handler(d["name"]))
        for d in TOOL_DEFINITIONS
    ]


def _build_mcp_server(executor: ToolExecutor, cb):
    """Build the in-process MCP server exposing the 9 KT tools to Claude."""
    from claude_agent_sdk import create_sdk_mcp_server

    return create_sdk_mcp_server(
        name=_MCP_SERVER, version="1.0.0", tools=_make_sdk_tools(executor, cb)
    )


def agent_ask_sdk(
    request: AskRequest,
    *,
    settings: Optional[Settings] = None,
    provider=None,
    event_callback=None,
) -> AskResponse:
    """Run the agentic loop via the Claude Agent SDK and return a grounded response."""
    settings = settings or get_settings()
    store = get_store(settings)
    trace_id = new_trace_id()
    errors: list[str] = []
    namespace = slugify(request.namespace)
    log_event("agent_ask_sdk_start", trace_id)

    if not store.exists(namespace):
        raise ValueError(f"namespace not indexed: {namespace}. Ingest the repo first.")

    if provider is None:
        provider = get_provider(settings)
    if not isinstance(provider, ClaudeAgentSDKProvider):
        # Defensive: this path requires the SDK provider. Caller dispatches on backend.
        raise TypeError("agent_ask_sdk requires the claude_sdk backend provider")

    executor = ToolExecutor(namespace=namespace, store=store)
    cb = event_callback if callable(event_callback) else (lambda _e: None)

    # ── Short-circuit purely conversational messages (mirror agent.py) ────────
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

    # ── Build the prompt (history folded in, same as agent.py) ────────────────
    history = [h.model_dump() for h in request.history]
    convo = ""
    for turn in history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        convo += f"{role}: {turn.get('content', '')[:600]}\n"
    user_content = (f"Previous conversation:\n{convo}\n" if convo else "") + request.question

    mcp_server = _build_mcp_server(executor, cb)
    allowed = [f"mcp__{_MCP_SERVER}__{d['name']}" for d in TOOL_DEFINITIONS]

    final_answer = NOT_FOUND
    iterations_used = 0

    async def _run() -> tuple[str, int]:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock, query

        options = provider.base_options(
            system_prompt=_SDK_SYSTEM_PROMPT,
            mcp_servers={_MCP_SERVER: mcp_server},
            allowed_tools=allowed,
            max_turns=MAX_ITERATIONS,
        )
        turns = 0
        last_text = ""
        result_text: str | None = None
        cb({"type": "thinking"})
        async for msg in query(prompt=user_content, options=options):
            if isinstance(msg, AssistantMessage):
                if msg.error:
                    # e.g. rate_limit / billing_error. Record and stop cleanly —
                    # breaking lets the SDK's async generator close normally
                    # (raising here crashes its aclose with "already running").
                    errors.append(f"sdk_error:{msg.error}")
                    break
                has_tool = any(isinstance(b, ToolUseBlock) for b in msg.content)
                if has_tool:
                    turns += 1
                    cb({"type": "iteration", "n": turns, "max": MAX_ITERATIONS})
                texts = [b.text for b in msg.content if isinstance(b, TextBlock) and b.text.strip()]
                if texts:
                    last_text = "\n".join(texts).strip()
            elif isinstance(msg, ResultMessage):
                if msg.is_error:
                    errors.append(f"sdk_result_error:{msg.errors or msg.subtype}")
                else:
                    result_text = msg.result
        cb({"type": "composing"})
        answer = (result_text or last_text or "").strip() or NOT_FOUND
        return answer, max(turns, 1)

    with timed() as t:
        try:
            final_answer, iterations_used = _run_sync(_run)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"agent_sdk_failed:{exc}")
            cb({"type": "error_iter", "message": str(exc), "retry": 0})

    return assemble_agent_response(
        request, settings, store, namespace, executor, final_answer,
        trace_id=trace_id, errors=errors, iterations_used=iterations_used,
        duration_ms=t["ms"],
    )
