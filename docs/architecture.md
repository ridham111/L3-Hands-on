# Architecture — Cortex

## What Cortex is

Cortex is a production-ready AI agent system for codebase onboarding. Engineers point it at any repository and it answers questions, generates briefings, and produces guided tours — all grounded in the actual code, never invented.

## System overview

```
┌──────────────────────────────────────────────────────────┐
│                      Access Layer                         │
│   Web UI (web/index.html)  │  REST API  │  CLI (Typer)   │
└──────────────────┬─────────────────────────┬─────────────┘
                   │                         │
          ┌────────▼─────────────────────────▼────────┐
          │              API Server (FastAPI)           │
          │   Auth · Rate-limit · Request-size guard    │
          └────────────────────┬──────────────────────┘
                               │
          ┌────────────────────▼──────────────────────┐
          │             Agent Orchestration             │
          │                                             │
          │  ┌─────────────┐   ┌──────────────────┐   │
          │  │  KT Agent   │   │  Briefing Agent  │   │
          │  │  (L3 loop)  │   │  (LLM + prompts) │   │
          │  └──────┬──────┘   └────────┬─────────┘   │
          │         │                   │              │
          │  ┌──────▼──────────────────▼─────────┐   │
          │  │           Tool Executor             │   │
          │  │  search · read · grep · symbols ·  │   │
          │  │  deps · call_graph · grep_ast       │   │
          │  └──────────────────┬────────────────┘   │
          └─────────────────────┼─────────────────────┘
                                │
          ┌─────────────────────▼─────────────────────┐
          │              Vector Store                   │
          │  TF-IDF  │  Dense (embeddings)  │  Hybrid  │
          │              .kt_index/ (local)             │
          └────────────────────────────────────────────┘
```

## Agents

Cortex exposes six specialised agents, each with a fixed scope and a read-only tool set:

| Agent | ID | Purpose | Entry point |
|---|---|---|---|
| KT Brain | `kt-agent-v1` | Q&A over indexed code (Level-3 loop); separates grounded facts from a labeled "General note (not from the repo)" | `kt/agent.py` |
| Briefing | `onboarding-brain` | Day-1 project overview | `onboarding.py` |
| Guided Tour | `tour` | Ordered file-by-file walk from entry point; each stop carries an LLM one-line insight ("what & why"), cached to `tour.json` | `kt/tour.py` |
| Project Walkthrough | `walkthrough` | Long-form plain-English deep dive; each section has a "Key takeaways" TL;DR and a "read this next" pointer | `kt/walkthrough.py` |
| Installation Guide | `installation-guide` | Toolchain prerequisites + run steps | `kt/agents/` |
| Gap Finder | `gap-finder` | Identifies underdocumented central files | `kt/gaps.py` |

### KT Agent — Level-3 architecture

The KT Agent satisfies every Level-3 criterion: the LLM orchestrates the entire execution; Python is a pure executor.

```
User question
     │
     ▼
┌──────────────────────────────────────────────────┐
│  LLM (Claude, via the Claude Agent SDK)          │
│                                                   │
│  1. Analyse the question                          │
│  2. Choose a tool from 9 available tools          │
│  3. Evaluate the result                           │
│  4. Decide: call another tool OR compose answer   │
│  5. Retry with different query if results are poor│
└────────────────┬─────────────────────────────────┘
                 │  tool_use / end_turn
                 ▼
         ToolExecutor (Python)
         executes exactly what LLM requested
                 │
                 ▼
         Tool result → back into LLM context
```

- `MAX_ITERATIONS = 12` — LLM has 12 rounds to gather evidence
- `_MAX_LLM_RETRIES = 2` — provider call retried up to 2× on transient errors before the loop aborts
- Python makes no retrieval decisions — all logic is LLM-driven

### Available tools (KT Agent)

| Tool | What it does |
|---|---|
| `search_code` | Semantic / TF-IDF search across indexed chunks |
| `read_file` | Full file content (accepts partial paths) |
| `find_files` | Glob / substring file discovery |
| `get_file_structure` | List every indexed path |
| `grep_code` | Exact string or regex across all files |
| `list_symbols` | Exported functions, classes, types with line numbers — no full read |
| `get_dependencies` | Parse package.json / requirements.txt / go.mod / Cargo.toml |
| `call_graph` | Callers + callees of a function (grep-based) |
| `run_grep_ast` | Find all classes, routes, components, decorators, exports |

## Data flow — ingest → answer

```
1. POST /v1/ingest (repo_path or clone_url)
        │
        ▼
   Ingestor: clone → scan files → chunk → embed (optional) → write .kt_index/
        │
        ▼
   VectorStore: chunks stored with metadata (path, line_start, line_end, language)
        │
        ▼
2. POST /v1/ask (namespace, question)
        │
        ▼
   KT Agent loop: LLM calls tools → ToolExecutor reads VectorStore → results fed back
        │
        ▼
   AskResponse: answer + sources + wiring diagram + trace
```

## Storage layout

```
.kt_index/
  <namespace>/
    chunks.json        — indexed text chunks with metadata
    embeddings.npy     — dense vectors (if hybrid/dense backend)
    briefing.json      — cached Day-1 briefing
    tour.json          — cached guided tour
    walkthrough.json   — cached project walkthrough
    chat_history.json  — per-namespace conversation history (or MongoDB)

trace.json             — JSONL append-only structured event log
```

## LLM backend — pure Claude Agent SDK

There is exactly **one backend: `claude_sdk`** ([`providers/claude_agent_sdk_provider.py`](../onboarding_brain/providers/claude_agent_sdk_provider.py)).
This is a deliberate "pure agent" stance — the value is the agent, not an LLM-call multiplexer.

| Backend | Config value | Auth |
|---|---|---|
| Claude Agent SDK | `claude_sdk` | Subscription OAuth: `CLAUDE_CODE_OAUTH_TOKEN` or `~/.claude` login — no billed key |

How it works: the **claude-agent-sdk** owns the agentic loop internally. Cortex's 9 tools are
handed to it as an in-process MCP server ([`kt/agent_sdk.py`](../onboarding_brain/kt/agent_sdk.py))
built from `TOOL_DEFINITIONS` + `ToolExecutor`. Built-in Claude Code tools (`Read`/`Bash`/…) are
disabled (`tools=[]`) and `allowed_tools` is locked to `mcp__kt__*`, preserving the read-only /
grounded guarantee. Grounding, sources, wiring, and traces are assembled by `assemble_agent_response()`
(in `kt/agent.py`). The SDK is async; the provider bridges to the app's sync code by running each
call in a worker thread with its own event loop. The single-shot helpers (briefing, install guide,
tour, walkthrough) call the same provider's `_complete()`.

> The diagram above shows the conceptual Level-3 loop. In this codebase the loop itself lives
> **inside the SDK**, not in `kt/agent.py` — `kt/agent.py` now only holds the system prompt, tool
> metadata, and shared response assembly. A deterministic RAG path + `StubProvider` remain in
> `kt/chat.py`/`evals/` **solely** as the offline test harness; no production backend reaches them.

## Security model

- **Read-only tools** — agents can read indexed content only; no write, execute, or shell access
- **Filesystem fence** — `ONBOARDING_ALLOWED_ROOTS` restricts which paths the ingestor may read
- **Prompt injection defence** — repo content is wrapped in `<<<REPO_CONTEXT_JSON>>>` markers; the system prompt instructs the LLM to treat everything inside as DATA, never as instructions
- **Grounding validation** — every cited source is checked against the real index; unresolved citations trigger a `warning` validation status
- **Bearer token auth** — constant-time HMAC comparison prevents timing attacks
- **Secrets never logged** — API keys and clone tokens are used in-memory and never written to disk or traces

## Observability

Every request produces:

- A `trace_id` (format: `tr_<16-hex>`) attached to all log lines and surfaced in error responses
- Structured JSON log lines to stderr (timestamp, level, event, trace_id)
- A JSONL entry in `trace.json` with: namespace, question excerpt, tool call count, files accessed, iteration count, duration ms, errors
- Per-response `grounding` dict embedded in the `Trace` object returned to the caller
