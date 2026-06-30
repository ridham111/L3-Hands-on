# API Reference — Cortex

Base URL: `http://localhost:8000`

All endpoints (except `GET /` and `GET /health`) require:
```
Authorization: Bearer <your-api-key>
```

Default local key: `dev-local-key`  
Configure keys via `ONBOARDING_API_KEYS` (comma-separated).

Rate limit: **60 requests / 60 seconds** per key (429 when exceeded).  
Request size limit: **16 KB** (413 when exceeded).

Interactive explorer: **http://localhost:8000/docs**

---

## Health & discovery

### `GET /health`
Liveness check. No auth required.

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### `GET /v1/agents`
List available agents and their descriptions.

```bash
curl -H "Authorization: Bearer dev-local-key" http://localhost:8000/v1/agents
```

### `GET /v1/namespaces`
List all indexed repositories.

```bash
curl -H "Authorization: Bearer dev-local-key" http://localhost:8000/v1/namespaces
```

---

## Ingest

### `POST /v1/ingest`
Index a repository. Accepts a local path or a remote git URL.

**Request**
```json
{
  "repo_path": "C:/path/to/repo",
  "namespace": "myrepo",
  "rebuild": false
}
```
Or clone from a URL:
```json
{
  "clone_url": "https://github.com/org/repo.git",
  "clone_token": "ghp_...",
  "namespace": "myrepo"
}
```

**Response**
```json
{
  "namespace": "myrepo",
  "files_indexed": 42,
  "chunks_indexed": 318,
  "already_indexed": false,
  "briefing_pending": true,
  "trace": { "trace_id": "tr_abc123", "duration_ms": 1240, ... }
}
```

**cURL**
```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"repo_path":"C:/path/to/repo","namespace":"myrepo","rebuild":true}'
```

### `POST /v1/resync/{namespace}`
Re-pull and re-index an already-ingested repo (picks up new commits).

```bash
curl -X POST http://localhost:8000/v1/resync/myrepo \
  -H "Authorization: Bearer dev-local-key"
```

---

## Q&A

### `POST /v1/ask`
Ask a question about an indexed repository.

**Request**
```json
{
  "namespace": "myrepo",
  "question": "how does auth work?",
  "history": [],
  "top_k": 8
}
```

> `backend` and `claude_model` may still be accepted in the request body for
> backward compatibility but are **ignored** — there is one backend, the Claude
> Agent SDK (see [architecture](architecture.md#llm-backend--pure-claude-agent-sdk)).

**Response**
```json
{
  "answer": "Authentication is handled in src/auth.py. The login() function at line 14...",
  "grounded": true,
  "sources": [
    {
      "path": "src/auth.py",
      "line_start": 1,
      "line_end": 48,
      "score": 1.0,
      "used": true,
      "snippet": "def login(username, password):\n    ..."
    }
  ],
  "wiring": { ... },
  "validation_status": "passed",
  "trace": { "tool_calls": 3, "iterations": 2, "duration_ms": 820, ... }
}
```

**cURL**
```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"namespace":"myrepo","question":"how does auth work?"}'
```

### `POST /v1/ask/stream`
Same as `/v1/ask` but streams server-sent events (SSE) so the UI can show progress.

Event types emitted during the stream:
- `thinking` — agent started reasoning
- `tool_call` — agent called a tool (includes tool name and argument)
- `tool_result` — tool returned (includes char count, ok/fail)
- `iteration` — loop iteration counter
- `composing` — agent is writing the final answer
- `done` — final `AskResponse` JSON payload

```bash
curl -N -X POST http://localhost:8000/v1/ask/stream \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"namespace":"myrepo","question":"how does auth work?"}'
```

---

## Briefing

### `GET /v1/briefing/{namespace}`
Get the Day-1 briefing for an indexed repo. Generated in the background after ingest.

```bash
curl -H "Authorization: Bearer dev-local-key" \
  http://localhost:8000/v1/briefing/myrepo
```

**Response includes**: `overview`, `key_features`, `folder_map`, `setup_steps`, `recent_work`, `owners`, `glossary`, `trace`.

---

## Tour

### `GET /v1/tour/{namespace}`
Get the guided codebase tour — an ordered list of files to read, starting at the real entry point.

```bash
curl -H "Authorization: Bearer dev-local-key" \
  http://localhost:8000/v1/tour/myrepo
```

**Response includes**: `entry_point`, `chapters` (each with `title`, `why`, `stops`), `total_stops`, `wiring`.

---

## Walkthrough

### `POST /v1/walkthrough/{namespace}`
Start generating the full project walkthrough (async — runs in background).

```bash
curl -X POST -H "Authorization: Bearer dev-local-key" \
  http://localhost:8000/v1/walkthrough/myrepo
```

### `GET /v1/walkthrough/{namespace}`
Poll for the completed walkthrough.

```bash
curl -H "Authorization: Bearer dev-local-key" \
  http://localhost:8000/v1/walkthrough/myrepo
```

**Response includes**: `title`, `stack`, `sections` (each with `key`, `title`, `body`, `files`), `generated_with`.

---

## Utilities

### `GET /v1/gaps/{namespace}`
Find files that are central to the codebase but lack documentation — candidates for annotation.

```bash
curl -H "Authorization: Bearer dev-local-key" \
  http://localhost:8000/v1/gaps/myrepo
```

### `GET /v1/file/{namespace}?path=src/auth.py`
Read the indexed content of a specific file.

```bash
curl -H "Authorization: Bearer dev-local-key" \
  "http://localhost:8000/v1/file/myrepo?path=src/auth.py"
```

### `GET /v1/annotations/{namespace}`
List saved team annotations for a repo.

### `POST /v1/annotations/{namespace}`
Save an annotation for a file or symbol.

```json
{ "file": "src/auth.py", "answer": "Uses JWT; refresh token stored in Redis.", "symbol": "login" }
```

### `GET /v1/chat/{namespace}`
Load conversation history for a namespace.

### `DELETE /v1/chat/{namespace}`
Clear conversation history for a namespace.

---

## Agents endpoint

### `POST /v1/agents/{agent_id}/run`
Run a named agent directly.

Available `agent_id` values: `briefing`, `installation-guide`

```bash
curl -X POST http://localhost:8000/v1/agents/installation-guide/run \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"repo_path":"C:/path/to/repo"}'
```

---

## Error responses

All errors return JSON with a `detail` field and (on server errors) an `error_id` for tracing:

| Status | Meaning |
|---|---|
| `400` | Bad request — missing field or invalid input |
| `401` | Missing or invalid Authorization header |
| `404` | Namespace not found — ingest the repo first |
| `413` | Request body exceeds 16 KB limit |
| `429` | Rate limit exceeded — slow down |
| `500` | Internal error — `error_id` in response maps to `trace.json` |

```json
{ "detail": "namespace not indexed: myrepo. Ingest the repo first." }
{ "detail": "Internal error", "error_id": "tr_abc123def456" }
```
