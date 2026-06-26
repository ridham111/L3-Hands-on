"""HTTP API + web UI for Onboarding Brain.

    GET  /                       -> single-page UI
    GET  /health                 -> liveness
    GET  /v1/agents              -> catalog
    POST /v1/agents/{id}/run     -> run the agent

Controls: bearer-token auth (stubbed), request-size cap, per-key rate limit,
uniform JSON errors, no stack-trace leakage.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from onboarding_brain import AGENT_ID, __version__
from onboarding_brain.config import get_settings
from onboarding_brain.contract import AnnotateRequest, AskRequest, IngestRequest, OnboardingRequest
from onboarding_brain.kt.chat import ask as kt_ask
from onboarding_brain.kt.chat import clear_chat_history, load_chat_history
from onboarding_brain.kt.ingest import _BRIEFING_ERRORS, _BRIEFING_JOBS, _BRIEFING_JOBS_LOCK, ingest_repo
from onboarding_brain.kt.store import get_store, slugify
from onboarding_brain.onboarding import RepoAccessError, generate_briefing
from onboarding_brain.trace import logger

app = FastAPI(title="Cortex — Codebase Intelligence", version=__version__)
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

_WINDOW_S = 60
_hits: dict[str, deque[float]] = defaultdict(deque)
# Ingest writes index files on disk; serialize per-repo so two concurrent
# rebuilds of the SAME repo can't interleave their writes — but different repos
# ingest in parallel (a global lock made one slow clone/embed block all others).
_INGEST_LOCKS: dict[str, threading.Lock] = {}
_INGEST_LOCKS_META = threading.Lock()


def _ingest_lock_for(key: str) -> threading.Lock:
    with _INGEST_LOCKS_META:
        lock = _INGEST_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _INGEST_LOCKS[key] = lock
        return lock


def _rate_limit(key: str, limit: int) -> None:
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > _WINDOW_S:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")
    q.append(now)


def _token_ok(token: str, keys: tuple[str, ...]) -> bool:
    # constant-time compare against each configured key to avoid a timing
    # side-channel that could leak the key length/prefix to a brute-forcer
    return any(hmac.compare_digest(token, k) for k in keys)


async def require_auth(authorization: str | None = Header(default=None)) -> str:
    settings = get_settings()
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
    # no keys configured => open mode (local dev); a startup warning is logged
    valid = (not settings.api_keys) or _token_ok(token, settings.api_keys)
    # Throttle BEFORE rejecting so key brute-forcing is rate-limited too; all
    # invalid tokens share one bucket, keeping the hit map bounded.
    _rate_limit(token if valid else "anon", settings.rate_limit_per_min)
    if not valid:
        raise HTTPException(status_code=401, detail="invalid_or_missing_api_key")
    return "ok"


@app.on_event("startup")
async def _warn_insecure_defaults() -> None:
    """Loudly flag deployment-unsafe defaults at boot (they're fine for local dev)."""
    s = get_settings()
    if not s.api_keys:
        logger.warning("AUTH OPEN: ONBOARDING_API_KEYS is empty — every request is accepted. "
                       "Set it before exposing this server.", extra={"event": "insecure_default"})
    elif "dev-local-key" in s.api_keys:
        logger.warning("AUTH using the default 'dev-local-key' — change ONBOARDING_API_KEYS for any shared/remote use.",
                       extra={"event": "insecure_default"})
    if not s.allowed_roots:
        logger.warning("FS UNRESTRICTED: ONBOARDING_ALLOWED_ROOTS is empty — any local path can be ingested/briefed. "
                       "Set it to confine the agents (least privilege) on a shared host.",
                       extra={"event": "insecure_default"})


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    index = _WEB_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    # never cache the single-page UI, so edits always show on a plain refresh
    return HTMLResponse(index.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/health")
async def health() -> dict[str, Any]:
    s = get_settings()
    return {"status": "ok", "agent_id": AGENT_ID, "version": __version__, "backend": s.backend}


@app.get("/v1/agents")
async def list_agents() -> dict[str, Any]:
    from onboarding_brain.agents import list_agents as _list
    return {"agents": _list(get_settings())}


@app.post("/v1/agents/{agent_id}/run")
async def run_agent(agent_id: str, request: Request, _: str = Depends(require_auth)):
    from onboarding_brain.agents import run_agent as _run
    settings = get_settings()
    model = _parse_body(await request.body(), OnboardingRequest, settings.max_request_bytes)
    try:
        # threadpool: repo scan + LLM call are blocking; keep the event loop free
        resp = await run_in_threadpool(_run, agent_id, model.model_dump(), settings=settings)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown_agent:{agent_id}")
    except RepoAccessError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(content=resp)


def _parse_body(raw: bytes, model_cls, max_bytes: int):
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="request_too_large")
    try:
        return model_cls.model_validate(json.loads(raw))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid_request: {exc}")


@app.get("/v1/namespaces")
async def namespaces(_: str = Depends(require_auth)) -> dict[str, Any]:
    return {"namespaces": get_store(get_settings()).list_namespaces()}


@app.post("/v1/ingest")
async def ingest(request: Request, _: str = Depends(require_auth)):
    settings = get_settings()
    model = _parse_body(await request.body(), IngestRequest, settings.max_request_bytes)
    # serialize only same-repo ingests; different repos run concurrently
    lock_key = (model.namespace or model.clone_url or model.repo_path or "").strip().lower()
    lock = _ingest_lock_for(lock_key)

    def _ingest_locked():
        with lock:
            return ingest_repo(model, settings=settings)

    try:
        resp = await run_in_threadpool(_ingest_locked)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (ImportError, ModuleNotFoundError) as exc:
        # dense/hybrid backend needs fastembed; missing dep must not 500
        raise HTTPException(
            status_code=503,
            detail=("semantic backend unavailable: install fastembed "
                    "(pip install fastembed) or set ONBOARDING_VECTOR_BACKEND=tfidf. "
                    f"[{exc}]"))
    except RuntimeError as exc:
        # e.g. embedding model can't download offline, provider misconfig
        raise HTTPException(status_code=503, detail=f"ingest_unavailable: {exc}")
    return JSONResponse(content=resp.model_dump(mode="json"))


@app.post("/v1/resync/{namespace}")
async def resync(namespace: str, request: Request, _: str = Depends(require_auth)) -> JSONResponse:
    """Re-pull an already-indexed repo and rebuild its index (new commits + code).
    Body (optional): {"clone_token": "..."} to supply a token for a private repo
    on the fly; otherwise ONBOARDING_BITBUCKET_TOKEN is used. Local/public repos
    need no token. Runs in the background; poll /v1/briefing to see it refresh."""
    settings = get_settings()
    raw = await request.body()
    token = ""
    if raw:
        try:
            token = str((json.loads(raw) or {}).get("clone_token", "") or "")
        except Exception:
            token = ""
    from onboarding_brain.kt.ingest import resync_namespace
    try:
        result = await run_in_threadpool(resync_namespace, namespace, clone_token=token, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(content=result)


@app.get("/v1/briefing/{namespace}")
async def get_briefing(namespace: str, _: str = Depends(require_auth)) -> dict[str, Any]:
    """Poll for a background briefing. When ready, returns the briefing fields at
    the TOP LEVEL alongside status:
      {status:'ready', overview, key_features, folder_map, setup_steps,
       recent_work, owners, glossary, starter_questions, validation_status}
    Otherwise returns {status:'running'|'pending'|'failed', error?}."""
    ns = slugify(namespace)
    brief_path = get_store(get_settings()).ns_dir(ns) / "briefing.json"
    if brief_path.is_file():
        try:
            from onboarding_brain.contract import OnboardingResponse
            briefing = OnboardingResponse.model_validate_json(brief_path.read_text(encoding="utf-8"))
            if briefing.overview.answer:
                from onboarding_brain.kt.ingest import _starter_questions
                return {
                    "status": "ready",
                    "overview": briefing.overview.model_dump(),
                    "key_features": [f.model_dump() for f in briefing.key_features],
                    "folder_map": [f.model_dump() for f in briefing.folder_map],
                    "setup_steps": briefing.setup_steps.model_dump(),
                    "recent_work": briefing.recent_work.model_dump(),
                    "owners": [o.model_dump() for o in briefing.owners],
                    "glossary": [g.model_dump() for g in briefing.glossary],
                    "starter_questions": _starter_questions(briefing),
                    "validation_status": briefing.validation_status,
                }
        except Exception:
            pass
    from onboarding_brain.kt.ingest import _BRIEFING_STARTED, briefing_max_age
    with _BRIEFING_JOBS_LOCK:
        running = namespace in _BRIEFING_JOBS or ns in _BRIEFING_JOBS
        err = _BRIEFING_ERRORS.get(namespace) or _BRIEFING_ERRORS.get(ns)
        started = _BRIEFING_STARTED.get(namespace) or _BRIEFING_STARTED.get(ns)
    if err and not running:
        return {"status": "failed", "error": err}
    if running and started and (time.time() - started) > briefing_max_age(get_settings()):
        # the job is still alive but has run far past its bound — treat as hung
        # so the UI stops polling instead of spinning forever
        return {"status": "failed", "error": "briefing timed out (the model took too long to respond)"}
    return {"status": "running" if running else "pending"}


@app.post("/v1/ask")
async def ask_endpoint(request: Request, _: str = Depends(require_auth)):
    settings = get_settings()
    model = _parse_body(await request.body(), AskRequest, settings.max_request_bytes)
    try:
        resp = await run_in_threadpool(kt_ask, model, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (RuntimeError, ImportError, ModuleNotFoundError) as exc:
        # provider couldn't be constructed (missing key/package) — clear 503,
        # not an opaque 500
        raise HTTPException(
            status_code=503,
            detail=f"LLM backend unavailable: {exc}")
    return JSONResponse(content=resp.model_dump(mode="json"))


@app.post("/v1/ask/stream")
async def ask_stream_endpoint(request: Request, _: str = Depends(require_auth)):
    """SSE endpoint — streams real agent progress events then the final answer."""
    settings = get_settings()
    model = _parse_body(await request.body(), AskRequest, settings.max_request_bytes)

    event_queue: Queue = Queue()
    result_holder: dict = {}

    def _run() -> None:
        try:
            resp = kt_ask(model, settings=settings,
                          event_callback=lambda e: event_queue.put(e))
            result_holder["resp"] = resp
        except ValueError as exc:
            result_holder["err"] = {"code": 400, "message": str(exc)}
        except (RuntimeError, ImportError, ModuleNotFoundError) as exc:
            result_holder["err"] = {"code": 503, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            result_holder["err"] = {"code": 500, "message": str(exc)}
        finally:
            event_queue.put({"type": "_done_sentinel"})

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    async def _generate():
        ping = 0
        while True:
            await asyncio.sleep(0.04)
            ping += 1
            drained = False
            while True:
                try:
                    ev = event_queue.get_nowait()
                except Empty:
                    break
                if ev.get("type") == "_done_sentinel":
                    err = result_holder.get("err")
                    if err:
                        yield f"data: {json.dumps({'type': 'error', **err})}\n\n"
                    else:
                        resp = result_holder.get("resp")
                        if resp:
                            payload = resp.model_dump(mode="json")
                            payload["type"] = "done"
                            yield f"data: {json.dumps(payload)}\n\n"
                    worker.join(timeout=2)
                    return
                yield f"data: {json.dumps(ev)}\n\n"
                drained = True
            if not drained and ping % 50 == 0:
                yield ": ping\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.get("/v1/gaps/{namespace}")
async def gaps(namespace: str, _: str = Depends(require_auth)) -> dict[str, Any]:
    from onboarding_brain.kt.knowledge import detect_gaps
    try:
        return {"namespace": namespace, "gaps": detect_gaps(namespace, settings=get_settings())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/v1/tour/{namespace}")
async def tour(namespace: str, refresh: bool = False,
               _: str = Depends(require_auth)) -> JSONResponse:
    """Guided Codebase Tour — an ordered learning path of real files for new joiners.
    Stops carry an LLM one-line insight (narration); since that is expensive it is
    cached to tour.json and reused. Pass ?refresh=true to regenerate.
    Response shape is the TourResponse contract (contract.py)."""
    from onboarding_brain.contract import TourResponse
    from onboarding_brain.kt.tour import build_tour, load_cached_tour, save_cached_tour
    settings = get_settings()

    # serve a narrated cache when available (unless the caller forces a refresh)
    if not refresh:
        cached = load_cached_tour(namespace, settings=settings)
        if cached and cached.get("narrated"):
            try:
                return JSONResponse(content=TourResponse.model_validate(cached).model_dump(mode="json"))
            except Exception:
                pass  # stale/invalid cache shape — fall through and rebuild

    try:
        raw = await run_in_threadpool(build_tour, namespace, narrate=True, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if raw.get("narrated"):  # only cache once narration succeeded
        await run_in_threadpool(save_cached_tour, namespace, raw, settings=settings)
    # validate the agent's output against the published contract before returning
    return JSONResponse(content=TourResponse.model_validate(raw).model_dump(mode="json"))


@app.get("/v1/file/{namespace}")
async def get_file(namespace: str, path: str, _: str = Depends(require_auth)) -> dict[str, Any]:
    """Return a file's FULL indexed text (stitched from its chunks) so the UI can
    show the whole file when a connection node is clicked. `path` is a query param."""
    store = get_store(get_settings())
    f = store.full_file(slugify(namespace), path)
    if not f:
        raise HTTPException(status_code=404, detail=f"file not indexed: {path}")
    return f


@app.get("/v1/walkthrough/{namespace}")
async def get_walkthrough(namespace: str, _: str = Depends(require_auth)) -> dict[str, Any]:
    """Poll the full project walkthrough. Returns {status:'ready', ...WalkthroughResponse}
    when cached, else {status:'generating'} or {status:'absent'} (POST to start one)."""
    from onboarding_brain.contract import WalkthroughResponse
    from onboarding_brain.kt.walkthrough import load_cached_walkthrough, walkthrough_running
    settings = get_settings()
    cached = load_cached_walkthrough(namespace, settings=settings)
    if cached:
        # a cache made offline (structural) is stale once an LLM backend is set —
        # treat it as absent so the client regenerates a full narrative version
        wt_backend = settings.walkthrough_backend or settings.backend
        stale = cached.get("generated_with") == "structural" and wt_backend != "mock"
        if not stale:
            try:
                doc = WalkthroughResponse.model_validate(cached).model_dump(mode="json")
                return {"status": "ready", **doc}
            except Exception:
                pass
    return {"status": "generating" if walkthrough_running(namespace) else "absent"}


@app.post("/v1/walkthrough/{namespace}")
async def start_walkthrough(namespace: str, _: str = Depends(require_auth)) -> dict[str, Any]:
    """Kick off (or restart) background generation of the full project walkthrough."""
    from onboarding_brain.kt.store import slugify
    from onboarding_brain.kt.walkthrough import fire_walkthrough_background
    settings = get_settings()
    if not get_store(settings).exists(slugify(namespace)):
        raise HTTPException(status_code=400, detail=f"namespace not indexed: {namespace}")
    started = fire_walkthrough_background(namespace, settings=settings)
    return {"status": "generating", "already_running": not started}


@app.get("/v1/annotations/{namespace}")
async def annotations_list(namespace: str, _: str = Depends(require_auth)) -> dict[str, Any]:
    from onboarding_brain.kt.knowledge import load_annotations
    try:
        return {"namespace": namespace, "annotations": load_annotations(namespace, settings=get_settings())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/v1/annotations/{namespace}")
async def annotations_save(namespace: str, request: Request, _: str = Depends(require_auth)):
    from onboarding_brain.kt.knowledge import save_annotation
    settings = get_settings()
    body = _parse_body(await request.body(), AnnotateRequest, settings.max_request_bytes)
    try:
        items = save_annotation(namespace, body.file, body.answer, symbol=body.symbol, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"namespace": namespace, "annotations": items}


@app.get("/v1/chat/{namespace}")
async def chat_history(namespace: str, _: str = Depends(require_auth)) -> dict[str, Any]:
    try:
        return {"namespace": namespace, "turns": load_chat_history(namespace, settings=get_settings())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/v1/chat/{namespace}")
async def chat_clear(namespace: str, _: str = Depends(require_auth)) -> dict[str, Any]:
    try:
        clear_chat_history(namespace, settings=get_settings())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"namespace": namespace, "cleared": True}


@app.exception_handler(HTTPException)
async def http_exc(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    # attach a correlation id so a user-reported 500 can be tied to the log line
    from onboarding_brain.trace import new_trace_id
    err_id = new_trace_id()
    logger.exception("unhandled_error error_id=%s", err_id, extra={"event": "unhandled_error", "error_id": err_id})
    return JSONResponse(status_code=500, content={"error": "internal_error", "error_id": err_id})
