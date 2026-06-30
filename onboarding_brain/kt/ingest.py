"""Ingestion: walk a repo, chunk it, build the vector index, and persist it."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from .. import AGENT_ID
from ..config import Settings, get_settings
from ..contract import IngestRequest, IngestResponse, Trace
from ..onboarding import _check_allowed
from ..trace import append_trace, log_event, logger, new_trace_id, timed
from .chunker import iter_chunks
from .store import get_store, slugify

_CLONE_TIMEOUT_S = 300  # 5 min max for a git clone

_RESYNC_LOCK = threading.Lock()



def _build_auth_header(token: str, clone_url: str) -> str | None:
    """Return a 'Authorization: Basic …' header value, or None for public repos.

    Accepts two token formats:
      - 'username:token'  — used as-is for Basic auth (email OK, no URL encoding)
      - bare token        — prefixes x-token-auth (Bitbucket) or x-access-token (GitHub/GitLab)

    Using an HTTP header avoids ALL URL-encoding issues with @ in email addresses.
    """
    if not token:
        return None
    colon = token.find(":")
    if colon > -1:
        user, passwd = token[:colon], token[colon + 1:]
    else:
        user = "x-token-auth" if "bitbucket.org" in clone_url else "x-access-token"
        passwd = token
    encoded = base64.b64encode(f"{user}:{passwd}".encode()).decode()
    return f"Authorization: Basic {encoded}"


def _namespace_from_clone_url(clone_url: str) -> str:
    """Derive the repo name from a git URL for the index namespace.

    Cloned repos are checked out into a temp dir literally named 'repo', so the
    on-disk folder name is useless — the real name lives in the URL:
      https://github.com/org/my-repo.git   -> my-repo
      git@github.com:org/my-repo.git        -> my-repo
      https://host/team/sub/some.repo/      -> some.repo
    Strips a trailing '/', any query/fragment, and the '.git' suffix.
    """
    url = (clone_url or "").strip().rstrip("/")
    if not url:
        return ""
    url = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    # last path segment, then handle scp-style "git@host:org/repo" (':' separator)
    tail = url.rsplit("/", 1)[-1]
    tail = tail.rsplit(":", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[:-4]
    return tail.strip()


# Track running re-syncs so repeated clicks collapse into one rebuild.
_RESYNC_JOBS: dict[str, threading.Thread] = {}


def _read_ns_meta(store, ns: str) -> dict:
    try:
        return json.loads((store.ns_dir(ns) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def resync_namespace(namespace: str, *, clone_token: str = "",
                     settings: Optional[Settings] = None) -> dict:
    """Re-pull an already-indexed repo and rebuild its index in the background
    so new commits + code show up. Source is taken from what was captured at
    first ingest:
      - cloned repo  -> re-clone (needs a token for PRIVATE repos: pass one, or
                        set ONBOARDING_BITBUCKET_TOKEN; public repos need none)
      - local repo   -> re-read the local path (no token; git log is live)
    Returns immediately; the rebuild runs on a daemon thread."""
    settings = settings or get_settings()
    store = get_store(settings)
    ns = slugify(namespace)
    if not store.exists(ns):
        raise ValueError(f"namespace not indexed: {ns}")

    meta = _read_ns_meta(store, ns)
    clone_url = meta.get("clone_url") or ""
    repo_path = meta.get("repo_path") or ""
    if clone_url:
        ireq = IngestRequest(clone_url=clone_url, clone_token=(clone_token or settings.bitbucket_token),
                             namespace=ns, rebuild=True)
        mode = "clone"
    elif repo_path and Path(repo_path).is_dir():
        ireq = IngestRequest(repo_path=repo_path, namespace=ns, rebuild=True)
        mode = "local"
    else:
        raise ValueError("cannot re-sync: the original source is unavailable — "
                         "re-ingest this repo from the Clone or Local tab")

    with _RESYNC_LOCK:
        existing = _RESYNC_JOBS.get(ns)
        if existing and existing.is_alive():
            return {"namespace": ns, "status": "already_resyncing", "mode": mode}

    def _run() -> None:
        try:
            ingest_repo(ireq, settings=settings)
            logger.info("resync_done namespace=%s", ns)
        except Exception:
            logger.exception("resync_failed namespace=%s", ns)
        finally:
            with _RESYNC_LOCK:
                _RESYNC_JOBS.pop(ns, None)

    t = threading.Thread(target=_run, daemon=True, name=f"resync-{ns}")
    with _RESYNC_LOCK:
        _RESYNC_JOBS[ns] = t
    t.start()
    return {"namespace": ns, "status": "resyncing", "mode": mode}


def _clone_repo(clone_url: str, token: str = "", *, interactive: bool = True) -> tuple[Path, Path]:
    """Clone *clone_url* into a temp dir. Returns (repo_path, tmpdir_to_cleanup).

    Auth modes:
      - token provided        : inject via http.extraHeader (Basic auth, no URL-encoding issues)
      - no token, interactive : let Git Credential Manager open a browser for OAuth — works
                                when the server runs on the user's own machine (local setup)
      - no token, headless    : suppress prompts and fail fast (no browser to pop), so a
                                private clone errors in ~1 min instead of hanging for 5
    """
    tmp = Path(tempfile.mkdtemp(prefix="cortex_clone_"))
    env = os.environ.copy()
    cmd = ["git", "clone", "--depth=1", "--single-branch"]
    timeout = _CLONE_TIMEOUT_S

    if token:
        # explicit credentials — suppress all interactive prompts, use header auth
        env["GIT_TERMINAL_PROMPT"] = "0"
        auth_header = _build_auth_header(token, clone_url)
        cmd += ["-c", f"http.extraHeader={auth_header}"]
        run_kwargs: dict = {"capture_output": True, "text": True}
    elif interactive:
        # no token — let GCM open the browser (OAuth popup)
        # do NOT set GIT_TERMINAL_PROMPT=0 and do NOT capture stderr so GCM can spawn the browser
        run_kwargs = {}
    else:
        # headless: no browser available — fail fast rather than hang on a prompt
        env["GIT_TERMINAL_PROMPT"] = "0"
        run_kwargs = {"capture_output": True, "text": True}
        timeout = 60

    cmd += [clone_url, str(tmp / "repo")]

    try:
        subprocess.run(cmd, check=True, timeout=timeout, env=env, **run_kwargs)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        stderr = (getattr(exc, "stderr", None) or "")[:400]
        raise ValueError(f"git clone failed: {stderr or 'authentication failed or repo not found'}") from exc
    except FileNotFoundError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError("git is not installed or not on PATH. Install Git and retry.")
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        hint = ("provide a clone_token for private repos"
                if not interactive else "complete the browser login or check your network")
        raise ValueError(f"git clone timed out — {hint}.")
    return tmp / "repo", tmp


def _cached_ingest_response(store, namespace: str, settings: Settings, trace_id: str) -> IngestResponse:
    """Serve an already-indexed repo from its persisted index WITHOUT cloning —
    the 'track record' that avoids re-cloning a repo we've seen before."""
    meta = _read_ns_meta(store, namespace)
    files = int(meta.get("n_files", 0) or 0)
    trace = Trace(
        trace_id=trace_id, agent_id=AGENT_ID, model_used=settings.model_used,
        duration_ms=0, strategy="index",
        repo_path=meta.get("repo_path", ""),
        files_scanned=files,
        grounding={"namespace": namespace, "chunks_indexed": 0, "already_indexed": True},
    )
    log_event("ingest_end", trace_id)
    return IngestResponse(
        namespace=namespace, repo_path=meta.get("repo_path", ""), files_indexed=files,
        chunks_indexed=0, already_indexed=True,
        validation_status="passed",
        trace=trace,
    )


def ingest_repo(request: IngestRequest, *, settings: Optional[Settings] = None) -> IngestResponse:
    settings = settings or get_settings()
    store = get_store(settings)

    # validate an explicit namespace up front so a junk override (e.g. "!!!")
    # fails fast instead of silently slugifying to the "repo" fallback
    if request.namespace and not any(ch.isalnum() for ch in request.namespace):
        raise ValueError("namespace must contain at least one letter or digit")

    # --- derive the namespace WITHOUT cloning, so an already-indexed repo is
    #     never re-cloned (track record = the persisted index + its clone_url) ---
    local_repo: Optional[Path] = None
    if request.clone_url:
        namespace = (slugify(request.namespace) if request.namespace
                     else slugify(_namespace_from_clone_url(request.clone_url) or "repo"))
    elif request.repo_path:
        local_repo = Path(request.repo_path).expanduser().resolve()
        namespace = slugify(request.namespace) if request.namespace else slugify(local_repo.name)
    else:
        raise ValueError("Provide either repo_path (local directory) or clone_url (git URL).")

    trace_id = new_trace_id()
    log_event("ingest_start", trace_id)

    # Track record: already indexed and no rebuild requested → serve cache, no clone.
    if store.exists(namespace) and not request.rebuild:
        return _cached_ingest_response(store, namespace, settings, trace_id)

    # --- not cached (or rebuild=True): resolve the real repo, then index ---
    tmpdir: Optional[Path] = None
    if request.clone_url:
        repo, tmpdir = _clone_repo(request.clone_url, token=request.clone_token,
                                   interactive=settings.clone_interactive)
    else:
        repo = local_repo

    if not repo.is_dir():
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise ValueError(f"not a directory: {repo}")
    try:
        _check_allowed(str(repo), settings)  # must run BEFORE anything is persisted
    except Exception:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    try:
        with timed() as t:
            chunks = list(iter_chunks(
                repo, chunk_chars=settings.chunk_chars, overlap=settings.chunk_overlap,
                max_files=settings.max_ingest_files, max_bytes=settings.max_ingest_file_bytes,
            ))
            files = len({c["metadata"]["path"] for c in chunks})
            # business-context harvest: fold i18n labels into searchable text so
            # business-vocabulary questions reach cryptic code. (Commit subjects
            # are NOT folded into code chunks — that broke dense-embedding reuse
            # on re-sync; history is indexed as its own chunks below instead.)
            from .enrich import build_commit_chunks, build_i18n_labels, enrich_chunks
            enrich_chunks(chunks, build_i18n_labels(repo))
            # index recent git commits as standalone searchable chunks so Cortex
            # can answer "what changed in auth last week?" directly from history
            commit_chunks = build_commit_chunks(repo, max_commits=settings.max_commit_chunks)
            if commit_chunks:
                chunks = chunks + commit_chunks
            store.index(namespace, chunks, {"repo_path": str(repo), "n_files": files,
                                            "clone_url": request.clone_url or "",
                                            "indexed_at": time.time()})
            chunks_indexed = len(chunks)
        is_git_repo = (repo / ".git").exists()
    finally:
        # guarantee the temp clone dir is removed on EVERY exit path
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    trace = Trace(
        trace_id=trace_id, agent_id=AGENT_ID, model_used=settings.model_used,
        duration_ms=t["ms"], strategy="index",
        repo_path=str(repo),
        is_git_repo=is_git_repo,
        files_scanned=files,
        grounding={"namespace": namespace, "chunks_indexed": chunks_indexed, "already_indexed": False},
    )
    append_trace({"trace_id": trace_id, "event": "ingest", "namespace": namespace,
                  "repo_path": str(repo), "chunks_indexed": chunks_indexed, "files": files,
                  "already_indexed": False, "duration_ms": t["ms"]})
    log_event("ingest_end", trace_id)

    return IngestResponse(
        namespace=namespace, repo_path=str(repo), files_indexed=files,
        chunks_indexed=chunks_indexed, already_indexed=False,
        validation_status="passed",
        trace=trace,
    )
