"""Onboarding Brain orchestration.

gather repo context (tools) -> LLM briefing (grounded prompt) -> citation check
-> traced response. Always returns a valid response; failures degrade safely.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import AGENT_ID
from .config import Settings, get_settings
from .contract import (
    FeatureInfo,
    FolderInfo,
    GlossaryItem,
    OnboardingRequest,
    OnboardingResponse,
    OwnerInfo,
    SetupSteps,
    Sourced,
    Trace,
)
from .grounding import check_sources
from .prompts import (
    BRIEFING_A_SYSTEM, BRIEFING_B_SYSTEM, BRIEFING_C_SYSTEM,
    SYSTEM_PROMPT,
    build_prompt_a, build_prompt_b, build_prompt_c, build_user_prompt,
)
from .providers import LLMError, get_provider
from .providers.base import LLMProvider
from .repo_reader import gather_repo_context
from .trace import append_trace, log_event, new_trace_id, timed

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class RepoAccessError(ValueError):
    pass


def _parse_json(text: str) -> dict:
    text = text.strip()
    # strip reasoning-model <think> blocks before locating the JSON object
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(text)
        if m:
            return json.loads(m.group(0))
        raise


def _check_allowed(repo_path: str, settings: Settings) -> None:
    if not settings.allowed_roots:
        return
    resolved = Path(repo_path).expanduser().resolve()
    for root in settings.allowed_roots:
        try:
            resolved.relative_to(Path(root).expanduser().resolve())
            return
        except ValueError:
            continue
    raise RepoAccessError(f"repo_path is outside ONBOARDING_ALLOWED_ROOTS: {resolved}")


def _briefing_parallel(
    provider: LLMProvider,
    ctx: dict,
    settings: Settings,
    errors: list,
    strategy: str,
) -> tuple[dict, list, str]:
    """Run three focused LLM calls in parallel and merge their JSON results.

    Each call handles ~2-3 questions with a focused subset of the repo context
    (~1/3 the token count of the monolithic prompt). Wall-clock time is
    max(call_A, call_B, call_C) instead of the sum — typically 3-5x faster.

    Falls back to the single monolithic call if all three partial calls fail.
    """
    CALLS = [
        ("A", BRIEFING_A_SYSTEM, build_prompt_a(ctx)),
        ("B", BRIEFING_B_SYSTEM, build_prompt_b(ctx)),
        ("C", BRIEFING_C_SYSTEM, build_prompt_c(ctx)),
    ]

    results: dict[str, dict] = {}
    call_errors: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as exe:
        futures = {
            exe.submit(provider.complete, sys_p, user_p): label
            for label, sys_p, user_p in CALLS
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                results[label] = _parse_json(future.result().text)
            except Exception as exc:
                call_errors.append(f"briefing_{label}_failed: {str(exc)[:120]}")

    # Merge with intent: each call owns its own fields.
    # Never let call C's "not found" overview (no readme in its context)
    # overwrite call A's correct answer.
    _OWNED: dict[str, list[str]] = {
        "A": ["overview", "key_features", "folder_map"],
        "B": ["setup_steps", "glossary"],
        "C": ["recent_work", "owners"],
    }
    merged: dict = {}
    for label, keys in _OWNED.items():
        for k in keys:
            val = results.get(label, {}).get(k)
            if val is not None:
                merged[k] = val

    _ov = merged.get("overview")
    if not merged or not (isinstance(_ov, dict) and _ov.get("answer")):
        # all three partial calls failed — fall back to the monolithic prompt
        errors.extend(call_errors)
        for attempt in (1, 2):
            try:
                result = provider.complete(SYSTEM_PROMPT, build_user_prompt(ctx, settings.context_budget_chars))
                return _parse_json(result.text), errors, strategy
            except (LLMError, ValueError) as exc:
                if attempt == 2:
                    errors.append(f"generation_failed: {exc}")
                    strategy = "degraded"
        return {}, errors, strategy

    if call_errors:
        errors.extend(call_errors)
        if len(call_errors) == len(CALLS):
            strategy = "degraded"
        else:
            strategy = "warning" if strategy != "degraded" else strategy

    return merged, errors, strategy


def _build_response(parsed: dict, ctx: dict, errors: list, strategy: str,
                    trace_id: str, settings: Settings, duration_ms: int) -> OnboardingResponse:
    """Map raw LLM JSON + context into an OnboardingResponse. Shared by sync and async paths."""
    # the LLM may return a scalar where we expect a dict (e.g. {"overview": "..."})
    # — coerce to {} so .get() never raises AttributeError on a str
    def _d(key: str) -> dict:
        v = parsed.get(key)
        return v if isinstance(v, dict) else {}

    ov = _d("overview")
    overview = Sourced(answer=str(ov.get("answer", "")).strip(),
                       sources=[str(s) for s in (ov.get("sources") or [])])

    key_features = [
        FeatureInfo(feature=str(f.get("feature", "")).strip(),
                    detail=str(f.get("detail", "")).strip(),
                    sources=[str(s) for s in (f.get("sources") or [])])
        for f in (parsed.get("key_features") or []) if isinstance(f, dict) and f.get("feature")
    ]
    folder_map = [
        FolderInfo(folder=str(f.get("folder", "")).strip(),
                   purpose=str(f.get("purpose", "")).strip(),
                   sources=[str(s) for s in (f.get("sources") or [])])
        for f in (parsed.get("folder_map") or []) if isinstance(f, dict) and f.get("folder")
    ]
    ss = _d("setup_steps")
    setup_steps = SetupSteps(steps=[str(s) for s in (ss.get("steps") or [])],
                             sources=[str(s) for s in (ss.get("sources") or [])])
    rw = _d("recent_work")
    recent_work = Sourced(answer=str(rw.get("answer", "")).strip(),
                          sources=[str(s) for s in (rw.get("sources") or [])])
    owners = [
        OwnerInfo(area=str(o.get("area", "")).strip(), owner=str(o.get("owner", "")).strip(),
                  sources=[str(s) for s in (o.get("sources") or [])])
        for o in (parsed.get("owners") or []) if isinstance(o, dict) and o.get("area")
    ]
    glossary = [
        GlossaryItem(term=str(g.get("term", "")).strip(), meaning=str(g.get("meaning", "")).strip(),
                     sources=[str(s) for s in (g.get("sources") or [])])
        for g in (parsed.get("glossary") or []) if isinstance(g, dict) and g.get("term")
    ]

    response_dict = {
        "overview": overview.model_dump(), "key_features": [f.model_dump() for f in key_features],
        "folder_map": [f.model_dump() for f in folder_map],
        "setup_steps": setup_steps.model_dump(), "recent_work": recent_work.model_dump(),
        "owners": [o.model_dump() for o in owners], "glossary": [g.model_dump() for g in glossary],
    }
    grounding = check_sources(response_dict, ctx.get("available_sources", []))
    status = grounding["validation_status"]
    if strategy == "degraded":
        status = "failed" if not overview.answer else "warning"

    trace = Trace(
        trace_id=trace_id, agent_id=AGENT_ID, model_used=settings.model_used,
        duration_ms=duration_ms, strategy=strategy, repo_path=ctx.get("repo_path", ""),
        is_git_repo=bool(ctx.get("is_git_repo")), files_scanned=int(ctx.get("file_count", 0)),
        errors=errors, grounding=grounding,
    )
    append_trace({
        "trace_id": trace_id, "agent_id": AGENT_ID, "repo_path": ctx.get("repo_path"),
        "model_used": settings.model_used, "duration_ms": duration_ms, "strategy": strategy,
        "files_scanned": ctx.get("file_count"), "validation_status": status,
        "grounding": grounding, "errors": errors,
    })
    return OnboardingResponse(
        overview=overview, key_features=key_features, folder_map=folder_map,
        setup_steps=setup_steps, recent_work=recent_work, owners=owners, glossary=glossary,
        validation_status=status, trace=trace,
    )


def generate_briefing_from_ctx(
    ctx: dict,
    *,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> OnboardingResponse:
    """Run the LLM briefing from a pre-gathered context dict.

    Callers that already ran gather_repo_context() (e.g. background threads)
    use this directly so we never re-read files that may have been cleaned up.
    """
    settings = settings or get_settings()
    provider = provider or get_provider(settings)
    trace_id = new_trace_id()
    errors: list[str] = []
    strategy = "mock" if settings.backend == "mock" else "llm"

    log_event("onboarding_start", trace_id)
    parsed: dict = {}
    with timed() as t:
        parsed, errors, strategy = _briefing_parallel(provider, ctx, settings, errors, strategy)
    duration_ms = t["ms"]

    result = _build_response(parsed, ctx, errors, strategy, trace_id, settings, duration_ms)
    log_event("onboarding_end", trace_id)
    return result


def generate_briefing(
    request: OnboardingRequest,
    *,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> OnboardingResponse:
    settings = settings or get_settings()
    _check_allowed(request.repo_path, settings)
    ctx = gather_repo_context(request.repo_path)
    if ctx.get("error"):
        raise RepoAccessError(ctx["error"])
    return generate_briefing_from_ctx(ctx, settings=settings, provider=provider)
