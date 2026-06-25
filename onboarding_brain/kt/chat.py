"""RAG chat: retrieve relevant chunks -> grounded LLM answer -> cited sources.

Trust layer: the model is told to answer ONLY from retrieved snippets and to
say "I couldn't find this in the indexed code" otherwise; we then verify that
every cited source was actually in the retrieved set. Works offline too — the
mock backend returns an extractive answer from the top snippets.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

from .. import AGENT_ID
from ..config import Settings, get_settings
from ..contract import AskRequest, AskResponse, Source, Trace
from ..prompts import (
    CHAT_SYSTEM_PROMPT,
    CONDENSE_SYSTEM_PROMPT,
    build_chat_prompt,
    build_condense_prompt,
)
from ..providers import LLMError, get_provider
from ..providers.base import LLMProvider
from ..trace import append_trace, log_event, new_trace_id, timed
from .chat_store import get_chat_store
from .store import _query_terms, get_store, slugify, split_identifiers

CHAT_AGENT_ID = "kt-brain"
NOT_FOUND = "I couldn't find this in the indexed code."
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")


def _norm_citation(s: str) -> str:
    """Models often cite "path.py:10-42" because context headers show line
    ranges; grounding compares file paths, so strip the suffix."""
    return _LINE_SUFFIX.sub("", s.strip())


def classify_citations(used: list[str], retrieved: set[str], known: set[str]
                       ) -> tuple[list[str], list[str]]:
    """Split citations that weren't retrieved into (hallucinated, inferred).
    A cited file that exists in the index (even extension-less, as TS/JS
    imports name them) was inferred from retrieved code — real, just not
    shown. Only citations matching NO indexed file are hallucinations."""
    known_stems = {p.rsplit(".", 1)[0] for p in known if "." in p}
    hallucinated, inferred = [], []
    for s in set(used):
        if s in retrieved:
            continue
        if s in known or s in known_stems:
            inferred.append(s)
        else:
            hallucinated.append(s)
    return sorted(hallucinated), sorted(inferred)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse(text: str) -> dict:
    text = text.strip()
    # reasoning models emit <think>…</think> before the answer — strip it so the
    # JSON extractor doesn't trip over braces inside the reasoning trace
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"answer": text, "used_sources": []}


def _sources_from_chunks(chunks: list[dict]) -> list[Source]:
    out = []
    for c in chunks:
        m = c.get("metadata", {})
        snippet = (c.get("text", "") or "").strip()
        out.append(Source(
            path=m.get("path", ""), language=m.get("language", ""),
            line_start=int(m.get("line_start", 0)), line_end=int(m.get("line_end", 0)),
            score=float(c.get("score", 0.0)), snippet=snippet[:400],
            symbol=str(m.get("symbol", "")),
        ))
    return out


_FOLLOWUP_REF = re.compile(r"\b(it|that|this|these|those|they|them|same|there|one)\b", re.IGNORECASE)

# the "General note" (model's own engineering knowledge) is only welcome on
# setup/run/dependency questions — never on what/how/where code questions
_SETUP_Q = re.compile(
    r"\b(set ?up|install|run|runs|running|build|compile|start|launch|serve|deploy|"
    r"depend|dependenc|prerequisit|requirement|version|node|npm|yarn|environment|"
    r"configure|configuration|getting started)\b", re.IGNORECASE)

# Broad/meta questions are answered badly by raw code chunks — route the
# persisted Day-1 briefing in as the top context snippet instead.
_BROAD_Q = re.compile(
    r"^\s*(hi|hello|hey)\b"
    r"|what (does|is) (this|the) (project|repo|app|codebase)"
    r"|(main|key) features"
    r"|\b(overview|brief|briefing|summar)"
    r"|how (can|do) you help|what can you (do|help)"
    r"|tell me about (this|the) (project|repo|app|codebase)"
    r"|how do i run (it|this)",
    re.IGNORECASE,
)


def _briefing_chunk(store, namespace: str) -> Optional[tuple[dict, set[str]]]:
    """Returns the briefing as a context chunk plus the sources the briefing
    itself cites (README.md, git log, …) — citing those is grounded, since the
    briefing content derives from them."""
    p = store.ns_dir(namespace) / "briefing.json"
    if not p.is_file():
        return None
    try:
        b = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not (b.get("overview") or {}).get("answer"):
        return None  # failed/empty briefing adds nothing — use code chunks
    features = "; ".join(
        f"{f.get('feature', '')} — {f.get('detail', '')}" for f in (b.get("key_features") or [])[:10]
    )
    folders = "; ".join(
        f"{f.get('folder', '')}: {f.get('purpose', '')}" for f in (b.get("folder_map") or [])[:10]
    )
    steps = " | ".join((b.get("setup_steps") or {}).get("steps", [])[:6])
    text = (
        "PROJECT BRIEFING (auto-generated at ingest)\n"
        f"Overview: {(b.get('overview') or {}).get('answer', '')}\n"
        + (f"Key features: {features}\n" if features else "")
        + f"Folders: {folders}\n"
        f"Run locally: {steps}\n"
        f"Recent work: {(b.get('recent_work') or {}).get('answer', '')}"
    )
    cited: set[str] = set()

    def collect(node) -> None:
        if isinstance(node, dict):
            for key, v in node.items():
                if key == "sources" and isinstance(v, list):
                    cited.update(str(x) for x in v)
                else:
                    collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)

    collect(b)
    chunk = {"id": "briefing#0", "score": 1.0, "text": text,
             "metadata": {"path": "project-briefing", "language": "", "symbol": "Day-1 briefing",
                          "line_start": 0, "line_end": 0}}
    return chunk, cited


def filename_candidates(known_paths: set[str], question: str,
                        exclude: set[str], limit: int = 2) -> list[str]:
    """File-locator: when the question names a file by its basename ("the main
    file", "API endpoints"), vector search may never fetch it — a tiny main.ts
    shares no vocabulary with "entry point". Match meaningful question terms
    against indexed file NAMES and return the best hits to pull in directly."""
    qterms = _query_terms(question)
    if not qterms:
        return []
    scored: list[tuple[int, int, str]] = []
    for p in known_paths:
        if p in exclude:
            continue
        base = p.rsplit("/", 1)[-1]
        btokens = set(re.findall(r"[a-z0-9]{3,}", split_identifiers(base)))
        overlap = len(btokens & qterms)
        if overlap:
            scored.append((overlap, -len(p), p))  # most terms, then shortest path
    scored.sort(reverse=True)
    # located files only WIDEN prompt context; they are shown to the user only
    # when the answer actually cites them (the source-gating in ask() enforces
    # this), so a loose filename match here can never leak a file into the UI.
    return [p for _, _, p in scored[:limit]]


# ---------------------------------------------------------------------------
# Conversation memory: each namespace keeps its chat log so context survives
# reloads and the UI can restore past sessions. The backend (JSON files or
# MongoDB) is chosen by settings — see kt/chat_store.py.


def load_chat_history(namespace: str, *, settings: Optional[Settings] = None) -> list[dict]:
    return get_chat_store(settings or get_settings()).load(namespace)


def clear_chat_history(namespace: str, *, settings: Optional[Settings] = None) -> None:
    get_chat_store(settings or get_settings()).clear(namespace)


def _append_history(namespace: str, question: str, answer: str,
                    *, settings: Optional[Settings] = None, extra: Optional[dict] = None) -> None:
    get_chat_store(settings or get_settings()).append(namespace, question, answer, extra=extra)


def _match_annotations(anns: list[dict], query: str) -> list[dict]:
    """Keep team annotations whose file/answer shares a meaningful term with the
    query — so curated notes surface for relevant questions, not every one."""
    if not anns:
        return []
    qterms = _query_terms(query)
    if not qterms:
        return []
    out = []
    for a in anns:
        toks = set(re.findall(r"[a-z0-9]{3,}", split_identifiers(a.get("index_text", ""))))
        if toks & qterms:
            out.append(a)
    return out


def select_relevant(chunks: list[dict], k_min: int, k_max: int) -> list[dict]:
    """Dynamic source count: keep candidates whose score holds up against the
    best hit, instead of a fixed top-k. A focused question has a steep score
    cliff (-> few sources); a broad one decays slowly (-> many).

    Uses a RELATIVE threshold only (≥35% of the top score), never an absolute
    one — score scales differ wildly per backend (TF-IDF cosine ~0.1-0.6,
    hybrid RRF ~0.016 per rank), so any absolute floor that suits one backend
    silently guts another. Backend-specific floors live in each store's
    search(). Falls back to top k_min so we never return zero results."""
    if not chunks:
        return chunks
    if len(chunks) <= k_min:
        return chunks[:k_max]
    top = chunks[0].get("score") or 1e-9
    kept = [c for c in chunks if (c.get("score") or 0.0) >= 0.20 * top]
    if len(kept) < k_min:
        kept = chunks[:k_min]
    return kept[:k_max]


def _expand_neighbors(store, namespace: str, chunks: list[dict],
                      limit: int = 5, max_chars: int = 6000) -> list[dict]:
    """For the strongest hits, splice in the adjacent chunks of the same file
    so the LLM reads the whole surrounding code, not a fragment. Only the
    prompt context widens — the user-facing snippet stays the original hit."""
    out = []
    for i, c in enumerate(chunks):
        if i < limit and "#" in c.get("id", ""):
            m = c["metadata"]
            path, idx = m.get("path", ""), int(m.get("chunk_index", 0))
            parts = [c["text"]]
            l0, l1 = m.get("line_start", 0), m.get("line_end", 0)
            prev = store.get_chunk(namespace, f"{path}#{idx - 1}") if idx > 0 else None
            nxt = store.get_chunk(namespace, f"{path}#{idx + 1}")
            if prev:
                parts.insert(0, prev["text"])
                l0 = prev["metadata"].get("line_start", l0)
            if nxt:
                parts.append(nxt["text"])
                l1 = nxt["metadata"].get("line_end", l1)
            if prev or nxt:
                c = {**c, "context_text": "".join(parts)[:max_chars], "context_lines": [l0, l1]}
        out.append(c)
    return out


def _looks_like_followup(question: str) -> bool:
    """Only short questions leaning on a reference ("which file is that in?")
    need rewriting. Standalone questions must be searched verbatim — an LLM
    rewrite of an already-complete question dilutes retrieval."""
    return len(question.split()) <= 8 and bool(_FOLLOWUP_REF.search(question))


def condense_question_offline(question: str, history: list[dict]) -> str:
    """No-LLM fallback: fold the previous user question's terms into the
    follow-up so retrieval has real keywords instead of a pronoun."""
    if not history or not _looks_like_followup(question):
        return question
    prev = [h.get("content", "") for h in history if h.get("role") == "user"]
    if prev:
        return f"{prev[-1]} {question}"
    return question


def _condense_question(question: str, history: list[dict], provider: LLMProvider,
                       settings: Settings, errors: list[str]) -> str:
    """Rewrite a follow-up into a standalone search query before retrieval.
    Falls back to the offline heuristic, never fails the request."""
    if not history or not _looks_like_followup(question):
        return question
    if settings.backend == "mock":
        return condense_question_offline(question, history)
    try:
        result = provider.complete(CONDENSE_SYSTEM_PROMPT, build_condense_prompt(question, history))
        rewritten = str(_parse(result.text).get("question") or "").strip()
        if rewritten:
            return rewritten
    except (LLMError, ValueError) as exc:
        errors.append(f"condense_failed: {exc}")
    return condense_question_offline(question, history)


def _mock_answer(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    if not chunks:
        return NOT_FOUND, []
    top = chunks[:3]
    lines = [f"Based on the indexed code, the most relevant places are:"]
    for c in top:
        m = c["metadata"]
        first = next((ln.strip() for ln in c["text"].splitlines() if ln.strip()), "")
        lines.append(f"- {m['path']} (lines {m['line_start']}-{m['line_end']}): {first[:120]}")
    lines.append("Open these files for details. (Offline mode: extractive answer; set a Groq key for natural-language answers.)")
    return "\n".join(lines), [c["metadata"]["path"] for c in top]


def ask(request: AskRequest, *, settings: Optional[Settings] = None,
        provider: Optional[LLMProvider] = None) -> AskResponse:
    import dataclasses
    settings = settings or get_settings()
    if not provider:
        if request.backend or request.claude_model:
            if request.claude_model:
                settings = dataclasses.replace(settings, claude_model=request.claude_model)
            provider = get_provider(settings, backend=request.backend or settings.backend)
        else:
            provider = get_provider(settings)
    store = get_store(settings)
    trace_id = new_trace_id()
    errors: list[str] = []
    k = request.top_k or settings.retrieval_top_k
    namespace = slugify(request.namespace)  # same normalization as ingest
    log_event("ask_start", trace_id)

    if not store.exists(namespace):
        raise ValueError(f"namespace not indexed: {namespace}. Ingest the repo first.")

    with timed() as t:
        history = [h.model_dump() for h in request.history]
        search_query = _condense_question(request.question, history, provider, settings, errors)
        fetched = store.search(namespace, search_query, k)
        # dynamic depth: relevance cliff decides how many real hits survive
        chunks = select_relevant(fetched, settings.retrieval_min_k, k)
        # team knowledge (human-curated annotations) injected AFTER relevance
        # selection — their sentinel score (1.0) would otherwise become the
        # cliff anchor and wipe out every real code chunk (RRF scores ~0.03)
        from .knowledge import annotation_chunks
        anns = _match_annotations(annotation_chunks(namespace, settings=settings), search_query)
        if anns:
            ann_ids = {a["id"] for a in anns}
            chunks = (anns + [c for c in chunks if c["id"] not in ann_ids])[:k]
        is_broad = bool(_BROAD_Q.search(request.question))
        briefing_sources: set[str] = set()
        if is_broad:
            from .knowledge import feature_surface
            head: list[dict] = []
            bc = _briefing_chunk(store, namespace)
            if bc:
                chunk, briefing_sources = bc
                head.append(chunk)
            # comprehensive real feature map so broad answers don't miss areas
            fm = feature_surface(namespace, settings=settings)
            if fm:
                head.append(fm)
                briefing_sources.add("feature-map")
            if head:
                chunks = head + chunks[: max(0, k - len(head))]
        # file-locator: pull in files the question names that search missed
        present = {c["metadata"]["path"] for c in chunks}
        known = store.known_paths(namespace)
        cands = filename_candidates(known, request.question, present)
        if is_broad:
            # broad questions also deserve the root README, not just code hits
            readme = next((p for p in sorted(known)
                           if p.lower() in ("readme.md", "readme.rst", "readme.txt", "readme")), None)
            if readme and readme not in present and readme not in cands:
                cands.insert(0, readme)
        located = [store.first_chunk(namespace, cand) for cand in cands]
        located = [c for c in located if c]
        if located:
            # drop the weakest retrieved chunks to stay within k
            chunks = chunks[: max(0, k - len(located))] + located
        retrieved_paths = {c["metadata"]["path"] for c in chunks} | briefing_sources
        # widen the strongest hits with their neighboring code (prompt-only)
        chunks = _expand_neighbors(store, namespace, chunks)

        if settings.backend == "mock":
            answer, used = _mock_answer(request.question, chunks)
            strategy = "mock"
        else:
            strategy = "llm"
            try:
                chat_prompt = build_chat_prompt(request.question, chunks, history,
                                                settings.chat_context_budget_chars)
                result = provider.complete(CHAT_SYSTEM_PROMPT, chat_prompt)
                parsed = _parse(result.text)
                answer = str(parsed.get("answer", "")).strip()
                # If the model returned not-found but we have chunks, retry once
                # with an explicit instruction to use the provided snippets.
                _nf = NOT_FOUND.lower()[:20]
                if answer.lower().startswith(_nf) and chunks:
                    retry_system = CHAT_SYSTEM_PROMPT + (
                        "\n\nCRITICAL OVERRIDE: The context block above contains real code snippets. "
                        "You MUST use them. Write a grounded answer referencing those files now. "
                        "Do NOT output not-found. This is mandatory."
                    )
                    result2 = provider.complete(retry_system, chat_prompt)
                    parsed2 = _parse(result2.text)
                    answer2 = str(parsed2.get("answer", "")).strip()
                    if answer2 and not answer2.lower().startswith(_nf):
                        parsed = parsed2
                        answer = answer2
                        errors.append("not_found_retry:success")
                    else:
                        errors.append("not_found_retry:failed")
                note = str(parsed.get("general_note", "")).strip()
                # only attach general guidance to setup/run questions; never to
                # "what/how/where" code questions (keeps answers repo-grounded)
                if (note and _SETUP_Q.search(request.question)
                        and not answer.lower().startswith(NOT_FOUND.lower()[:20])):
                    answer += f"\n\n**General note (not from the repo):** {note}"
                raw_used = parsed.get("used_sources")
                used = [_norm_citation(str(s)) for s in raw_used] if isinstance(raw_used, list) else []
            except (LLMError, ValueError) as exc:
                errors.append(f"answer_failed: {exc}")
                answer, used, strategy = NOT_FOUND, [], "degraded"

    # grounding: cited sources must be retrieved snippets or, at minimum,
    # real indexed files inferred from them (e.g. named in an import)
    hallucinated, inferred = classify_citations(used, retrieved_paths, store.known_paths(namespace))
    _nf_prefix = NOT_FOUND.lower()[:20]
    is_not_found = answer.strip().lower().startswith(_nf_prefix)
    grounded = not hallucinated and (bool(chunks) or is_not_found)
    status = "passed"
    if hallucinated:
        status = "warning"
    if errors:
        status = "warning"
    if not chunks and not is_not_found:
        status = "warning"

    trace = Trace(
        trace_id=trace_id, agent_id=CHAT_AGENT_ID, model_used=settings.model_used,
        duration_ms=t["ms"], strategy=strategy, errors=errors,
        grounding={"namespace": namespace, "retrieved": len(fetched), "selected": len(chunks),
                   "top_k": k,
                   "hallucinated_sources": hallucinated,
                   **({"inferred_sources": inferred} if inferred else {}),
                   "top_score": chunks[0]["score"] if chunks else 0.0,
                   **({"search_query": search_query} if search_query != request.question else {})},
    )
    append_trace({"trace_id": trace_id, "event": "ask", "namespace": namespace,
                  "question": request.question[:200], "retrieved": len(chunks),
                  "grounded": grounded, "duration_ms": t["ms"]})
    log_event("ask_end", trace_id)

    # ---- Sources shown = STRICTLY the files the answer actually used ----
    # Never dump the retrieval pool. A synthetic context block
    # (briefing / feature-map) appears only when the answer genuinely cited it
    # (broad overview questions); it is excluded from the fallback and wiring.
    _SKIP_PATHS = {"project-briefing", "feature-map", "git-history"}
    used_set = {u for u in used if u in retrieved_paths}
    cited_chunks = [c for c in chunks if c["metadata"]["path"] in used_set]
    if cited_chunks:
        shown = cited_chunks
    elif is_not_found or strategy == "degraded":
        # nothing found / model errored — show nothing rather than noise
        shown = []
    else:
        # a real answer that under-cited: surface only the single strongest
        # real hit as a starting point — never the whole candidate pool
        shown = [c for c in chunks if c["metadata"]["path"] not in _SKIP_PATHS][:1]
    sources = _sources_from_chunks(shown)
    for s in sources:
        s.used = s.path in used_set
    sources.sort(key=lambda s: not s.used)

    # ---- Wiring diagram: built ONLY from the cited files ----
    # Drawing un-cited retrieval candidates manufactured false relationships
    # (the hub-fallback would force-connect unrelated files). Cited-only keeps
    # the diagram strictly scoped to the answer's context.
    wiring = None
    if not is_not_found:
        wpaths = [s.path for s in sources if s.path not in _SKIP_PATHS]
        from .wiring import build_wiring
        try:
            wiring = build_wiring(namespace, wpaths, settings=settings)
        except Exception:
            wiring = None

    # persist the turn AFTER wiring is built, so a refreshed session restores
    # the connectivity diagram and the source snippets exactly as they were shown
    _hist_extra: dict = {}
    if wiring:
        _hist_extra["wiring"] = wiring
    # compact source list — enough for the click-to-view panel after a reload
    src_compact = [
        {"path": s.path, "line_start": s.line_start, "line_end": s.line_end,
         "score": round(s.score, 3), "used": s.used,
         "symbol": (s.symbol or "")[:60], "snippet": (s.snippet or "")[:280]}
        for s in sources if s.path not in _SKIP_PATHS
    ][:14]
    if src_compact:
        _hist_extra["sources"] = src_compact
    _append_history(namespace, request.question, answer, settings=settings,
                    extra=_hist_extra or None)

    return AskResponse(answer=answer, sources=sources, grounded=grounded,
                       validation_status=status, wiring=wiring, trace=trace)
