"""Pluggable, persistent, namespaced vector index.

Default backend: TF-IDF (scikit-learn) — fully offline, no model download, no
key, strong for keyword-dense code. Each repo is a namespace persisted under
ONBOARDING_INDEX_DIR ("our DB"), so re-opening doesn't re-ingest.

The VectorStore interface is backend-agnostic; a Chroma + dense-embedding
backend can be added behind the same interface (set ONBOARDING_VECTOR_BACKEND).
"""
from __future__ import annotations

import abc
import json
import re
from pathlib import Path
from typing import Any, Optional

from ..config import Settings, get_settings


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower()).strip("-")
    return s[:80] or "repo"


# Module-level (picklable) so the fitted vectorizer can be persisted with joblib.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEP = re.compile(r"[_./\\\-]+")


def split_identifiers(text: str) -> str:
    """Code-aware preprocessing: split camelCase / snake_case / dotted paths into
    subwords AND keep the original words, so a query for "rate limiting" matches
    `rate_limit` and "authentication" matches `requireAuth`. Big recall win for
    code over plain TF-IDF tokenization."""
    camel = _CAMEL.sub(" ", text)
    expanded = _SEP.sub(" ", camel)
    return (camel + " " + expanded).lower()


# Test/spec/mock files mention every feature by name, so they outrank the real
# implementation. Unless the user asks about tests, halve their scores.
_NOISE_PATH = re.compile(
    r"(^|/)(tests?|__tests__|test_utils|e2e|__mocks__|fixtures)(/|$)"
    r"|\.(spec|test|stories)\.[a-z]+$|_test\.[a-z]+$|(^|/)(conftest|test_)[^/]*\.py$",
    re.IGNORECASE,
)
_WANTS_TESTS = re.compile(r"\b(test|tests|spec|specs|e2e|coverage|mock)\b", re.IGNORECASE)
# git-history chunks are only relevant when the question is about change history;
# otherwise they compete with real code and dilute results.
_WANTS_HISTORY = re.compile(
    r"\b(commit|commits|committed|history|changelog|recent|recently|"
    r"changed|changes|who (wrote|made|added|worked)|last (week|month|release)|"
    r"latest|version history|author|blame)\b", re.IGNORECASE)

# Question scaffolding that must not trigger filename boosts.
_QUERY_STOP = {
    "the", "and", "are", "was", "were", "how", "what", "where", "when", "why", "who",
    "which", "does", "doing", "this", "that", "with", "from", "for", "into", "can",
    "you", "your", "all", "any", "get", "use", "used", "using", "work", "works",
    "handled", "handle", "defined", "located", "file", "files", "code", "project",
}


def _query_terms(query: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", split_identifiers(query)) if t not in _QUERY_STOP}


def rerank(results: list[dict], query: str, k: int) -> list[dict]:
    """Re-rank over-fetched results, then cut to k (stable for ties):
    - boost chunks whose FILE NAME matches a meaningful query term — asking
      about "authentication" should surface authentication.service.ts even
      when many files mention the word;
    - penalize test/spec files unless the question is about tests."""
    qterms = _query_terms(query)
    wants_tests = bool(_WANTS_TESTS.search(query))
    wants_history = bool(_WANTS_HISTORY.search(query))
    for r in results:
        path = r.get("metadata", {}).get("path", "")
        if qterms:
            basename = path.rsplit("/", 1)[-1]
            btokens = set(re.findall(r"[a-z0-9]{3,}", split_identifiers(basename)))
            if btokens & qterms:
                r["score"] = round(r["score"] * 1.4, 4)
        if not wants_tests and _NOISE_PATH.search(path):
            r["score"] = round(r["score"] * 0.5, 4)
        # commit chunks only earn their place on history questions; otherwise
        # heavily penalize so they don't crowd out real code
        if path == "git-history":
            r["score"] = round(r["score"] * (1.3 if wants_history else 0.25), 4)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


class VectorStore(abc.ABC):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.index_dir)
        self._paths_cache: dict[str, tuple[int, list[dict]]] = {}

    def ns_dir(self, namespace: str) -> Path:
        # defense-in-depth: a namespace is a single directory name, never a path
        if not namespace or namespace in (".", "..") or any(ch in namespace for ch in "/\\"):
            raise ValueError(f"invalid namespace: {namespace!r}")
        return self.root / namespace

    def exists(self, namespace: str) -> bool:
        return (self.ns_dir(namespace) / "meta.json").exists()

    def list_namespaces(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for d in sorted(self.root.iterdir()):
            meta = d / "meta.json"
            if meta.is_file() and self.exists(d.name):  # backend-aware
                try:
                    out.append(json.loads(meta.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        return out

    def _load_chunk_docs(self, namespace: str) -> list[dict]:
        """The namespace's persisted chunks, cached per chunks.json mtime."""
        f = self.ns_dir(namespace) / "chunks.json"
        try:
            mtime = f.stat().st_mtime_ns
        except OSError:
            return []
        cached = self._paths_cache.get(namespace)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            chunks = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        self._paths_cache[namespace] = (mtime, chunks)
        return chunks

    def known_paths(self, namespace: str) -> set[str]:
        """Every file path in the namespace's index — the universe of real
        files for citation checks and filename lookups."""
        return {c.get("metadata", {}).get("path", "") for c in self._load_chunk_docs(namespace)} - {""}

    def full_file(self, namespace: str, path: str) -> Optional[dict]:
        """Reconstruct a file's full indexed text by stitching its chunks back
        together in order — so the UI can show the WHOLE file, not one snippet.
        (Indexed content is capped at ingest by max_ingest_file_bytes; very large
        files are truncated, flagged via `truncated`.)"""
        chunks = [c for c in self._load_chunk_docs(namespace)
                  if c.get("metadata", {}).get("path") == path]
        if not chunks:
            return None
        chunks.sort(key=lambda c: int(c.get("metadata", {}).get("chunk_index", 0)))
        content = "".join(c.get("text", "") for c in chunks)
        m0 = chunks[0].get("metadata", {})
        return {
            "path": path, "language": m0.get("language", ""),
            "content": content, "lines": content.count("\n") + 1,
            "truncated": len(content) >= self.settings.max_ingest_file_bytes - 200,
        }

    def first_chunk(self, namespace: str, path: str) -> Optional[dict]:
        """Exact-path lookup: the file's first chunk, for pulling in files the
        question names by basename but vector search missed."""
        for c in self._load_chunk_docs(namespace):
            if c.get("metadata", {}).get("path") == path:
                return {"id": c["id"], "score": 0.0, "text": c["text"], "metadata": c["metadata"]}
        return None

    def get_chunk(self, namespace: str, chunk_id: str) -> Optional[dict]:
        """Exact-id lookup ("path#idx") — used to pull a hit's neighboring
        chunks so the LLM sees the surrounding code, not just the fragment."""
        for c in self._load_chunk_docs(namespace):
            if c.get("id") == chunk_id:
                return {"id": c["id"], "score": 0.0, "text": c["text"], "metadata": c["metadata"]}
        return None

    @abc.abstractmethod
    def index(self, namespace: str, chunks: list[dict], meta: dict) -> None: ...

    @abc.abstractmethod
    def search(self, namespace: str, query: str, k: int) -> list[dict]: ...


class TfidfStore(VectorStore):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._cache: dict[str, Any] = {}

    def exists(self, namespace: str) -> bool:
        # backend-aware: a namespace indexed with a different backend is "not indexed" here
        return (self.ns_dir(namespace) / "tfidf.joblib").exists()

    def index(self, namespace: str, chunks: list[dict], meta: dict) -> None:
        from joblib import dump
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not chunks:
            raise ValueError("nothing to index (no chunks)")
        # index_text includes the file path so file/folder names aid retrieval
        texts = [c.get("index_text") or c["text"] for c in chunks]
        vec = TfidfVectorizer(
            preprocessor=split_identifiers, token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2), min_df=1, max_features=80000, sublinear_tf=True,
        )
        matrix = vec.fit_transform(texts)

        d = self.ns_dir(namespace)
        d.mkdir(parents=True, exist_ok=True)
        dump({"vectorizer": vec, "matrix": matrix}, d / "tfidf.joblib")
        (d / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        meta = {**meta, "namespace": namespace, "n_chunks": len(chunks), "backend": "tfidf"}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self._cache.pop(namespace, None)

    def _load(self, namespace: str):
        if namespace in self._cache:
            return self._cache[namespace]
        from joblib import load

        d = self.ns_dir(namespace)
        if not (d / "tfidf.joblib").exists():
            raise FileNotFoundError(f"namespace not indexed: {namespace}")
        blob = load(d / "tfidf.joblib")
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        entry = {"vec": blob["vectorizer"], "matrix": blob["matrix"], "chunks": chunks}
        self._cache[namespace] = entry
        return entry

    def search(self, namespace: str, query: str, k: int) -> list[dict]:
        from sklearn.metrics.pairwise import linear_kernel

        e = self._load(namespace)
        qv = e["vec"].transform([query])
        sims = linear_kernel(qv, e["matrix"]).ravel()  # tfidf is l2-normalized -> cosine
        if not len(sims):
            return []
        order = sims.argsort()[::-1][: max(k * 3, 12)]  # over-fetch for re-ranking
        results = []
        for i in order:
            score = float(sims[i])
            # 0.005 cosine floor — only drop truly zero-overlap chunks
            if score < 0.005:
                continue
            c = e["chunks"][int(i)]
            results.append({
                "id": c["id"], "score": round(score, 4),
                "text": c["text"], "metadata": c["metadata"],
            })
        return rerank(results, query, k)


# One store per (backend, index dir) so the loaded index survives across
# requests — without this, every /v1/ask reloads the matrix from disk.
_STORES: dict[tuple[str, str], VectorStore] = {}


def get_store(settings: Optional[Settings] = None) -> VectorStore:
    settings = settings or get_settings()
    key = (settings.vector_backend, str(Path(settings.index_dir).resolve()))
    store = _STORES.get(key)
    if store is None:
        if settings.vector_backend == "dense":
            from .dense_store import DenseStore

            store = DenseStore(settings)
        elif settings.vector_backend == "hybrid":
            from .hybrid_store import HybridStore

            store = HybridStore(settings)
        else:
            store = TfidfStore(settings)
        _STORES[key] = store
    return store
