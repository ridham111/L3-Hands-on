"""Dense semantic vector store (fastembed + numpy cosine).

Uses a small local ONNX embedding model (BAAI/bge-small-en-v1.5, 384-dim) via
fastembed — no torch, no API key, downloaded once and cached. Gives true
semantic retrieval (e.g. a question about "authentication" matches code using
`auth`/`login`), which lexical TF-IDF cannot. Persisted per-namespace like the
TF-IDF store, behind the same VectorStore interface.

Re-indexing is INCREMENTAL: vectors of unchanged chunks (same id + same text
hash, tracked in dense_keys.json) are reused, so only changed files pay the
CPU embedding cost. Progress is logged every few batches.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import numpy as np

from ..config import Settings
from ..trace import logger
from .store import VectorStore, rerank

_EMBEDDER: Any = None
_EMBEDDER_NAME: str = ""


def _embedder(model_name: str):
    global _EMBEDDER, _EMBEDDER_NAME
    if _EMBEDDER is None or _EMBEDDER_NAME != model_name:
        import os

        n = max(1, os.cpu_count() or 4)
        # hint OpenMP / ONNX Runtime to use all cores (read at session init).
        # Previously the threads= kwarg was wrapped in a bare try/except that
        # SILENTLY fell back to single-threaded ONNX — the #1 cause of slow
        # embedding on a multi-core box. Now we set env threads too and LOG the
        # effective count so a single-threaded fallback is visible, not silent.
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("ORT_NUM_THREADS", str(n))

        from fastembed import TextEmbedding

        try:
            _EMBEDDER = TextEmbedding(model_name=model_name, threads=n)
            logger.info("dense_embedder_init model=%s threads=%d", model_name, n,
                        extra={"event": "dense_embedder_init"})
        except TypeError:
            _EMBEDDER = TextEmbedding(model_name=model_name)
            logger.warning("dense_embedder_init model=%s threads=UNSUPPORTED (single-threaded)",
                           model_name, extra={"event": "dense_embedder_init"})
        _EMBEDDER_NAME = model_name
    return _EMBEDDER


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-9, None)


def _chunk_key(c: dict) -> str:
    """Identity of a chunk's embedded content: id + hash of the indexed text."""
    text = c.get("index_text") or c["text"]
    return c["id"] + ":" + hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


class DenseStore(VectorStore):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._cache: dict[str, Any] = {}

    def exists(self, namespace: str) -> bool:
        # backend-aware: only namespaces indexed with dense embeddings count here
        return (self.ns_dir(namespace) / "dense.npy").exists()

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Batched embedding with progress logs — a big repo takes many
        minutes on CPU and silence reads as a hang."""
        emb = _embedder(self.settings.embed_model)
        vecs: list = []
        batch = 256  # larger slice amortizes per-call overhead on CPU
        for i in range(0, len(texts), batch):
            vecs.extend(emb.embed(texts[i:i + batch]))
            done = min(i + batch, len(texts))
            if done == len(texts) or (i // batch) % 4 == 3:
                logger.info("dense_embed %d/%d chunks", done, len(texts),
                            extra={"event": "dense_embed_progress"})
        return _normalize(np.array(vecs, dtype=np.float32))

    def _reusable_vectors(self, d) -> dict[str, np.ndarray]:
        """Vectors from the previous index keyed by chunk identity. Sidecar
        dense_keys.json is authoritative; pre-sidecar layouts fall back to
        chunks.json, which was written in the same index() call as dense.npy."""
        try:
            if not (d / "dense.npy").exists():
                return {}
            matrix = np.load(d / "dense.npy")
            side = d / "dense_keys.json"
            if side.is_file():
                data = json.loads(side.read_text(encoding="utf-8"))
                if data.get("embed_model") != self.settings.embed_model:
                    return {}
                keys = data.get("keys") or []
            else:
                old = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
                keys = [_chunk_key(c) for c in old]
            if len(keys) != matrix.shape[0]:
                return {}
            return dict(zip(keys, matrix))
        except Exception:
            return {}

    def _embed_query(self, query: str) -> np.ndarray:
        emb = _embedder(self.settings.embed_model)
        # bge-style models retrieve better with the query-specific encoding
        embed_fn = getattr(emb, "query_embed", None) or emb.embed
        vec = np.array(list(embed_fn([query])), dtype=np.float32)
        return _normalize(vec)

    def index(self, namespace: str, chunks: list[dict], meta: dict) -> None:
        if not chunks:
            raise ValueError("nothing to index (no chunks)")
        d = self.ns_dir(namespace)
        d.mkdir(parents=True, exist_ok=True)

        # incremental: reuse vectors of unchanged chunks, embed only the rest
        keys = [_chunk_key(c) for c in chunks]
        reuse = self._reusable_vectors(d)
        missing = [i for i, key in enumerate(keys) if key not in reuse]
        logger.info("dense_index reuse=%d embed=%d", len(chunks) - len(missing), len(missing),
                    extra={"event": "dense_index_plan"})
        if missing:
            # index_text includes the file path so file/folder names aid retrieval
            new_vecs = self._embed([chunks[i].get("index_text") or chunks[i]["text"] for i in missing])
            dim = int(new_vecs.shape[1])
        else:
            new_vecs = None
            dim = int(next(iter(reuse.values())).shape[0])

        matrix = np.zeros((len(chunks), dim), dtype=np.float32)
        ni = 0
        for i, key in enumerate(keys):
            if key in reuse:
                matrix[i] = reuse[key]
            else:
                matrix[i] = new_vecs[ni]
                ni += 1

        np.save(d / "dense.npy", matrix)
        (d / "dense_keys.json").write_text(
            json.dumps({"embed_model": self.settings.embed_model, "keys": keys}), encoding="utf-8")
        (d / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        meta = {**meta, "namespace": namespace, "n_chunks": len(chunks),
                "backend": "dense", "embed_model": self.settings.embed_model, "dim": dim}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self._cache.pop(namespace, None)

    def _load(self, namespace: str):
        if namespace in self._cache:
            return self._cache[namespace]
        d = self.ns_dir(namespace)
        if not (d / "dense.npy").exists():
            raise FileNotFoundError(f"namespace not indexed (dense): {namespace}")
        matrix = np.load(d / "dense.npy")
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        entry = {"matrix": matrix, "chunks": chunks}
        self._cache[namespace] = entry
        return entry

    def search(self, namespace: str, query: str, k: int) -> list[dict]:
        e = self._load(namespace)
        q = self._embed_query(query)[0]
        sims = e["matrix"] @ q
        order = np.argsort(sims)[::-1][: max(k * 3, 12)]  # over-fetch for re-ranking
        results = []
        for i in order:
            score = float(sims[int(i)])
            if score <= 0.05:  # weak match floor
                continue
            c = e["chunks"][int(i)]
            results.append({"id": c["id"], "score": round(score, 4),
                            "text": c["text"], "metadata": c["metadata"]})
        return rerank(results, query, k)
