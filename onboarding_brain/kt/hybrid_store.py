"""Hybrid retrieval: lexical TF-IDF + dense semantic, fused with RRF.

TF-IDF nails exact identifiers ("rate_limit", "TfidfVectorizer"); dense
embeddings bridge vocabulary gaps ("sign in" -> `login`). Reciprocal Rank
Fusion combines both rankings without score calibration: each result
contributes 1/(K + rank) from every list it appears in, so chunks that both
retrievers agree on float to the top. Set ONBOARDING_VECTOR_BACKEND=hybrid.

Costs the dense ingest time (ONNX embedding on CPU), so it suits small/medium
repos; tfidf remains the fast default.
"""
from __future__ import annotations

import json

from ..config import Settings
from ..trace import logger
from .dense_store import DenseStore
from .store import TfidfStore, VectorStore

_RRF_K = 60  # standard damping: rank 0 ≈ 0.016, rank 9 ≈ 0.014


def rrf_merge(result_lists: list[list[dict]], k: int) -> list[dict]:
    """Fuse ranked result lists by Reciprocal Rank Fusion, keep top-k."""
    fused: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (_RRF_K + rank + 1)
            by_id.setdefault(r["id"], r)
    order = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]
    out = []
    for cid in order:
        r = dict(by_id[cid])
        r["score"] = round(fused[cid], 4)
        out.append(r)
    return out


class HybridStore(VectorStore):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._tfidf = TfidfStore(settings)
        self._dense = DenseStore(settings)

    def exists(self, namespace: str) -> bool:
        # TF-IDF is always present; dense is OPTIONAL (a large repo may be
        # indexed TF-IDF-only). So a namespace counts as indexed if TF-IDF is.
        return self._tfidf.exists(namespace)

    def index(self, namespace: str, chunks: list[dict], meta: dict) -> None:
        # large-repo safety valve: above the cap, skip the slow dense embedding
        # and index TF-IDF only so huge repos stay fast.
        dense_ok = len(chunks) <= self.settings.hybrid_max_chunks
        if dense_ok:
            # dense FIRST: its incremental reuse may need the previous chunks.json,
            # which tfidf.index overwrites
            self._dense.index(namespace, chunks, meta)
        else:
            logger.info("hybrid_dense_skipped namespace=%s chunks=%d > cap=%d — TF-IDF only",
                        namespace, len(chunks), self.settings.hybrid_max_chunks,
                        extra={"event": "hybrid_dense_skipped"})
            # remove any stale dense artifacts so search won't use mismatched vectors
            d = self.ns_dir(namespace)
            for f in ("dense.npy", "dense_keys.json"):
                try:
                    (d / f).unlink(missing_ok=True)
                except OSError:
                    pass
            self._dense._cache.pop(namespace, None)
        self._tfidf.index(namespace, chunks, meta)
        # last sub-store stamped backend="tfidf"; correct the record
        meta_path = self.ns_dir(namespace) / "meta.json"
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            m["backend"] = "hybrid"
            m["dense"] = dense_ok
            meta_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            pass

    def search(self, namespace: str, query: str, k: int) -> list[dict]:
        fetch = max(k * 2, 10)  # over-fetch so fusion has signal beyond top-k
        lists = [self._tfidf.search(namespace, query, fetch)]
        # fuse dense only when this namespace actually has dense vectors
        if self._dense.exists(namespace):
            lists.append(self._dense.search(namespace, query, fetch))
        if len(lists) == 1:
            return lists[0][:k]
        return rrf_merge(lists, k)
