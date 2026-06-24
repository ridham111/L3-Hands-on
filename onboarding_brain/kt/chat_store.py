"""Chat-history storage — two backends behind one interface.

  * JsonChatStore  : one chat_history.json per namespace under the index dir.
                     Zero setup, fully offline, the default.
  * MongoChatStore : one document per namespace in MongoDB; the turn list is
                     capped atomically with $push / $slice. Used when a Mongo
                     URI is configured.

Selection (settings.chat_store):
  "auto"  -> Mongo if ONBOARDING_MONGO_URI is set AND pymongo imports AND the
             server answers a ping; otherwise JSON.
  "mongo" -> force Mongo (still falls back to JSON if it can't connect).
  "json"  -> force JSON.

A turn is {"role": "user"|"assistant", "content": str, "ts": float}. Every
namespace keeps at most _HISTORY_CAP turns. Any Mongo error degrades to JSON so
a database outage can never lose a turn or crash a request.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from ..config import Settings, get_settings
from ..trace import logger
from .store import get_store, slugify

_HISTORY_CAP = 80  # turns kept per namespace
_QUESTION_CAP = 2000  # chars stored per user question
_ANSWER_CAP = 4000    # chars stored per assistant answer


def _turns(question: str, answer: str, extra: Optional[dict] = None) -> list[dict]:
    now = time.time()
    assistant = {"role": "assistant", "content": answer[:_ANSWER_CAP], "ts": now}
    if extra:
        # carry the connectivity diagram (and any future per-answer artifacts)
        # so a refreshed session restores exactly what was on screen
        assistant.update(extra)
    return [
        {"role": "user", "content": question[:_QUESTION_CAP], "ts": now},
        assistant,
    ]


class JsonChatStore:
    """File-backed history — the offline default. One JSON array per namespace."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _path(self, namespace: str):
        store = get_store(self.settings)
        return store.ns_dir(slugify(namespace)) / "chat_history.json"

    def load(self, namespace: str) -> list[dict]:
        p = self._path(namespace)
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []
        except (OSError, json.JSONDecodeError):
            return []

    def append(self, namespace: str, question: str, answer: str,
               extra: Optional[dict] = None) -> None:
        p = self._path(namespace)
        turns = self.load(namespace)
        turns.extend(_turns(question, answer, extra))
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(turns[-_HISTORY_CAP:], ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # persistence is best-effort

    def clear(self, namespace: str) -> None:
        try:
            self._path(namespace).unlink(missing_ok=True)
        except OSError:
            pass


class MongoChatStore:
    """MongoDB-backed history. One document per namespace: {_id, turns:[...]}.

    Turns are appended and capped to the last _HISTORY_CAP in a single atomic
    update ($push with $slice), so concurrent questions can't corrupt the log.
    """

    def __init__(self, collection):
        self._col = collection

    def load(self, namespace: str) -> list[dict]:
        doc = self._col.find_one({"_id": slugify(namespace)}, {"turns": 1})
        return list(doc.get("turns", [])) if doc else []

    def append(self, namespace: str, question: str, answer: str,
               extra: Optional[dict] = None) -> None:
        self._col.update_one(
            {"_id": slugify(namespace)},
            {"$push": {"turns": {"$each": _turns(question, answer, extra), "$slice": -_HISTORY_CAP}}},
            upsert=True,
        )

    def clear(self, namespace: str) -> None:
        self._col.delete_one({"_id": slugify(namespace)})


def _try_mongo(settings: Settings):
    """Return a connected MongoChatStore, or None if Mongo is unavailable."""
    if not settings.mongo_uri:
        return None
    try:
        from pymongo import MongoClient  # noqa: PLC0415
    except ImportError:
        logger.info("chat_store_mongo_unavailable",
                    extra={"event": "chat_store_mongo_unavailable", "reason": "pymongo_not_installed"})
        return None
    try:
        client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")  # fail fast if the server is unreachable
        col = client[settings.mongo_db]["chat_turns"]
        logger.info("chat_store_selected",
                    extra={"event": "chat_store_selected", "backend": "mongo", "db": settings.mongo_db})
        return MongoChatStore(col)
    except Exception as exc:  # connection/auth/timeout -> degrade to JSON
        logger.info("chat_store_mongo_unavailable",
                    extra={"event": "chat_store_mongo_unavailable", "reason": str(exc)[:200]})
        return None


# one resolved store per (chat_store, uri, db) combo — avoid reconnecting per call
_CACHE: dict[tuple, object] = {}


def get_chat_store(settings: Optional[Settings] = None):
    settings = settings or get_settings()
    key = (settings.chat_store, settings.mongo_uri, settings.mongo_db, settings.index_dir)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    store = None
    if settings.chat_store in ("auto", "mongo"):
        store = _try_mongo(settings)
    if store is None:
        store = JsonChatStore(settings)
    _CACHE[key] = store
    return store
