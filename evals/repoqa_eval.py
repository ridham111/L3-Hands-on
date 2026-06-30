"""RepoQA-adapted retrieval benchmark for the Claude Agent SDK agent.

RepoQA (https://github.com/evalplus/repoqa) is a public "search-needle-function"
benchmark: given a natural-language description of one function, locate that
function in a repository. The standard harness stuffs the whole repo into the
prompt — that tests a model's long-context recall, NOT a RAG system.

This is the ADAPTED version that tests OUR agent honestly: we materialize each
RepoQA repo, INGEST it into our index, then ask the agent to locate each needle
function from its description and check whether it names the right function/file.
So this measures *our retrieval pipeline*, and is comparable to the RepoQA
leaderboard only loosely (different protocol) — label it as such.

Live Claude subscription required for the queries. From the repo root:

    python -m evals.repoqa_eval
    REPOQA_LANGS=typescript,python REPOQA_REPOS=1 REPOQA_NEEDLES=5 \
      AB_MODELS=haiku,sonnet,opus python -m evals.repoqa_eval

Writes evals/repoqa_results.json. The dataset is cached under evals/.cache/.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("ANTHROPIC_API_KEY", None)

import onboarding_brain.kt.ingest as _ingest
_ingest._fire_briefing_background = lambda *a, **k: None  # no LLM briefing during benchmark ingest

from onboarding_brain.config import get_settings
from onboarding_brain.contract import AskRequest, IngestRequest
from onboarding_brain.kt.chat import ask as kt_ask
from onboarding_brain.kt.ingest import ingest_repo
from onboarding_brain.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider

HERE = Path(__file__).parent
CACHE = HERE / ".cache"
OUT = HERE / "repoqa_results.json"
DATA_VERSION = os.getenv("REPOQA_DATA_VERSION", "2024-06-23")
DATA_URL = f"https://github.com/evalplus/repoqa_release/releases/download/{DATA_VERSION}/repoqa-{DATA_VERSION}.json.gz"

MODELS = [m.strip() for m in os.getenv("AB_MODELS", "haiku,sonnet,opus").split(",") if m.strip()]
LANGS = [s.strip() for s in os.getenv("REPOQA_LANGS", "typescript,python").split(",") if s.strip()]
REPOS_PER_LANG = int(os.getenv("REPOQA_REPOS", "1"))
NEEDLES = int(os.getenv("REPOQA_NEEDLES", "5"))


def load_dataset() -> dict:
    override = os.getenv("REPOQA_DATA")
    if override and Path(override).exists():
        return json.loads(Path(override).read_text(encoding="utf-8"))
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"repoqa-{DATA_VERSION}.json"
    if not cached.exists():
        print(f"downloading RepoQA dataset {DATA_VERSION} ...")
        raw = urllib.request.urlopen(
            urllib.request.Request(DATA_URL, headers={"User-Agent": "curl/8"}), timeout=120).read()
        cached.write_text(json.dumps(json.loads(gzip.decompress(raw))), encoding="utf-8")
    return json.loads(cached.read_text(encoding="utf-8"))


def materialize_and_ingest(repo: dict, namespace: str, settings) -> bool:
    """Write the repo's files to a temp dir and ingest into our index."""
    tmp = Path(tempfile.mkdtemp(prefix="repoqa_"))
    for rel, text in repo["content"].items():
        fp = tmp / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            fp.write_text(text, encoding="utf-8")
        except Exception:
            fp.write_text(text, encoding="utf-8", errors="replace")
    resp = ingest_repo(IngestRequest(repo_path=str(tmp), namespace=namespace, rebuild=True),
                       settings=settings)
    return getattr(resp, "files_indexed", 0) > 0


def score_needle(needle: dict, answer: str, source_paths: list[str]) -> dict:
    """A needle is 'located' if the answer names the right function AND points at
    the right file (named in the answer or among the cited sources)."""
    name = (needle.get("name") or "").lower()
    path = (needle.get("path") or "").lower()
    base = path.rsplit("/", 1)[-1]
    a = (answer or "").lower()
    name_hit = bool(name) and re.search(rf"\b{re.escape(name)}\b", a) is not None
    path_hit = bool(base) and (base in a or any(base in p.lower() or path in p.lower() for p in source_paths))
    return {"name": needle.get("name"), "path": needle.get("path"),
            "name_hit": name_hit, "path_hit": path_hit, "located": name_hit and path_hit}


def build_tasks(ds: dict, settings) -> list[dict]:
    """Ingest the sampled repos once; return the per-needle query tasks."""
    tasks = []
    for lang in LANGS:
        repos = ds.get(lang, [])[:REPOS_PER_LANG]
        for ri, repo in enumerate(repos):
            ns = f"repoqa-{lang}-{ri}"
            print(f"  ingesting {lang} repo '{repo.get('repo')}' -> {ns}", flush=True)
            try:
                if not materialize_and_ingest(repo, ns, settings):
                    print("    (no files indexed, skipping)"); continue
            except Exception as exc:  # noqa: BLE001
                print(f"    ingest failed: {exc}"); continue
            for needle in repo.get("needles", [])[:NEEDLES]:
                tasks.append({"lang": lang, "namespace": ns, "repo": repo.get("repo"), "needle": needle})
    return tasks


QUERY = ("This repository has been indexed. Identify the SINGLE function that matches the "
         "description below. Reply with the function's exact name and the file path where it "
         "is defined.\n\nDescription:\n{desc}")


PACE_S = float(os.getenv("REPOQA_PACE_S", "3"))      # gap between queries to ease rate limits
RL_TRIES = int(os.getenv("REPOQA_RL_TRIES", "4"))     # retries when rate-limited


def _ask_with_backoff(settings, provider, namespace, question):
    """kt_ask, but retry with growing backoff when the subscription rate-limits.
    The agent now returns a NOT_FOUND answer with 'rate_limit' in trace.errors
    rather than raising, so we detect it there."""
    r = None
    for attempt in range(RL_TRIES):
        r = kt_ask(AskRequest(namespace=namespace, question=question, history=[]),
                   settings=settings, provider=provider)
        errs = " ".join(r.trace.errors).lower() if (r.trace and r.trace.errors) else ""
        if "rate_limit" not in errs and "rate limit" not in errs:
            return r
        wait = 30 * (attempt + 1)
        print(f"      rate-limited; backing off {wait}s (attempt {attempt + 1}/{RL_TRIES})", flush=True)
        time.sleep(wait)
    return r


def run_model(model: str, tasks: list[dict]) -> dict:
    settings = dataclasses.replace(get_settings(),
                                   claude_sdk_model=("" if model in ("", "default") else model))
    try:
        provider = ClaudeAgentSDKProvider(settings)
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "error": str(exc), "rows": []}
    rows = []
    for t in tasks:
        nd = t["needle"]
        print(f"   [{model}] {t['lang']}/{nd.get('name')} ...", flush=True)
        try:
            r = _ask_with_backoff(settings, provider, t["namespace"],
                                  QUERY.format(desc=nd.get("description", "")))
            errs = " ".join(r.trace.errors).lower() if (r.trace and r.trace.errors) else ""
            if "rate_limit" in errs:
                sc = {"error": "rate_limit (exhausted retries)"}
            else:
                sc = score_needle(nd, r.answer, [s.path for s in r.sources])
        except Exception as exc:  # noqa: BLE001
            sc = {"error": str(exc)}
        sc["lang"] = t["lang"]
        rows.append(sc)
        print(f"      located={sc.get('located')} name={sc.get('name_hit')} "
              f"path={sc.get('path_hit')} {sc.get('error','')}")
        time.sleep(PACE_S)
    return {"model": model, "rows": rows}


def summarize(result: dict) -> dict:
    rows = [r for r in result.get("rows", []) if "error" not in r]
    n = len(rows)
    loc = sum(1 for r in rows if r["located"])
    nm = sum(1 for r in rows if r["name_hit"])
    return {"model": result["model"], "error": result.get("error"), "n": n,
            "located_acc": round(loc / n, 3) if n else None,
            "name_acc": round(nm / n, 3) if n else None}


def main():
    settings = get_settings()
    ds = load_dataset()
    print(f"RepoQA-adapted | langs={LANGS} repos/lang={REPOS_PER_LANG} needles={NEEDLES} "
          f"| models={MODELS}")
    tasks = build_tasks(ds, settings)
    print(f"total needle tasks: {len(tasks)}")

    results = []
    for model in MODELS:
        print(f"\n=== MODEL: {model} ===")
        results.append(run_model(model, tasks))
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summaries = [summarize(r) for r in results]
    print("\n" + "=" * 60)
    print(f"{'model':14}{'located_acc':>14}{'name_acc':>12}{'n':>6}")
    print("-" * 60)
    for s in summaries:
        if s.get("error"):
            print(f"{s['model']:14}  ERROR: {s['error'][:40]}"); continue
        print(f"{s['model']:14}{str(s['located_acc']):>14}{str(s['name_acc']):>12}{s['n']:>6}")
    print("=" * 60)
    print("located_acc = found the right function name AND file (adapted RepoQA protocol).")
    OUT.write_text(json.dumps({"summaries": summaries, "detail": results}, indent=2), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
