"""RAGAS-style quality scorecard for the Claude Agent SDK backend.

Grades the agent the way the industry grades RAG systems — with reference-free
metrics scored by an LLM judge, so the numbers are methodology-comparable to what
RAG vendors publish (no bespoke per-repo "gold answers" required):

  • faithfulness       — fraction of the answer's claims actually supported by the
                         code snippets the agent retrieved (i.e. NOT hallucinated)
  • answer_relevancy   — how directly the answer addresses the question
  • context_relevance  — signal-vs-noise of the retrieved snippets for the question
  • abstention (negative control) — for a feature that doesn't exist, did it stay
                         ungrounded with no sources? (honest "I couldn't find it")

Each is 0.0–1.0. Runs across one or more models so you can compare tiers.

Live Claude subscription required. Run from the repo root:

    python -m evals.ragas_eval                          # default models
    AB_MODELS=haiku,sonnet,opus python -m evals.ragas_eval
    AB_LIMIT=2 python -m evals.ragas_eval               # smoke test (first N questions)

Writes evals/ragas_results.json.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("ANTHROPIC_API_KEY", None)  # force subscription OAuth

from onboarding_brain.config import get_settings
from onboarding_brain.contract import AskRequest
from onboarding_brain.kt.chat import ask as kt_ask
from onboarding_brain.kt.store import get_store
from onboarding_brain.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider

HERE = Path(__file__).parent
PROBE = json.loads((HERE / "probe_questions.json").read_text(encoding="utf-8"))
OUT = HERE / "ragas_results.json"

MODELS = [m.strip() for m in os.getenv("AB_MODELS", "haiku,sonnet,opus").split(",") if m.strip()]
JUDGE_MODEL = os.getenv("AB_JUDGE_MODEL", "sonnet")
LIMIT = int(os.getenv("AB_LIMIT", "0"))
QUESTIONS = PROBE["questions"][:LIMIT] if LIMIT else PROBE["questions"]
NAMESPACE = PROBE["namespace"]

JUDGE_SYS = (
    "You are a strict RAG evaluation judge. You are given a QUESTION, the ANSWER an AI gave, "
    "and the CONTEXTS (code snippets) the AI retrieved and was supposed to ground its answer in. "
    "Score three metrics from 0.0 to 1.0:\n"
    "  faithfulness: the fraction of the answer's factual, repo-specific claims that are directly "
    "supported by the CONTEXTS. Penalize any claim not backed by a context snippet.\n"
    "  answer_relevancy: how directly and completely the ANSWER addresses the QUESTION.\n"
    "  context_relevance: how relevant the CONTEXTS are to the QUESTION (1.0 = all on-point, "
    "lower if they include unrelated files).\n"
    'Return ONLY JSON: {"faithfulness":x,"answer_relevancy":x,"context_relevance":x}.'
)


def judge_ragas(judge_provider, question: str, answer: str, contexts: list[str]) -> dict:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))[:12000] or "(none retrieved)"
    prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer[:3000]}\n\nCONTEXTS:\n{ctx}"
    try:
        raw = judge_provider.complete(JUDGE_SYS, prompt).text
        v = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return {k: float(v[k]) for k in ("faithfulness", "answer_relevancy", "context_relevance") if k in v}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def avg(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def run_model(model: str, judge_provider) -> dict:
    settings = dataclasses.replace(
        get_settings(), claude_sdk_model=("" if model in ("", "default") else model))
    try:
        provider = ClaudeAgentSDKProvider(settings)
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "error": f"provider build failed: {exc}", "rows": []}

    store = get_store(settings)
    rows = []
    for q in QUESTIONS:
        print(f"   [{model}] {q['id']} ...", flush=True)
        t = time.time()
        try:
            r = kt_ask(AskRequest(namespace=NAMESPACE, question=q["q"], history=[]),
                       settings=settings, provider=provider)
        except Exception as exc:  # noqa: BLE001
            rows.append({"id": q["id"], "error": str(exc)})
            continue
        # Contexts = what the agent actually grounded on. The cited sources only
        # carry a short snippet, but the agent read the full files via read_file —
        # so pull fuller file content for a fair faithfulness judgement.
        contexts = []
        for s in r.sources:
            full = store.full_file(NAMESPACE, s.path)
            body = (full.get("content") or full.get("text") if full else None) or (s.snippet or "")
            contexts.append(f"{s.path}\n{body[:2000]}")
        row = {"id": q["id"], "latency": round(time.time() - t, 1),
               "grounded": r.grounded, "n_sources": len(r.sources), "answer": r.answer}
        if q.get("negative"):
            row["abstention_ok"] = (not r.grounded) and len(r.sources) == 0
        else:
            row["ragas"] = judge_ragas(judge_provider, q["q"], r.answer, contexts)
        rows.append(row)
        print(f"      grounded={row['grounded']} n_src={row['n_sources']} "
              f"ragas={row.get('ragas')} abst={row.get('abstention_ok')} {row['latency']}s")
    return {"model": model, "rows": rows}


def summarize(result: dict) -> dict:
    rows = [r for r in result.get("rows", []) if "error" not in r]
    pos = [r for r in rows if "ragas" in r and "error" not in r["ragas"]]
    neg = [r for r in rows if "abstention_ok" in r]
    return {
        "model": result["model"],
        "error": result.get("error"),
        "faithfulness": avg([r["ragas"].get("faithfulness") for r in pos]),
        "answer_relevancy": avg([r["ragas"].get("answer_relevancy") for r in pos]),
        "context_relevance": avg([r["ragas"].get("context_relevance") for r in pos]),
        "abstention_ok": (sum(1 for r in neg if r["abstention_ok"]) / len(neg)) if neg else None,
        "avg_latency": avg([r["latency"] for r in rows]),
    }


def main():
    print(f"RAGAS scorecard | models={MODELS} | judge={JUDGE_MODEL} | "
          f"questions={len(QUESTIONS)} | repo={NAMESPACE}")
    judge_provider = ClaudeAgentSDKProvider(
        dataclasses.replace(get_settings(), claude_sdk_model=JUDGE_MODEL))

    results = []
    for model in MODELS:
        print(f"\n=== MODEL: {model} ===")
        results.append(run_model(model, judge_provider))
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")  # incremental

    summaries = [summarize(r) for r in results]
    print("\n" + "=" * 82)
    print(f"{'model':14}{'faithful':>11}{'ans_relev':>11}{'ctx_relev':>11}{'abstain':>10}{'latency':>11}")
    print("-" * 82)
    for s in summaries:
        if s.get("error"):
            print(f"{s['model']:14}  ERROR: {s['error'][:55]}")
            continue
        print(f"{s['model']:14}{str(s['faithfulness']):>11}{str(s['answer_relevancy']):>11}"
              f"{str(s['context_relevance']):>11}{str(s['abstention_ok']):>10}"
              f"{str(s['avg_latency'])+'s':>11}")
    print("=" * 82)
    print("Scores 0.0-1.0 (higher=better). faithfulness=no hallucination; "
          "abstain=honest 'not found' on the negative control.")
    OUT.write_text(json.dumps({"summaries": summaries, "detail": results}, indent=2), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
