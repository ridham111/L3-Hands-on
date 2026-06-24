# Evaluation Rubric — Cortex agents

Two layers. The **deterministic gate** (below) is authoritative for regression;
an **LLM-as-judge** can be layered later for narrative-quality scoring.

## Coverage (31 cases across 5 agents)

| Agent | Cases | Entry point exercised |
|---|---|---|
| briefing (onboarding-brain) | 10 | `generate_briefing` |
| chat (kt-brain / RAG) | 10 | `kt_ask` (ingest → ask) |
| guided tour | 5 | `build_tour` (ingest → tour) |
| project walkthrough | 3 | `build_walkthrough` (ingest → walkthrough) |
| installation-guide | 3 | `run_agent("installation-guide", …)` |

The two primary agents meet the rubric's ≥10-per-agent bar; tour and
installation-guide are narrower derived agents with focused suites. Per-agent
pass counts are recorded under `per_agent` in `results.json`.

## Deterministic checks (authoritative gate)

Each case in `runner.py` builds a throwaway fixture repo and asserts:

| Check | Agent | Meaning |
|---|---|---|
| `overview_contains` / `_equals` / `_excludes` | briefing | Overview reflects the README (and **excludes** injected text) |
| `overview_source` / `setup_source` | briefing | The answer cites the real file it came from |
| `setup_contains` | briefing / install | Run steps derived from the actual config (npm/pip/docker/ng/go/cargo) |
| `folder` | briefing | Each top-level folder is surfaced in the folder map |
| `recent_equals: "not found in repo"` | briefing | No git history → the agent says so, never invents commits |
| `no_unresolved_sources` | briefing | Every cited source resolves to a real repo artifact (trust check) |
| `status_not: failed` | briefing / install | The run produced a usable result |
| `source_contains` | chat | Retrieval surfaced the right file for the question |
| `grounded` / `no_hallucinated` | chat | Answer is grounded; no cited file is invented |
| `entry_endswith` | tour | The bootstrap/entry file is detected (FastAPI/`__main__`/Node/Go/Flask) |
| `min_stops` / `has_main` / `first_chapter` | tour | Flow chapters populate; the entry is flagged and read first |
| `stack_contains` / `has_section` / `min_sections` / `grounded_files` | walkthrough | Framework detected; the right sections are produced and grounded in real files |
| `prereq_contains` | install | Toolchain prerequisite inferred from the detected stack (Node/Python/Go) |

A case passes only if **all** its checks pass; the gate passes when the overall
pass rate ≥ `--threshold` (default `1.0`).

## What the gate protects

1. **Grounding** — answers cite real files; invented citations are flagged.
2. **No hallucination** — missing data yields "not found in repo".
3. **Prompt-injection resistance** — instructions embedded in a README are not
   surfaced or obeyed.
4. **Config-driven setup** — run steps come from the actual build files.
5. **Retrieval relevance** — a question surfaces the right file (chat).
6. **Entry-point detection** — the tour starts at the real bootstrap file across
   stacks (Python/Node/Go/Flask/FastAPI).
7. **Prerequisite inference** — the installation guide names the right toolchain.

## Interpreting `results.json`

- `gate_passed: true`, `pass_rate: 1.0` → safe to ship.
- A failed case prints the exact failing check.
- `trace.grounding.unresolved_sources` non-empty on a real-LLM run = the model
  cited a file that doesn't exist → investigate before shipping.
