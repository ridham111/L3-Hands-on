# Evaluation Guide — Cortex

## Overview

Cortex uses a two-layer evaluation strategy:

1. **Deterministic gate** (`evals/runner.py`) — fast, hermetic, CI-blocking. No LLM needed.
2. **LLM-as-judge** (rubric in `evals/rubric.md`) — for narrative quality scoring on real repos.

The deterministic gate is the regression guard. It must pass before any change is merged.

---

## Running the evals

```powershell
# Full quality gate (31 cases, all agents)
.\.venv\Scripts\python.exe -m evals.runner

# With custom pass threshold (default: 1.0 = 100%)
.\.venv\Scripts\python.exe -m evals.runner --threshold 0.9

# Run only one agent
.\.venv\Scripts\python.exe -m evals.runner --agent chat

# Write results to a custom path
.\.venv\Scripts\python.exe -m evals.runner --output evals/results_ci.json
```

Exit code `0` = gate passed. Exit code `1` = one or more cases failed (blocks CI).

---

## Test coverage

31 deterministic cases across 5 agents:

| Agent | Cases | What is tested |
|---|---|---|
| `briefing` | 10 | Overview grounding, setup steps, folder map, injection resistance, empty-repo handling |
| `chat` | 10 | Retrieval relevance, grounding, no hallucination, follow-up with history |
| `tour` | 5 | Entry-point detection across Python / Node / Go / Flask / FastAPI |
| `walkthrough` | 3 | Stack detection, section structure, file grounding |
| `installation` | 3 | Prerequisite inference for Angular / FastAPI / Go stacks |

Each case builds a **throwaway fixture repo in memory** — no real git clone, no embedding model download, no LLM call required. Tests are fully hermetic and run in seconds.

### How it stays deterministic — the stub provider

A real LLM phrases answers differently every run, so exact-match checks can only pass against a deterministic generator. The gate uses a **stub provider** (`evals/stub_provider.py`) — a test double that produces grounded, repeatable output purely from the repo context (it only cites files that exist and says "not found in repo" when data is absent).

Crucially, the stub is **not a user-facing backend**. User backends stay exactly `claude | groq | openrouter`. The stub enters only via `install_stub()`, which monkeypatches `get_provider` in the agent-flow modules at runtime:

- `evals/runner.py` calls `install_stub(settings)` — the whole gate runs on it.
- `tests/conftest.py` calls `install_stub(settings, patch_source=False)` — the pytest suite runs on it too, but `patch_source=False` leaves the real `providers.get_provider` intact so the provider-selection tests (FallbackProvider / OpenRouterProvider) still verify the real factory.

The stub dispatches by prompt shape: chat → grounded answer citing the context files; walkthrough → section body + takeaways; otherwise → briefing JSON. `results.json` records `"model_used": "stub/deterministic-v1"` so it's obvious a run was hermetic.

---

## What each check verifies

| Check | Agent | Meaning |
|---|---|---|
| `overview_contains` | briefing | The project description mentions the right name/purpose |
| `overview_equals` | briefing | Exact match — used for "not found in repo" assertions |
| `overview_excludes` | briefing | Injected text is NOT present in output (injection resistance) |
| `overview_source` | briefing | The overview cites the README as its source |
| `setup_contains` | briefing / install | Run steps match the actual build config (npm / pip / docker / go / cargo) |
| `setup_source` | briefing / install | Setup steps cite the real config file |
| `folder` | briefing | Top-level folder appears in the folder map |
| `recent_equals` | briefing | No git history → `"not found in repo"`, never invented |
| `no_unresolved_sources` | briefing | Every cited source resolves to a real repo artifact |
| `status_not: failed` | briefing / install | The run produced a usable result |
| `source_contains` | chat | The right file was surfaced for the question |
| `grounded` | chat | Answer has at least one real source attached |
| `no_hallucinated` | chat | No cited file is invented / absent from the index |
| `entry_endswith` | tour | The bootstrap file is detected correctly for each stack |
| `min_stops` | tour | Tour has enough stops to be useful |
| `has_main` | tour | The entry-point stop is flagged as `is_entry: true` |
| `first_chapter` | tour | Chapter titles match expected framework-aware flow |
| `stack_contains` | walkthrough | Detected tech stack includes the expected frameworks |
| `min_sections` | walkthrough | Walkthrough has the full 9-section structure |
| `has_section` | walkthrough | Named sections are present |
| `grounded_files` | walkthrough | Sections reference real indexed files |
| `prereq_contains` | install | The right toolchain prerequisite is stated |

A case **passes** only when ALL its checks pass.  
The gate **passes** when `passed / total >= threshold` (default 1.0).

---

## Reading results.json

```json
{
  "gate_passed": true,
  "pass_rate": 1.0,
  "total_cases": 31,
  "passed": 31,
  "failed": 0,
  "per_agent": {
    "briefing":      { "total": 10, "passed": 10 },
    "chat":          { "total": 10, "passed": 10 },
    "tour":          { "total":  5, "passed":  5 },
    "walkthrough":   { "total":  3, "passed":  3 },
    "installation":  { "total":  3, "passed":  3 }
  },
  "cases": [ ... ]
}
```

**`gate_passed: true`** — safe to ship.

**When a case fails**, find it in `cases[]` and read the failing check:
```json
{
  "agent": "chat",
  "id": "rag_finds_auth_file",
  "passed": false,
  "checks": [
    { "check": "source_contains", "passed": false, "detail": "expected src/auth.py, got []" }
  ]
}
```
The `detail` field explains what the agent returned vs. what was expected.

---

## Adding a new test case

1. Open `evals/runner.py`.
2. Add a new dict to the relevant agent's case list (e.g. `BRIEFING_CASES`):
```python
{
    "id": "my_new_case",
    "files": {
        "README.md": "# MyTool\nA tool for X.\n",
        "pyproject.toml": "[tool.poetry]\nname = 'mytool'\n",
    },
    "question": "what does this project do?",   # for chat cases
    "expect": {
        "overview_contains": "MyTool",
        "no_unresolved_sources": True,
    },
},
```
3. Run `python -m evals.runner` to confirm it passes (or debug the failure).
4. Commit both the new case and the updated `results.json`.

---

## LLM-as-judge (narrative quality)

The deterministic gate verifies structure and grounding. To evaluate prose quality on real repos:

1. Run `python -m evals.runner --agent chat --real-repo <namespace>` (if implemented) or collect answers manually.
2. Score each answer against the rubric in `evals/rubric.md`:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Grounding** | Invented files cited | Some sources unresolved | All sources resolve to real files |
| **Accuracy** | Wrong answer | Partially correct | Correct and complete |
| **Conciseness** | Verbose / padded | Acceptable | Tight, senior-engineer Slack style |
| **Source citation** | No citations | Vague ("in some file") | Exact path + line numbers |
| **Honest uncertainty** | Guesses when unsure | Hedges but doesn't say "not found" | Explicitly says "I couldn't find this" |

Score ≥ 8 / 10 = acceptable. Score < 6 = investigate prompt or retrieval.

---

## CI integration

The quality gate runs automatically on every push via `.github/workflows/ci.yml`:

```yaml
- name: Quality gate
  run: python -m evals.runner --threshold 1.0
```

The step exits non-zero on any regression, blocking the merge. `results.json` is uploaded as a build artifact so failures can be inspected without re-running locally.

---

## Known failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `source_contains` fails on chat case | Retrieval missed the file; wrong embedding | Check chunk size, try `hybrid` backend |
| `overview_equals: "not found in repo"` fails | Model invented content from empty repo | Strengthen system prompt grounding rules |
| `no_unresolved_sources` fails on real LLM run | Model cited a file that doesn't exist | Grounding validator should catch this; check `grounding.py` |
| `injection_in_readme_neutralized` fails | System prompt DATA boundary not working | Verify `<<<REPO_CONTEXT_JSON>>>` markers in `prompts.py` |
| Gate passes but answers are bad | Gate checks structure, not prose | Run LLM-as-judge rubric on real-repo samples |
