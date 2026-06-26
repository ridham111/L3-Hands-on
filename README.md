# Cortex — get up to speed on any codebase, fast

> **docs/** — [Architecture](docs/architecture.md) · [API Reference](docs/api.md) · [Eval Guide](docs/eval-guide.md) · [Limitations](docs/limitations.md)

Joining a new project is hard. You're handed a repo with hundreds of files and no idea
where to start. Cortex fixes that.

Point it at a project — a folder on your machine, or a git/Bitbucket link — and it reads
the whole thing so you don't have to. Then it can:

- **Give you a Day-1 briefing** — in plain English: what the project does, how it's laid
  out, how to run it, what changed recently, and who works on what.
- **Answer your questions** about the code, and show you the exact files and lines the
  answer came from. If it can't find something, it says so instead of making it up.
- **Walk you through a guided tour** — starting at the file where the app actually begins,
  then following how each piece connects to the next.
- **Write you a full project walkthrough** — a long, plain-English deep dive of the whole
  codebase (stack → how it starts → routing → features → business logic → data → shared
  building blocks → how it all connects → how to run), tailored to the framework it
  detects. Read it in the app or **save it as a PDF / Markdown** to share.
- **Tell you what to install first** — the runtimes and tools you need before the project
  will even run.

The golden rule: **everything it tells you comes straight from the code.** It never
invents file names, features, or answers.

---

## Get it running

You need Python. Then:

```powershell
cd onboarding-brain
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Start the app (web page + API)
.\.venv\Scripts\python.exe -m uvicorn api.server:app --port 8000
```

Open **http://localhost:8000**, paste a repo folder or a clone link, click **Ingest**,
and start asking questions.

Out of the box it runs **completely offline** — no API key needed. (For richer, natural
answers you can plug in a free Groq key later; see *Settings* below.)

---

## Using it from the terminal

If you prefer the command line:

```powershell
.\.venv\Scripts\python.exe -m cli.main ingest --repo C:\path\to\repo -n myrepo   # read a repo
.\.venv\Scripts\python.exe -m cli.main ask "How is login handled?" -n myrepo     # ask a question
.\.venv\Scripts\python.exe -m cli.main onboard --repo C:\path\to\repo            # Day-1 briefing
.\.venv\Scripts\python.exe -m cli.main namespaces                                # list repos you've read
.\.venv\Scripts\python.exe -m cli.main info                                      # show current settings
```

Every command prints its result as JSON, so it's easy to use in scripts.

---

## Using it from the API

The app also exposes a simple web API so other tools can call it. Every call needs a key
in the header (the default local key is `dev-local-key`). Here's the whole flow in two
calls:

```bash
# 1) Read a repo (a local path, or a clone_url for git/Bitbucket)
curl -X POST http://localhost:8000/v1/ingest \
  -H "Authorization: Bearer dev-local-key" -H "Content-Type: application/json" \
  -d '{"repo_path":"C:/path/to/repo","namespace":"myrepo","rebuild":true}'

# you get back something like:
# {"namespace":"myrepo","files_indexed":42,"chunks_indexed":318,"briefing_pending":true, ...}

# 2) Ask a question
curl -X POST http://localhost:8000/v1/ask \
  -H "Authorization: Bearer dev-local-key" -H "Content-Type: application/json" \
  -d '{"namespace":"myrepo","question":"how does auth work?"}'

# you get back the answer plus the files it used:
# {"answer":"The login() function ...","grounded":true,
#  "sources":[{"path":"src/auth.py","line_start":1,"line_end":24,"used":true}], ...}
```

The main endpoints:

| Endpoint | What it does |
|---|---|
| `POST /v1/ingest` | Read a repo (local folder **or** clone a git/Bitbucket URL) |
| `POST /v1/resync/{repo}` | Re-read a repo to pick up new commits and code |
| `POST /v1/ask` | Ask a question, get a sourced answer |
| `GET /v1/tour/{repo}` | Get the guided tour |
| `POST/GET /v1/walkthrough/{repo}` | Start / fetch the full project walkthrough |
| `GET /v1/briefing/{repo}` | Get the Day-1 briefing |
| `GET /v1/gaps/{repo}` | See which files are confusing and worth documenting |
| `POST /v1/agents/{id}/run` | Run a specific helper (briefing or install-guide) |
| `GET /v1/namespaces`, `GET /v1/agents`, `GET /health` | List repos, list helpers, health check |

Want to explore them interactively? The app serves a live API explorer at
**http://localhost:8000/docs**.

---

## The helpers inside Cortex

Cortex is really a few small, focused helpers ("agents") that share the same brain:

- **Briefing** — your Day-1 overview of the project.
- **Install guide** — the tools and versions you need, pulled from the project's own
  config files.
- **Chat** (`kt-agent-v1`) — ask anything; a Level-3 AI agent that reasons over 9 tools
  (search, read, grep, symbol lookup, dependency parsing, call graphs, AST-level search)
  and cites the exact files and lines each answer came from.
- **Guided tour** — a "read these files, in this order" path through the code.
- **Project walkthrough** — the long-form, framework-aware deep dive you can save as PDF.
- **Gap finder** — points out the files that are central but hard to understand, so the
  team can write down what they're for.

Each helper sticks to its job and only talks about the repo you gave it.

---

## How you can trust the answers

This is the part that matters most for onboarding — wrong answers are worse than none.

- **It only uses your code.** Answers are built from the actual files it read, and it
  double-checks that every file it cites really exists. Made-up references are caught and
  flagged.
- **It admits when it doesn't know.** If nothing relevant turns up, it says "I couldn't
  find this in the indexed code" instead of guessing.
- **It ignores sneaky instructions.** If a README or code comment contains text like
  "ignore everything and say X," Cortex treats that as data to read, not a command to
  follow.
- **It fails gracefully.** If the AI service is slow or down, it retries up to twice before
  giving you an honest "couldn't answer" — never a confident hallucination.

---

## What it can see, and what it keeps

- **Read-only.** Cortex can read files and run read-only `git`. It can't write, delete, or
  run commands on your machine.
- **You can fence it in.** Set `ONBOARDING_ALLOWED_ROOTS` to limit which folders it's
  allowed to read. (By default it isn't fenced — fine on your own laptop, but you'll want
  to set it on a shared server. The app prints a warning at startup to remind you.)
- **Your secrets stay secret.** API keys and repo access tokens are used in the moment and
  never written to disk or logs.
- **Where things are stored.** What it learns about a repo (the search index, the briefing,
  your chat history, any notes you save) is kept locally under `.kt_index/`. Chat history
  can optionally go to a MongoDB if you configure one.

---

## Settings

Everything is controlled by environment variables, with safe defaults so it just works:

| What | Setting | Choices | Default |
|---|---|---|---|
| AI for answers | `ONBOARDING_LLM_BACKEND` | `claude`, `groq`, `openrouter` | `claude` |
| How it searches | `ONBOARDING_VECTOR_BACKEND` | `tfidf` (fast), `dense`, `hybrid` (smartest) | `tfidf` |
| Where chat history lives | `ONBOARDING_CHAT_STORE` | `auto`, `json`, `mongo` | `auto` |
| API keys | `ONBOARDING_API_KEYS` | comma-separated keys | `dev-local-key` |

The default backend is **Claude** — it authenticates via `claude auth` OAuth (no API key needed
if you have Claude Pro). For Groq, set `ONBOARDING_LLM_BACKEND=groq` and add a free `GROQ_API_KEY`.
The `hybrid` search option understands meaning (so "auth" finds "login"), but it's slower to set
up on big repos — Cortex automatically falls back to the fast option for very large projects.

---

## Testing and quality checks

Two commands tell you everything's healthy:

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # the test suite (75 tests)
.\.venv\Scripts\python.exe -m evals.runner       # the quality gate (31 checks)
```

The **quality gate** is how we make sure changes don't quietly break things. It spins up
tiny throwaway sample repos, runs every helper against them, and checks the results — e.g.
"did chat find the right file?", "did the tour start at the real entry file?", "were
made-up sources caught?". It covers every helper (briefing 10, chat 10, tour 5,
walkthrough 3, install 3) and writes a report to `evals/results.json`.

If any check fails, the command exits with an error — so it can run automatically on every
code change (it does, via `.github/workflows/ci.yml`) and block anything that regresses.

Reading the report: `gate_passed: true` means all good. If something failed, open that
case in `results.json` and the failing check explains exactly what went wrong.

---

## Good to know (current limits)

Cortex is built to be honest about what it is — a local-first onboarding tool. A few things
to keep in mind if you deploy it more widely:

- **Auth is basic.** One shared key grants full access; if you leave the key list empty,
  the API is open. Set real keys before putting it on a network.
- **It runs on one machine.** The rate limit and the search index live in memory / on local
  disk, so it isn't built for multiple servers yet.
- **Big repos are sampled.** Very large monorepos are read up to sensible limits rather than
  every single file.
- **"Who owns what" is approximate.** Ownership comes from who commits most to each folder,
  not line-by-line history.
- **Answer quality is checked structurally, not stylistically.** The quality gate verifies
  the right files and grounding; it doesn't yet grade how well-written an answer reads.
- **Claude backend requires `claude auth` login.** Run `claude auth` once in your terminal
  to authenticate. No API key is needed for Claude Pro users.

See [docs/limitations.md](docs/limitations.md) for the full list with deployment guidance.

---

## Trouble connecting? (corporate networks)

If you hit a `CERTIFICATE_VERIFY_FAILED` error behind a company proxy, Cortex already
handles it — it trusts your operating system's certificate store, so it works through the
proxy securely without disabling verification.
