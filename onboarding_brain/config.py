"""Environment-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Single backend by design: this is a PURE AGENT built on the Claude Agent SDK
    # (claude-agent-sdk). The SDK owns the tool-use loop; the app only provides the
    # 9 code-aware tools as an in-process MCP server. There is no alternative LLM
    # backend — the value is the agent, not an LLM-call multiplexer.
    backend: str = field(default_factory=lambda: os.getenv("ONBOARDING_LLM_BACKEND", "claude_sdk").lower())

    # --- Claude Agent SDK backend (the only backend) ---
    # Drives the bundled Claude Code CLI via the claude-agent-sdk. Authenticates
    # with a Claude Pro/Max SUBSCRIPTION over OAuth — no billed API key. Either
    # run `claude setup-token` and set CLAUDE_CODE_OAUTH_TOKEN, or log in once
    # interactively with `claude` (creds at ~/.claude/.credentials.json). Make
    # sure ANTHROPIC_API_KEY is UNSET or it takes precedence and bills the API.
    # Empty model = let the CLI use the subscription's default model.
    claude_sdk_model: str = field(default_factory=lambda: os.getenv("CLAUDE_SDK_MODEL", ""))
    claude_sdk_oauth_token: str = field(default_factory=lambda: os.getenv("CLAUDE_CODE_OAUTH_TOKEN", ""))

    # When True (default, local single-user setup), a tokenless clone of a
    # private repo lets Git Credential Manager pop a browser for OAuth. On a
    # HEADLESS server set this False so a tokenless private clone fails fast
    # instead of hanging up to 5 min on an invisible credential prompt.
    clone_interactive: bool = field(
        default_factory=lambda: os.getenv("ONBOARDING_CLONE_INTERACTIVE", "true").lower()
        in ("1", "true", "yes", "on"))

    # --- Bitbucket re-sync ---
    # Optional service token used to RE-CLONE a private repo when the user hits
    # "Re-sync" (format "user:token" or a bare token, same as clone_token). Set
    # it once and re-sync works for private repos without re-entering anything;
    # local and public repos need no token at all.
    bitbucket_token: str = field(default_factory=lambda: os.getenv("ONBOARDING_BITBUCKET_TOKEN", ""))

    temperature: float = field(default_factory=lambda: _float("ONBOARDING_TEMPERATURE", 0.1))
    # accuracy-first: generous ceiling so thorough answers never truncate
    # (.env raises this further; tokens are not the constraint here)
    max_tokens: int = field(default_factory=lambda: _int("ONBOARDING_MAX_TOKENS", 4000))
    request_timeout_s: float = field(default_factory=lambda: _float("ONBOARDING_REQUEST_TIMEOUT_S", 45.0))
    max_retries: int = field(default_factory=lambda: _int("ONBOARDING_MAX_RETRIES", 2))

    api_keys: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            k.strip() for k in os.getenv("ONBOARDING_API_KEYS", "dev-local-key").split(",") if k.strip()
        )
    )
    max_request_bytes: int = field(default_factory=lambda: _int("ONBOARDING_MAX_REQUEST_BYTES", 16384))
    rate_limit_per_min: int = field(default_factory=lambda: _int("ONBOARDING_RATE_LIMIT_PER_MIN", 60))

    allowed_roots: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            r.strip() for r in os.getenv("ONBOARDING_ALLOWED_ROOTS", "").split(",") if r.strip()
        )
    )

    log_level: str = field(default_factory=lambda: os.getenv("ONBOARDING_LOG_LEVEL", "INFO").upper())
    trace_file: str = field(default_factory=lambda: os.getenv("ONBOARDING_TRACE_FILE", "trace.json"))

    # --- RAG / knowledge index ---
    vector_backend: str = field(default_factory=lambda: os.getenv("ONBOARDING_VECTOR_BACKEND", "tfidf").lower())
    embed_model: str = field(default_factory=lambda: os.getenv("ONBOARDING_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    # above this many chunks, the hybrid backend indexes TF-IDF only (skips the
    # slow CPU dense-embedding pass) so huge repos stay fast. TF-IDF with
    # identifier-splitting already has strong code recall.
    hybrid_max_chunks: int = field(default_factory=lambda: _int("ONBOARDING_HYBRID_MAX_CHUNKS", 4000))
    # cap git commits indexed as searchable history chunks (each must be embedded)
    max_commit_chunks: int = field(default_factory=lambda: _int("ONBOARDING_MAX_COMMIT_CHUNKS", 100))
    index_dir: str = field(default_factory=lambda: os.getenv("ONBOARDING_INDEX_DIR", ".kt_index"))
    # retrieval depth is DYNAMIC: up to top_k candidates are fetched, then a
    # relevance cutoff keeps between min_k and top_k — focused questions get
    # few sharp sources, broad ones keep many.
    # accuracy-first: retrieve a wide candidate pool; the relevance cliff in
    # select_relevant trims it back, so more candidates only helps recall
    retrieval_top_k: int = field(default_factory=lambda: _int("ONBOARDING_RETRIEVAL_TOP_K", 24))
    # min_k=2: never force-pad with irrelevant files; 2 strong hits beat 4 diluted ones
    retrieval_min_k: int = field(default_factory=lambda: _int("ONBOARDING_RETRIEVAL_MIN_K", 2))
    chunk_chars: int = field(default_factory=lambda: _int("ONBOARDING_CHUNK_CHARS", 1200))
    chunk_overlap: int = field(default_factory=lambda: _int("ONBOARDING_CHUNK_OVERLAP", 150))
    max_ingest_files: int = field(default_factory=lambda: _int("ONBOARDING_MAX_INGEST_FILES", 4000))
    max_ingest_file_bytes: int = field(default_factory=lambda: _int("ONBOARDING_MAX_INGEST_FILE_BYTES", 60000))
    # briefing prompt budget (~chars). 48000 ≈ 12k tokens — accuracy-first; the
    # large-context cloud models (gpt-oss-120b) handle this comfortably.
    context_budget_chars: int = field(default_factory=lambda: _int("ONBOARDING_CONTEXT_BUDGET_CHARS", 48000))
    # chat answer prompt budget (~chars). Higher = the model reads more retrieved
    # code per question, the single biggest lever on answer accuracy.
    chat_context_budget_chars: int = field(default_factory=lambda: _int("ONBOARDING_CHAT_CONTEXT_BUDGET_CHARS", 48000))

    # --- Q&A / chat-history storage ---
    # "auto" -> MongoDB when ONBOARDING_MONGO_URI is set (and pymongo imports and
    # the server answers a ping); otherwise JSON files on disk. Force a backend
    # with "json" or "mongo". Any Mongo failure degrades to JSON, never crashes.
    chat_store: str = field(default_factory=lambda: os.getenv("ONBOARDING_CHAT_STORE", "auto").lower())
    mongo_uri: str = field(default_factory=lambda: os.getenv("ONBOARDING_MONGO_URI", ""))
    mongo_db: str = field(default_factory=lambda: os.getenv("ONBOARDING_MONGO_DB", "cortex"))

    @property
    def model_used(self) -> str:
        return f"claude_sdk/{self.claude_sdk_model or 'subscription-default'}"


def get_settings() -> Settings:
    return Settings()
