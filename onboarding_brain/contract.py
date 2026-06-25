"""Agent I/O contract (Pydantic)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OnboardingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_path: str = Field(..., description="Path to a local repository to brief")


class Sourced(BaseModel):
    answer: str = ""
    sources: list[str] = Field(default_factory=list)


class SetupSteps(BaseModel):
    steps: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class FolderInfo(BaseModel):
    folder: str
    purpose: str = ""
    sources: list[str] = Field(default_factory=list)


class FeatureInfo(BaseModel):
    feature: str
    detail: str = ""
    sources: list[str] = Field(default_factory=list)


class OwnerInfo(BaseModel):
    area: str
    owner: str = ""
    sources: list[str] = Field(default_factory=list)


class GlossaryItem(BaseModel):
    term: str
    meaning: str = ""
    sources: list[str] = Field(default_factory=list)


class Trace(BaseModel):
    trace_id: str
    agent_id: str
    model_used: str
    duration_ms: int
    strategy: str = "llm"               # llm | mock | degraded
    repo_path: str = ""
    is_git_repo: bool = False
    files_scanned: int = 0
    errors: list[str] = Field(default_factory=list)
    grounding: dict[str, Any] | None = None


class OnboardingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overview: Sourced = Field(default_factory=Sourced)
    key_features: list[FeatureInfo] = Field(default_factory=list)
    folder_map: list[FolderInfo] = Field(default_factory=list)
    setup_steps: SetupSteps = Field(default_factory=SetupSteps)
    recent_work: Sourced = Field(default_factory=Sourced)
    owners: list[OwnerInfo] = Field(default_factory=list)
    glossary: list[GlossaryItem] = Field(default_factory=list)
    validation_status: Literal["passed", "warning", "failed"] = "passed"
    trace: Trace


# --------------------------------------------------------------------------- #
# KT platform: ingest + RAG chat
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_path: str = Field("", description="Absolute local path to an already-cloned repo")
    clone_url: str = Field("", description="Git URL to clone (https or ssh) — cloned to a temp dir then ingested")
    clone_token: str = Field("", description="PAT / API token for private repos — injected server-side, never logged")
    namespace: str | None = Field(None, description="Override the index name (defaults to a slug of the repo)")
    rebuild: bool = Field(False, description="Re-index even if already indexed")


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: str
    repo_path: str
    files_indexed: int = 0
    chunks_indexed: int = 0
    already_indexed: bool = False
    briefing_pending: bool = False  # True when briefing is generating in background
    overview: Sourced = Field(default_factory=Sourced)
    starter_questions: list[str] = Field(default_factory=list)
    validation_status: Literal["passed", "warning", "failed"] = "passed"
    trace: Trace


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: str
    question: str
    history: list[ChatTurn] = Field(default_factory=list)
    top_k: int | None = None
    backend: str | None = None        # override active LLM backend (claude/groq/openrouter/mock)
    claude_model: str | None = None   # override Claude model for this request


class AnnotateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    answer: str
    symbol: str = ""


class Source(BaseModel):
    path: str
    language: str = ""
    line_start: int = 0
    line_end: int = 0
    score: float = 0.0
    snippet: str = ""
    symbol: str = ""    # enclosing def/class/heading the chunk starts at
    used: bool = False  # the answer actually cited this snippet (vs. merely retrieved)


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    grounded: bool = True
    validation_status: Literal["passed", "warning", "failed"] = "passed"
    wiring: dict[str, Any] | None = None  # how the cited files connect (for the diagram)
    trace: Trace


# --------------------------------------------------------------------------- #
# Guided codebase tour — ordered, bootstrap-first learning path
# --------------------------------------------------------------------------- #
class TourStop(BaseModel):
    path: str
    symbol: str = ""
    language: str = ""
    line_start: int = 0
    line_end: int = 0
    excerpt: str = ""
    depth: int = 0              # 0 = the entry/main file; deeper = later in the flow
    imports: list[str] = Field(default_factory=list)
    reason: str = ""           # why this file is a stop, in plain English
    note: str = ""             # captured team annotation, if any
    is_entry: bool = False


class TourChapter(BaseModel):
    title: str
    why: str = ""
    stops: list[TourStop] = Field(default_factory=list)


class TourResponse(BaseModel):
    namespace: str
    overview: str = ""
    entry_point: str | None = None
    entry_points: list[str] = Field(default_factory=list)
    total_stops: int = 0
    chapters: list[TourChapter] = Field(default_factory=list)
    wiring: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Project walkthrough — a long-form, framework-aware, plain-English deep dive
# --------------------------------------------------------------------------- #
class WalkthroughSection(BaseModel):
    key: str
    title: str
    body: str = ""                              # Markdown explanation, grounded in the files
    files: list[str] = Field(default_factory=list)  # the real files this section covers


class WalkthroughResponse(BaseModel):
    namespace: str
    title: str = ""
    stack: list[str] = Field(default_factory=list)  # detected frameworks/languages
    sections: list[WalkthroughSection] = Field(default_factory=list)
    wiring: dict[str, Any] | None = None
    generated_with: str = ""                    # model used (or "structural" offline)
    generated_at: str = ""
