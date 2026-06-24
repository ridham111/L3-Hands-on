"""Onboarding Brain CLI.

    python -m cli.main onboard --repo .
    python -m cli.main onboard --repo C:\\path\\to\\repo --pretty
    python -m cli.main info
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

# Windows consoles default to cp1252, which crashes on Unicode in code
# snippets (arrows, box-drawing, emoji). Force UTF-8, degrade lossily if not.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass

from onboarding_brain import AGENT_ID, __version__
from onboarding_brain.config import get_settings
from onboarding_brain.contract import OnboardingRequest
from onboarding_brain.onboarding import RepoAccessError, generate_briefing

app = typer.Typer(add_completion=False, help="Onboarding Brain — brief a new dev on any local repo")


@app.command()
def onboard(
    repo: Path = typer.Option(Path("."), "--repo", "-r", help="Path to a local repository"),
    pretty: bool = typer.Option(True, help="Pretty-print JSON"),
):
    """Read a repo and print a beginner-friendly briefing as JSON."""
    settings = get_settings()
    try:
        resp = generate_briefing(OnboardingRequest(repo_path=str(repo)), settings=settings)
    except RepoAccessError as exc:
        typer.echo(json.dumps({"error": "repo_access", "detail": str(exc)}), err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(resp.model_dump(), indent=2 if pretty else None, ensure_ascii=False, default=str))
    raise typer.Exit(code=0 if resp.validation_status != "failed" else 1)


@app.command()
def ingest(
    repo: Path = typer.Option(Path("."), "--repo", "-r", help="Repo to index"),
    namespace: str = typer.Option(None, "--namespace", "-n", help="Index name (defaults to repo folder)"),
    rebuild: bool = typer.Option(False, help="Re-index even if already indexed"),
):
    """Index a repo into the knowledge base (vector store)."""
    from onboarding_brain.contract import IngestRequest
    from onboarding_brain.kt.ingest import ingest_repo

    try:
        resp = ingest_repo(IngestRequest(repo_path=str(repo), namespace=namespace, rebuild=rebuild),
                           settings=get_settings())
    except ValueError as exc:
        typer.echo(json.dumps({"error": "ingest_failed", "detail": str(exc)}), err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False, default=str))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the codebase"),
    namespace: str = typer.Option(..., "--namespace", "-n", help="Indexed repo name"),
):
    """Ask the KT chatbot a question about an indexed repo."""
    from onboarding_brain.contract import AskRequest
    from onboarding_brain.kt.chat import ask as kt_ask

    try:
        resp = kt_ask(AskRequest(namespace=namespace, question=question), settings=get_settings())
    except ValueError as exc:
        typer.echo(json.dumps({"error": "ask_failed", "detail": str(exc)}), err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False, default=str))


@app.command()
def namespaces():
    """List indexed repos."""
    from onboarding_brain.kt.store import get_store
    typer.echo(json.dumps(get_store(get_settings()).list_namespaces(), indent=2, default=str))


@app.command()
def info():
    """Print agent + backend configuration."""
    s = get_settings()
    typer.echo(json.dumps({
        "agent_id": AGENT_ID, "version": __version__,
        "backend": s.backend, "model_used": s.model_used,
    }, indent=2))


if __name__ == "__main__":
    app()
