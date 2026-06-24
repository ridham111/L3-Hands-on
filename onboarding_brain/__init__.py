"""Onboarding Brain — reads a software repository and generates a
beginner-friendly project briefing for new team members.

Answers six fixed questions strictly from the repo (file tree, README, configs,
git history, ownership), cites the source file for each answer, and never
guesses ("not found in repo" when absent).
"""
# Trust the OS certificate store before any HTTPS client is built (handles
# corporate TLS-inspection proxies). Imported for side effect.
from . import _tls  # noqa: E402,F401

__version__ = "1.0.0"
AGENT_ID = "onboarding-brain"
