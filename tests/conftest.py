import os
import tempfile

import pytest

# Use claude backend for tests; disable fallback chaining.
os.environ.setdefault("ONBOARDING_LLM_BACKEND", "claude")
os.environ["ONBOARDING_LLM_FALLBACK_BACKEND"] = ""  # no provider chaining in tests
os.environ["ONBOARDING_VECTOR_BACKEND"] = "tfidf"  # no model download/embedding in tests
# hermetic chat store: never touch a real MongoDB even if the dev .env sets a URI
os.environ["ONBOARDING_CHAT_STORE"] = "json"
os.environ["ONBOARDING_MONGO_URI"] = ""
os.environ["ONBOARDING_TRACE_FILE"] = os.path.join(tempfile.gettempdir(), "onboarding_test_trace.jsonl")
os.environ["ONBOARDING_ALLOWED_ROOTS"] = ""  # allow temp dirs
# Isolate the knowledge index so tests never touch a real .kt_index.
os.environ["ONBOARDING_INDEX_DIR"] = os.path.join(tempfile.mkdtemp(prefix="kt_idx_"), "index")


@pytest.fixture(autouse=True, scope="session")
def _use_stub_provider():
    """Hermetic tests: route every agent flow through the deterministic stub
    provider (test double, not a user backend) — no network, no API key, fully
    repeatable. patch_source=False keeps the real `providers.get_provider` so the
    provider-selection tests (FallbackProvider / OpenRouterProvider) still pass."""
    from evals.stub_provider import install_stub
    from onboarding_brain.config import get_settings
    install_stub(get_settings(), patch_source=False)
    yield
