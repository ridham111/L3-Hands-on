import os
import tempfile

# Deterministic, offline backend + isolated trace for all tests.
os.environ["ONBOARDING_LLM_BACKEND"] = "mock"
os.environ["ONBOARDING_LLM_FALLBACK_BACKEND"] = ""  # no provider chaining in tests
os.environ["ONBOARDING_VECTOR_BACKEND"] = "tfidf"  # no model download/embedding in tests
# hermetic chat store: never touch a real MongoDB even if the dev .env sets a URI
os.environ["ONBOARDING_CHAT_STORE"] = "json"
os.environ["ONBOARDING_MONGO_URI"] = ""
os.environ["ONBOARDING_TRACE_FILE"] = os.path.join(tempfile.gettempdir(), "onboarding_test_trace.jsonl")
os.environ["ONBOARDING_ALLOWED_ROOTS"] = ""  # allow temp dirs
# Isolate the knowledge index so tests never touch a real .kt_index.
os.environ["ONBOARDING_INDEX_DIR"] = os.path.join(tempfile.mkdtemp(prefix="kt_idx_"), "index")
