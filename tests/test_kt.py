"""KT engine: chunking, vector store, ingest, and grounded RAG chat (offline)."""
import shutil
from pathlib import Path

from onboarding_brain.config import get_settings
from onboarding_brain.contract import AskRequest, IngestRequest
from onboarding_brain.kt.chunker import iter_chunks
from onboarding_brain.kt.chat import ask
from onboarding_brain.kt.ingest import ingest_repo
from onboarding_brain.kt.store import TfidfStore, slugify
from tests.fixtures import make_repo

CODE_REPO = {
    "README.md": "# Shop API\n\nA small order service.\n",
    "src/auth.py": "def login(user, password):\n    # validate credentials against the user store\n    return check_password(user, password)\n",
    "src/orders.py": "def create_order(cart):\n    # build an order from the shopping cart and save it\n    return save(cart)\n",
    "package.json": '{"name":"shop","scripts":{"start":"node ."}}',
}


def test_chunker_emits_metadata():
    repo = make_repo(CODE_REPO)
    try:
        chunks = list(iter_chunks(Path(repo), chunk_chars=1200, overlap=100, max_files=50, max_bytes=60000))
        paths = {c["metadata"]["path"] for c in chunks}
        assert "src/auth.py" in paths and "src/orders.py" in paths
        assert all("line_start" in c["metadata"] for c in chunks)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_store_index_and_search():
    s = TfidfStore(get_settings())
    chunks = [
        {"id": "a#0", "text": "def login(user, password): validate credentials", "metadata": {"path": "src/auth.py", "line_start": 1, "line_end": 3, "language": "Python"}},
        {"id": "b#0", "text": "def create_order(cart): save order from shopping cart", "metadata": {"path": "src/orders.py", "line_start": 1, "line_end": 3, "language": "Python"}},
    ]
    s.index("nstest", chunks, {"repo_path": "/x"})
    assert s.exists("nstest")
    hits = s.search("nstest", "how do users log in with a password", k=2)
    assert hits and hits[0]["metadata"]["path"] == "src/auth.py"


def test_ingest_then_ask_is_grounded():
    repo = make_repo(CODE_REPO)
    try:
        ing = ingest_repo(IngestRequest(repo_path=str(repo), namespace="shoptest", rebuild=True))
        assert ing.chunks_indexed >= 3
        assert ing.starter_questions
        r = ask(AskRequest(namespace="shoptest", question="how is login handled?"))
        assert r.validation_status == "passed"
        assert r.grounded is True
        # the auth file should be among retrieved sources
        assert any("auth.py" in s.path for s in r.sources)
        # offline answer cites real retrieved paths only -> no hallucinated sources
        assert r.trace.grounding["hallucinated_sources"] == []
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_ask_unindexed_namespace_errors():
    import pytest
    # Use a real code question (not a greeting): conversational messages like
    # "hi" short-circuit before the namespace check and never need an index.
    with pytest.raises(ValueError):
        ask(AskRequest(namespace="does-not-exist", question="how does authentication work?"))


def test_slugify():
    assert slugify("My Repo!") == "my-repo"


def test_chunker_aligns_to_function_boundaries():
    body = (
        "import os\n\n"
        "def alpha():\n" + "    a = 1\n" * 6 + "    return a\n\n"
        "def beta():\n" + "    b = 2\n" * 6 + "    return b\n\n"
        "def gamma():\n" + "    c = 3\n" * 6 + "    return c\n"
    )
    repo = make_repo({"src/funcs.py": body})
    try:
        chunks = list(iter_chunks(Path(repo), chunk_chars=120, overlap=30, max_files=10, max_bytes=60000))
        ours = [c for c in chunks if c["metadata"]["path"] == "src/funcs.py"]
        assert len(ours) >= 3
        # every chunk after the first opens at a def boundary, not mid-function
        for c in ours[1:]:
            assert c["text"].lstrip().startswith("def "), c["text"][:40]
        # symbols anchor the chunks and line numbers point at the real def lines
        beta = next(c for c in ours if c["metadata"]["symbol"].startswith("def beta"))
        lines = body.splitlines()
        assert lines[beta["metadata"]["line_start"] - 1].startswith("def beta")
        # index_text carries the path for retrieval
        assert ours[0]["index_text"].startswith("src/funcs.py")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_chunker_survives_monster_lines():
    repo = make_repo({"data/blob.json": '{"k":"' + "x" * 10000 + '"}'})
    try:
        chunks = list(iter_chunks(Path(repo), chunk_chars=1000, overlap=150, max_files=10, max_bytes=60000))
        assert chunks  # terminated, produced bounded chunks
        assert all(len(c["text"]) <= 2000 for c in chunks)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_rrf_merge_prefers_agreement():
    from onboarding_brain.kt.hybrid_store import rrf_merge

    a = [{"id": "x", "score": 0.9, "text": "", "metadata": {}},
         {"id": "y", "score": 0.5, "text": "", "metadata": {}}]
    b = [{"id": "z", "score": 0.8, "text": "", "metadata": {}},
         {"id": "y", "score": 0.7, "text": "", "metadata": {}}]
    merged = rrf_merge([a, b], k=3)
    # y appears in both lists -> outranks the single-list winners
    assert merged[0]["id"] == "y"
    assert {m["id"] for m in merged} == {"x", "y", "z"}


def test_broad_question_routes_to_briefing():
    repo = make_repo(CODE_REPO)
    try:
        ingest_repo(IngestRequest(repo_path=str(repo), namespace="broadtest", rebuild=True))
        r = ask(AskRequest(namespace="broadtest", question="what does this project do?"))
        assert any(s.path == "project-briefing" for s in r.sources)
        assert r.grounded is True
        # specific questions don't get the briefing injected
        r2 = ask(AskRequest(namespace="broadtest", question="how is login handled?"))
        assert all(s.path != "project-briefing" for s in r2.sources)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_spec_files_rank_below_implementation():
    from onboarding_brain.kt.store import rerank

    results = [
        {"id": "spec", "score": 0.6, "text": "", "metadata": {"path": "src/app/auth/authentication.service.spec.ts"}},
        {"id": "impl", "score": 0.5, "text": "", "metadata": {"path": "src/app/auth/authentication.service.ts"}},
        {"id": "tests-dir", "score": 0.45, "text": "", "metadata": {"path": "tests/test_auth.py"}},
    ]
    out = rerank([dict(r) for r in results], "how is authentication handled", k=3)
    assert out[0]["id"] == "impl"  # spec penalized below the implementation
    assert out[1]["id"] == "spec"
    # explicit test questions: no penalty, so the spec stays above the impl
    out = rerank([dict(r) for r in results], "where are the auth tests", k=3)
    ids = [r["id"] for r in out]
    assert ids.index("spec") < ids.index("impl")


def test_self_folder_excluded_when_nested(tmp_path):
    """When KT Brain's own folder sits inside the target repo, it must not be
    indexed — its prompts/evals contain users' exact question phrases."""
    from onboarding_brain.kt import chunker

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main(): pass\n", encoding="utf-8")
    (tmp_path / "tool").mkdir()
    (tmp_path / "tool" / "prompts.py").write_text("What does this project do?\n", encoding="utf-8")
    original = chunker._SELF_ROOT
    chunker._SELF_ROOT = (tmp_path / "tool").resolve()
    try:
        paths = {c["metadata"]["path"] for c in
                 chunker.iter_chunks(tmp_path, chunk_chars=1200, overlap=100, max_files=50, max_bytes=60000)}
    finally:
        chunker._SELF_ROOT = original
    assert "src/app.py" in paths
    assert not any(p.startswith("tool/") for p in paths)


def test_filename_match_boosts_ranking():
    from onboarding_brain.kt.store import rerank

    results = [
        {"id": "i18n", "score": 0.12, "text": "", "metadata": {"path": "src/assets/i18n/en.json"}},
        {"id": "svc", "score": 0.10, "text": "", "metadata": {"path": "src/app/auth/authentication.service.ts"}},
    ]
    out = rerank([dict(r) for r in results], "how is authentication handled?", k=2)
    assert out[0]["id"] == "svc"  # filename matches "authentication" -> boosted past noise
    # no meaningful term overlap -> original order kept
    out = rerank([dict(r) for r in results], "where is the configuration?", k=2)
    assert out[0]["id"] == "i18n"


def test_tool_copies_excluded_by_marker(tmp_path):
    """ANY copy of the tool nested in the target repo is skipped — detected by
    its package marker, not just this install's path."""
    from onboarding_brain.kt import chunker

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main(): pass\n", encoding="utf-8")
    nested = tmp_path / "old-copy" / "onboarding_brain"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "old-copy" / "prompts.py").write_text("What does this project do?\n", encoding="utf-8")
    paths = {c["metadata"]["path"] for c in
             chunker.iter_chunks(tmp_path, chunk_chars=1200, overlap=100, max_files=50, max_bytes=60000)}
    assert "src/app.py" in paths
    assert not any(p.startswith("old-copy/") for p in paths)


def test_filename_candidates_locates_named_files():
    from onboarding_brain.kt.chat import filename_candidates

    known = {"src/main.ts", "src/app/app.module.ts", "src/styles.scss",
             "src/app/api/api.service.ts"}
    out = filename_candidates(known, "Where is the entry point / main file?", exclude=set())
    assert "src/main.ts" in out
    out = filename_candidates(known, "where are the api endpoints?", exclude=set())
    assert "src/app/api/api.service.ts" in out
    # no meaningful term overlap -> nothing injected
    assert filename_candidates(known, "how does it work?", exclude=set()) == []


def test_standalone_question_not_condensed_on_live_backend():
    from onboarding_brain.config import Settings
    from onboarding_brain.kt.chat import _condense_question

    history = [{"role": "user", "content": "how do I run it locally?"}]
    q = "How is authentication handled in the application?"
    # provider=None proves no LLM call is attempted for standalone questions
    out = _condense_question(q, history, provider=None, settings=Settings(), errors=[])
    assert out == q


def test_dense_index_is_incremental(monkeypatch, tmp_path):
    """Re-indexing reuses vectors of unchanged chunks — only changed text is
    re-embedded (this is what makes hybrid rebuilds fast)."""
    import numpy as np

    from onboarding_brain.config import Settings
    from onboarding_brain.kt import dense_store

    calls = []

    class FakeEmbedder:
        def embed(self, texts):
            calls.append(len(texts))
            return [np.full(8, float(len(t) + 1), dtype=np.float32) for t in texts]

    monkeypatch.setattr(dense_store, "_embedder", lambda name: FakeEmbedder())
    s = dense_store.DenseStore(Settings(index_dir=str(tmp_path)))
    chunks = [
        {"id": "a#0", "text": "alpha code", "metadata": {"path": "a.py"}},
        {"id": "b#0", "text": "beta code", "metadata": {"path": "b.py"}},
    ]
    s.index("inc", chunks, {})
    assert sum(calls) == 2
    # unchanged -> zero new embeddings
    calls.clear()
    s.index("inc", chunks, {})
    assert sum(calls) == 0
    # one chunk changed -> exactly one embedding
    calls.clear()
    chunks[1]["text"] = "beta code v2"
    s.index("inc", chunks, {})
    assert sum(calls) == 1


def test_select_relevant_is_dynamic():
    from onboarding_brain.kt.chat import select_relevant

    mk = lambda i, s: {"id": f"f{i}", "score": s, "text": "", "metadata": {}}
    # steep score cliff -> only the sharp hits survive
    steep = [mk(0, .50), mk(1, .45), mk(2, .40), mk(3, .38), mk(4, .05), mk(5, .04)]
    assert len(select_relevant(steep, 3, 16)) == 4
    # flat decay (broad question) -> keep many, up to the cap
    flat = [mk(i, .30 - i * .005) for i in range(20)]
    assert len(select_relevant(flat, 3, 16)) == 16
    # fewer hits than the minimum -> keep them all
    assert len(select_relevant(steep[:2], 3, 16)) == 2


def test_neighbor_expansion_widens_context(tmp_path):
    from onboarding_brain.config import Settings
    from onboarding_brain.kt.chat import _expand_neighbors

    s = TfidfStore(Settings(index_dir=str(tmp_path)))
    chunks = [
        {"id": "f.py#0", "text": "ALPHA-BLOCK", "metadata": {"path": "f.py", "chunk_index": 0, "line_start": 1, "line_end": 10}},
        {"id": "f.py#1", "text": "BETA-BLOCK", "metadata": {"path": "f.py", "chunk_index": 1, "line_start": 11, "line_end": 20}},
    ]
    s.index("nbtest", chunks, {})
    hit = {"id": "f.py#1", "score": .5, "text": "BETA-BLOCK", "metadata": dict(chunks[1]["metadata"])}
    out = _expand_neighbors(s, "nbtest", [hit])
    assert "ALPHA-BLOCK" in out[0]["context_text"]   # neighbor spliced in for the LLM
    assert out[0]["context_lines"][0] == 1           # widened line range
    assert out[0]["text"] == "BETA-BLOCK"            # display snippet unchanged


def test_chat_history_persists_and_clears():
    from onboarding_brain.kt.chat import clear_chat_history, load_chat_history

    repo = make_repo(CODE_REPO)
    try:
        ingest_repo(IngestRequest(repo_path=str(repo), namespace="histtest", rebuild=True))
        ask(AskRequest(namespace="histtest", question="how is login handled?"))
        ask(AskRequest(namespace="histtest", question="where are orders created?"))
        turns = load_chat_history("histtest")
        assert len(turns) == 4
        assert turns[0]["role"] == "user" and "login" in turns[0]["content"]
        assert turns[1]["role"] == "assistant" and turns[1]["content"]
        clear_chat_history("histtest")
        assert load_chat_history("histtest") == []
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_chat_store_defaults_to_json(tmp_path):
    from onboarding_brain.config import Settings
    from onboarding_brain.kt.chat_store import JsonChatStore, get_chat_store

    s = get_chat_store(Settings(index_dir=str(tmp_path)))
    assert isinstance(s, JsonChatStore)


def test_chat_store_degrades_to_json_when_mongo_unreachable(tmp_path):
    from onboarding_brain.config import Settings
    from onboarding_brain.kt.chat_store import JsonChatStore, get_chat_store

    # a URI that cannot connect (or pymongo absent) must fall back, never raise
    s = get_chat_store(Settings(index_dir=str(tmp_path),
                                chat_store="mongo",
                                mongo_uri="mongodb://127.0.0.1:1/"))
    assert isinstance(s, JsonChatStore)
    s.append("nsx", "q1", "a1")
    turns = s.load("nsx")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    s.clear("nsx")
    assert s.load("nsx") == []


class _FakeMongoCol:
    """Minimal in-memory stand-in honoring the $push/$each/$slice we rely on."""

    def __init__(self):
        self.docs: dict = {}

    def find_one(self, flt, proj=None):
        return self.docs.get(flt["_id"])

    def update_one(self, flt, update, upsert=False):
        doc = self.docs.setdefault(flt["_id"], {"_id": flt["_id"], "turns": []})
        push = update["$push"]["turns"]
        doc["turns"].extend(push["$each"])
        sl = push.get("$slice")
        if isinstance(sl, int) and sl < 0:
            doc["turns"] = doc["turns"][sl:]

    def delete_one(self, flt):
        self.docs.pop(flt["_id"], None)


def test_chat_store_persists_wiring_on_assistant_turn(tmp_path):
    from onboarding_brain.config import Settings
    from onboarding_brain.kt.chat_store import JsonChatStore

    s = JsonChatStore(Settings(index_dir=str(tmp_path)))
    wiring = {"nodes": [{"id": "a", "label": "a.ts"}, {"id": "b", "label": "b.ts"}],
              "edges": [{"from": "a", "to": "b", "kind": "import"}]}
    s.append("nsw", "how do files connect?", "A imports B.", extra={"wiring": wiring})
    turns = s.load("nsw")
    assert turns[0]["role"] == "user"  # user turn carries no diagram
    assert "wiring" not in turns[0]
    assert turns[1]["role"] == "assistant"
    assert turns[1]["wiring"]["edges"][0]["kind"] == "import"  # diagram survives reload


def test_mongo_chat_store_caps_and_roundtrips():
    from onboarding_brain.kt.chat_store import _HISTORY_CAP, MongoChatStore

    store = MongoChatStore(_FakeMongoCol())
    for i in range(50):  # 50 calls * 2 turns = 100, must cap to last _HISTORY_CAP
        store.append("ns", f"q{i}", f"a{i}")
    turns = store.load("ns")
    assert len(turns) == _HISTORY_CAP  # 80
    assert turns[-1] == {"role": "assistant", "content": "a49", "ts": turns[-1]["ts"]}
    assert turns[0]["role"] == "user" and turns[0]["content"] == "q10"  # oldest 20 dropped
    store.clear("ns")
    assert store.load("ns") == []


def test_enrich_folds_i18n_labels():
    # i18n labels are folded into index_text; commit subjects are intentionally
    # NOT (that broke dense-embed reuse on re-sync — history lives in its own chunks)
    from onboarding_brain.kt.enrich import enrich_chunks

    chunks = [
        {"id": "a#0", "text": "title = translate('RANK_BY_STORE')",
         "index_text": "src/rank.ts\ntitle = translate('RANK_BY_STORE')",
         "metadata": {"path": "src/rank.ts"}},
    ]
    labels = {"RANK_BY_STORE": "Rank by Store"}
    n = enrich_chunks(chunks, labels)
    assert n == 1
    it = chunks[0]["index_text"]
    assert "Rank by Store" in it          # UI label folded in
    assert chunks[0]["text"] == "title = translate('RANK_BY_STORE')"  # display text untouched


def test_commit_chunks_use_stable_hash_keys():
    # commit chunks must be keyed by hash (stable across re-sync), not position
    import subprocess
    from onboarding_brain.kt.enrich import build_commit_chunks

    repo = make_repo({"a.py": "x = 1\n"})
    try:
        for cmd in (["init"], ["add", "-A"], ["-c", "user.email=t@t.co", "-c", "user.name=t",
                    "commit", "-m", "first commit"]):
            subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True)
        chunks = build_commit_chunks(repo, max_commits=10)
        if chunks:  # git may be unavailable in some envs
            import re as _re
            suffix = chunks[0]["id"].split("#", 1)[1]
            # a git short hash (>=6 hex chars), not a 0-based positional index
            assert _re.fullmatch(r"[0-9a-f]{6,}", suffix)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_i18n_label_builder(tmp_path):
    from onboarding_brain.kt.enrich import build_i18n_labels

    (tmp_path / "src" / "assets" / "i18n").mkdir(parents=True)
    (tmp_path / "src" / "assets" / "i18n" / "en.json").write_text(
        '{"RANK_BY_STORE": "Rank by Store", "nested": {"LASSO": "Lasso Selection"}}', encoding="utf-8")
    labels = build_i18n_labels(tmp_path)
    assert labels.get("RANK_BY_STORE") == "Rank by Store"
    assert labels.get("LASSO") == "Lasso Selection"


def test_knowledge_gaps_and_annotations(tmp_path):
    from onboarding_brain.config import Settings
    from onboarding_brain.kt import knowledge
    from onboarding_brain.kt.store import get_store

    s = Settings(index_dir=str(tmp_path))
    chunks = [
        # cryptic, undocumented, referenced -> should be asked about
        {"id": "x#0", "text": "class XfrmSvc { run() { XfrmSvc.go() } }",
         "metadata": {"path": "src/XfrmSvc.ts", "symbol": "class XfrmSvc", "line_start": 1, "line_end": 3}},
        {"id": "r#0", "text": "// the readme\nclear documented helper",
         "metadata": {"path": "README.md", "symbol": "", "line_start": 1, "line_end": 2}},
    ]
    get_store(s).index("gaptest", chunks, {})
    gaps = knowledge.detect_gaps("gaptest", settings=s)
    assert any("XfrmSvc" in g["file"] for g in gaps)

    knowledge.save_annotation("gaptest", "src/XfrmSvc.ts",
                              "The transform service reconciles nightly billing exports.",
                              symbol="class XfrmSvc", settings=s)
    # answered file no longer asked about
    assert not any("XfrmSvc" in g["file"] for g in knowledge.detect_gaps("gaptest", settings=s))
    # annotation becomes a retrievable chunk
    ac = knowledge.annotation_chunks("gaptest", settings=s)
    assert ac and "billing" in ac[0]["text"]


def test_wiring_builds_import_graph(tmp_path):
    from onboarding_brain.config import Settings
    from onboarding_brain.kt.store import get_store
    from onboarding_brain.kt.wiring import build_wiring

    s = Settings(index_dir=str(tmp_path))
    chunks = [
        {"id": "c#0", "text": "import { OrderService } from './order.service';\nexport class CartComponent {}",
         "metadata": {"path": "src/cart.component.ts", "line_start": 1, "line_end": 2}},
        {"id": "s#0", "text": "export class OrderService { save() {} }",
         "metadata": {"path": "src/order.service.ts", "line_start": 1, "line_end": 1}},
    ]
    get_store(s).index("wiretest", chunks, {})
    w = build_wiring("wiretest", ["src/cart.component.ts", "src/order.service.ts"], settings=s)
    assert w and len(w["nodes"]) == 2
    roles = {n["label"]: n["role"] for n in w["nodes"]}
    assert roles["cart.component.ts"] == "component" and roles["order.service.ts"] == "service"
    imp = [e for e in w["edges"] if e["kind"] == "import"]
    assert {(e["from"], e["to"]) for e in imp} >= {("src/cart.component.ts", "src/order.service.ts")}
    # single file -> nothing to draw
    assert build_wiring("wiretest", ["src/cart.component.ts"], settings=s) is None

    # unrelated files (no imports, different folders) still get connected
    chunks2 = [
        {"id": "a#0", "text": "export class A {}", "metadata": {"path": "src/x/a.ts", "line_start": 1, "line_end": 1}},
        {"id": "b#0", "text": "export class B {}", "metadata": {"path": "src/y/b.ts", "line_start": 1, "line_end": 1}},
    ]
    get_store(s).index("wiretest2", chunks2, {})
    w2 = build_wiring("wiretest2", ["src/x/a.ts", "src/y/b.ts"], settings=s)
    assert w2 and len(w2["edges"]) >= 1  # hub fallback connects them


def test_feature_surface_lists_real_areas(tmp_path):
    from onboarding_brain.config import Settings
    from onboarding_brain.kt import knowledge
    from onboarding_brain.kt.store import get_store

    s = Settings(index_dir=str(tmp_path))
    chunks = [
        {"id": f"f{i}", "text": f"export class Comp{i} {{ run() {{ return doWork(); }} }}",
         "metadata": {"path": p, "line_start": 1, "line_end": 1}}
        for i, p in enumerate([
            "src/app/rank-by-store/a.ts", "src/app/rank-by-store/b.ts",
            "src/app/asset-review/c.ts", "src/app/checkout/d.ts",
            "src/app/shared/util.ts",  # generic -> excluded
        ])
    ]
    get_store(s).index("feattest", chunks, {})
    fm = knowledge.feature_surface("feattest", settings=s)
    txt = fm["text"].lower()
    assert "rank by store" in txt and "asset review" in txt and "checkout" in txt
    assert "shared" not in txt  # generic scaffolding excluded


def test_general_note_only_on_setup_questions():
    from onboarding_brain.kt.chat import _SETUP_Q
    assert _SETUP_Q.search("how do I run it locally?")
    assert _SETUP_Q.search("what version of node is needed?")
    assert not _SETUP_Q.search("what does this project do?")
    assert not _SETUP_Q.search("how is authentication handled?")


def test_classify_citations_inferred_vs_invented():
    from onboarding_brain.kt.chat import classify_citations

    retrieved = {"src/app/login.component.ts"}
    known = {"src/app/login.component.ts", "src/app/auth/authentication.service.ts"}
    halluc, inferred = classify_citations(
        ["src/app/login.component.ts",            # retrieved -> fine
         "src/app/auth/authentication.service",   # real file, import-style citation -> inferred
         "src/app/made-up.service.ts"],           # not in the repo -> hallucinated
        retrieved, known)
    assert halluc == ["src/app/made-up.service.ts"]
    assert inferred == ["src/app/auth/authentication.service"]


def test_condense_offline_resolves_followup():
    from onboarding_brain.kt.chat import condense_question_offline

    history = [{"role": "user", "content": "where are orders created from the cart?"},
               {"role": "assistant", "content": "In src/orders.py."}]
    out = condense_question_offline("which file is that in?", history)
    assert "orders" in out
    # standalone questions pass through untouched
    assert condense_question_offline("how does login work?", history) == "how does login work?"


def test_ns_dir_rejects_path_traversal():
    import pytest
    s = TfidfStore(get_settings())
    for bad in ("..", ".", "", "a/b", "a\\b", "../../etc"):
        with pytest.raises(ValueError):
            s.ns_dir(bad)


def test_ingest_blocked_outside_allowed_roots(tmp_path):
    import pytest
    from onboarding_brain.config import Settings
    from onboarding_brain.kt.store import get_store
    from onboarding_brain.onboarding import RepoAccessError

    repo = make_repo(CODE_REPO)
    try:
        s = Settings(allowed_roots=(str(tmp_path / "only-here"),))
        with pytest.raises(RepoAccessError):
            ingest_repo(IngestRequest(repo_path=str(repo), namespace="forbiddentest", rebuild=True), settings=s)
        # the check fired before indexing — nothing was persisted
        assert not get_store(s).exists("forbiddentest")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Regression tests for the peak-accuracy audit fixes
# --------------------------------------------------------------------------- #
def test_namespace_from_clone_url():
    from onboarding_brain.kt.ingest import _namespace_from_clone_url

    assert _namespace_from_clone_url("https://github.com/org/my-repo.git") == "my-repo"
    assert _namespace_from_clone_url("git@github.com:org/my-repo.git") == "my-repo"
    assert _namespace_from_clone_url("https://host/team/sub/some.repo/") == "some.repo"
    assert _namespace_from_clone_url("https://dev.azure.com/org/proj/_git/Repo") == "Repo"
    assert _namespace_from_clone_url("") == ""


def test_build_response_tolerates_scalar_fields():
    # an LLM that returns {"overview": "a string"} (wrong shape) must not crash
    from onboarding_brain.onboarding import _build_response

    parsed = {"overview": "just a string", "setup_steps": "nope", "recent_work": 123,
              "key_features": "x", "folder_map": None, "owners": "y", "glossary": "z"}
    ctx = {"repo_path": "/x", "available_sources": [], "is_git_repo": False, "file_count": 0}
    resp = _build_response(parsed, ctx, [], "mock", "tr_x", get_settings(), 10)
    assert resp.overview.answer == ""  # coerced, no AttributeError
    assert resp.key_features == [] and resp.owners == []


def test_sources_strictly_scoped_to_answer():
    # the displayed sources must never include synthetic context blocks, and
    # must never be the whole retrieval pool dumped via a fallback
    repo = make_repo(CODE_REPO)
    try:
        ingest_repo(IngestRequest(repo_path=str(repo), namespace="scoped", rebuild=True))
        r = ask(AskRequest(namespace="scoped", question="how does login work?"))
        paths = {s.path for s in r.sources}
        assert not (paths & {"project-briefing", "feature-map", "git-history"})
        assert len(r.sources) <= 3  # scoped to cited hits, not every indexed file
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_resync_unindexed_raises():
    import pytest

    from onboarding_brain.kt.ingest import resync_namespace

    with pytest.raises(ValueError):
        resync_namespace("never-indexed-namespace")


def test_already_indexed_clone_is_not_recloned():
    # re-ingesting an already-indexed namespace must serve from cache WITHOUT
    # cloning — a bogus clone URL would error if a clone were attempted
    repo = make_repo(CODE_REPO)
    try:
        ingest_repo(IngestRequest(repo_path=str(repo), namespace="dedup", rebuild=True))
        resp = ingest_repo(IngestRequest(clone_url="https://example.invalid/x.git", namespace="dedup"))
        assert resp.already_indexed is True
        assert resp.chunks_indexed == 0
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_resync_local_repo():
    # a locally-ingested repo re-syncs from its path with no token needed
    from onboarding_brain.kt.ingest import resync_namespace

    repo = make_repo(CODE_REPO)
    try:
        ingest_repo(IngestRequest(repo_path=str(repo), namespace="resynctest", rebuild=True))
        res = resync_namespace("resynctest")
        assert res["mode"] == "local"
        assert res["status"] in ("resyncing", "already_resyncing")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_guided_tour_builds():
    from onboarding_brain.kt.tour import build_tour

    repo = make_repo(CODE_REPO)
    try:
        ingest_repo(IngestRequest(repo_path=str(repo), namespace="tourtest", rebuild=True))
        t = build_tour("tourtest")
        assert t["chapters"], "tour should produce at least one chapter"
        stops = [s for ch in t["chapters"] for s in ch["stops"]]
        assert stops and all(s.get("path") and "excerpt" in s for s in stops)
        assert t["entry_point"]  # always picks a first file
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_full_file_reconstructs_whole_file():
    # the file viewer needs the WHOLE file, stitched back from its chunks
    repo = make_repo({"README.md": "# X\n\nApp.\n",
                      "src/big.py": "\n".join("def f%d():\n    return %d" % (i, i) for i in range(30))})
    from onboarding_brain.kt.store import get_store
    try:
        ingest_repo(IngestRequest(repo_path=str(repo), namespace="fftest", rebuild=True))
        store = get_store(get_settings())
        f = store.full_file("fftest", "src/big.py")
        assert f and "def f0(" in f["content"] and "def f29(" in f["content"]  # first AND last
        assert store.full_file("fftest", "does/not/exist.py") is None
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_walkthrough_builds_and_detects_stack():
    from onboarding_brain.kt.walkthrough import build_walkthrough

    repo = make_repo({
        "README.md": "# Svc\n\nAn order service.\n", "requirements.txt": "fastapi\nuvicorn\n",
        "app/server.py": "from fastapi import FastAPI\napp = FastAPI()\nif __name__ == '__main__':\n    import uvicorn; uvicorn.run(app)\n",
        "app/services.py": "def work():\n    return 1\n",
    })
    try:
        ingest_repo(IngestRequest(repo_path=str(repo), namespace="walktest", rebuild=True))
        doc = build_walkthrough("walktest")
        assert "FastAPI" in doc["stack"]
        assert len(doc["sections"]) >= 4
        assert all("title" in s and "body" in s for s in doc["sections"])
        # offline (mock) backend produces the structural fallback, still grounded in real files
        assert any(s.get("files") for s in doc["sections"])
    finally:
        shutil.rmtree(repo, ignore_errors=True)
