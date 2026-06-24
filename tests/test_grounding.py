from onboarding_brain.grounding import check_sources


def test_resolved_sources_pass():
    resp = {"overview": {"answer": "x", "sources": ["README.md"]},
            "setup_steps": {"steps": ["npm i"], "sources": ["package.json"]}}
    r = check_sources(resp, ["README.md", "package.json", "git log", "file tree"])
    assert r["validation_status"] == "passed"
    assert r["unresolved_sources"] == []


def test_meta_sources_allowed():
    resp = {"recent_work": {"answer": "...", "sources": ["git log"]},
            "folder_map": [{"folder": "src/", "purpose": "", "sources": ["file tree"]}]}
    assert check_sources(resp, ["src/a.py"])["validation_status"] == "passed"


def test_directory_citation_resolves():
    resp = {"folder_map": [{"folder": "src/", "purpose": "", "sources": ["src"]}]}
    assert check_sources(resp, ["src/app.js", "src/utils/x.js"])["validation_status"] == "passed"


def test_invented_source_flagged():
    resp = {"overview": {"answer": "x", "sources": ["TOTALLY_MADE_UP.md"]}}
    r = check_sources(resp, ["README.md"])
    assert r["validation_status"] == "warning"
    assert "TOTALLY_MADE_UP.md" in r["unresolved_sources"]
