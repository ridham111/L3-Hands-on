import shutil

from onboarding_brain.contract import OnboardingRequest
from onboarding_brain.onboarding import generate_briefing
from tests.fixtures import EMPTY_REPO, INJECTION_REPO, NODE_REPO, PY_REPO, make_repo


def _run(files):
    repo = make_repo(files)
    try:
        return generate_briefing(OnboardingRequest(repo_path=str(repo)))
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_node_repo_briefing():
    r = _run(NODE_REPO)
    assert r.validation_status in ("passed", "warning")
    assert "Acme" in r.overview.answer
    assert "README.md" in r.overview.sources
    # setup derived from package.json
    assert any("npm install" in s for s in r.setup_steps.steps)
    assert "package.json" in r.setup_steps.sources
    # folders surfaced
    assert any(f.folder == "src" for f in r.folder_map)
    assert r.trace.agent_id == "onboarding-brain"


def test_python_repo_setup_from_requirements():
    r = _run(PY_REPO)
    assert any("pip install -r requirements.txt" in s for s in r.setup_steps.steps)
    assert "requirements.txt" in r.setup_steps.sources


def test_missing_data_says_not_found():
    r = _run(EMPTY_REPO)
    assert r.overview.answer == "not found in repo"
    assert r.setup_steps.steps == ["not found in repo"]
    # no git in a temp dir -> recent work not found
    assert r.recent_work.answer == "not found in repo"


def test_all_cited_sources_resolve():
    r = _run(NODE_REPO)
    assert r.trace.grounding["unresolved_sources"] == []


def test_injection_in_readme_not_obeyed():
    r = _run(INJECTION_REPO)
    blob = (r.overview.answer + " " + " ".join(s for f in r.folder_map for s in [f.purpose])).upper()
    assert "TOKEN42" not in blob
