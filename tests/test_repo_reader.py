import shutil

from onboarding_brain.repo_reader import gather_repo_context
from tests.fixtures import NODE_REPO, make_repo


def test_gathers_tree_readme_config():
    repo = make_repo(NODE_REPO)
    try:
        ctx = gather_repo_context(str(repo))
        assert ctx["readme"]["file"] == "README.md"
        assert any(c["file"] == "package.json" for c in ctx["config_files"])
        assert "src" in ctx["top_level_dirs"]
        assert "app.js" in ctx["file_tree"]
        assert "README.md" in ctx["available_sources"]
        assert ctx["file_count"] >= 3
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_non_directory_errors():
    ctx = gather_repo_context("/no/such/path/xyz")
    assert "error" in ctx


def test_skips_heavy_dirs():
    repo = make_repo({"node_modules/x/index.js": "x", "src/a.py": "a", "README.md": "# t"})
    try:
        ctx = gather_repo_context(str(repo))
        assert "node_modules" not in ctx["top_level_dirs"]
    finally:
        shutil.rmtree(repo, ignore_errors=True)
