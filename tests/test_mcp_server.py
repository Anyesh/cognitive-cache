import os
import subprocess
import tempfile

import pytest

from cognitive_cache.mcp_server import _handle_select_context, _index_cache


def _create_fake_repo(tmpdir):
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)

    os.makedirs(os.path.join(tmpdir, "src"))
    with open(os.path.join(tmpdir, "src/auth.py"), "w") as f:
        f.write("def login(username, password):\n    return True\n")
    with open(os.path.join(tmpdir, "src/db.py"), "w") as f:
        f.write("def get_user(name):\n    return name\n")

    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)


def test_handle_select_context_returns_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        _index_cache.clear()

        result = _handle_select_context(tmpdir, "fix the login bug", 12000)

        assert "files" in result
        assert "total_tokens" in result
        assert "budget" in result
        assert "budget_remaining" in result
        assert len(result["files"]) > 0

        first = result["files"][0]
        assert "path" in first
        assert "score" in first
        assert "signals" in first
        assert "content" in first


def test_handle_select_context_caches_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        _index_cache.clear()

        abs_path = os.path.abspath(tmpdir)

        _handle_select_context(tmpdir, "fix login", 12000)
        assert abs_path in _index_cache

        _handle_select_context(tmpdir, "different task", 12000)
        assert abs_path in _index_cache


def test_handle_select_context_respects_budget():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        _index_cache.clear()

        result = _handle_select_context(tmpdir, "fix login", 50)
        assert result["total_tokens"] <= 50


def test_handle_select_context_nonexistent_repo():
    _index_cache.clear()
    with pytest.raises(FileNotFoundError):
        _handle_select_context("/nonexistent/path", "fix bug", 12000)


def test_handle_select_context_returns_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        _index_cache.clear()

        result = _handle_select_context(tmpdir, "fix login", 12000)

        first = result["files"][0]
        assert "language" in first
        assert "is_test" in first
        assert first["language"] == "python"
        assert first["is_test"] is False


def test_handle_select_context_excludes_tests():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        os.makedirs(os.path.join(tmpdir, "tests"), exist_ok=True)
        with open(os.path.join(tmpdir, "tests", "test_auth.py"), "w") as f:
            f.write("def test_login(): assert True\n")
        subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add tests"], cwd=tmpdir, capture_output=True
        )
        _index_cache.clear()

        result = _handle_select_context(
            tmpdir, "fix login bug", 12000, include_tests=False
        )

        paths = [f["path"] for f in result["files"]]
        assert all("test_" not in p for p in paths)


def test_handle_select_context_max_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        _index_cache.clear()

        result = _handle_select_context(tmpdir, "fix login", 12000, max_files=1)

        assert len(result["files"]) <= 1


def test_handle_select_context_min_score():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        _index_cache.clear()

        result = _handle_select_context(tmpdir, "fix login", 12000, min_score=0.99)

        for f in result["files"]:
            assert f["score"] >= 0.99
