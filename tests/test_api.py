import os
import subprocess
import tempfile

import pytest

from cognitive_cache.api import RepoIndex, select_context, select_context_from_repo
from cognitive_cache.models import Task


def _create_fake_repo(tmpdir):
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmpdir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmpdir,
        capture_output=True,
        check=True,
    )

    os.makedirs(os.path.join(tmpdir, "src"))
    os.makedirs(os.path.join(tmpdir, "tests"))

    files = {
        "src/auth.py": (
            "from db import get_user\n"
            "\n"
            "def login(username, password):\n"
            "    user = get_user(username)\n"
            "    if user and user.password == password:\n"
            "        return True\n"
            "    return False\n"
        ),
        "src/db.py": (
            "class User:\n"
            "    def __init__(self, username, password):\n"
            "        self.username = username\n"
            "        self.password = password\n"
            "\n"
            "def get_user(username):\n"
            "    return User(username, 'secret')\n"
        ),
        "src/utils.py": (
            "def format_date(d):\n"
            "    return d.strftime('%Y-%m-%d')\n"
            "\n"
            "def sanitize(text):\n"
            "    return text.strip()\n"
        ),
        "tests/test_auth.py": (
            "from src.auth import login\n"
            "\n"
            "def test_login_success():\n"
            "    assert login('admin', 'secret') == True\n"
            "\n"
            "def test_login_failure():\n"
            "    assert login('admin', 'wrong') == False\n"
        ),
    }
    for path, content in files.items():
        full = os.path.join(tmpdir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)

    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmpdir,
        capture_output=True,
        check=True,
    )


# -- RepoIndex.build tests --


class TestRepoIndexBuild:
    def test_indexes_sources(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        assert len(index.sources) >= 4
        paths = {s.path for s in index.sources}
        assert "src/auth.py" in paths
        assert "src/db.py" in paths

    def test_has_recency_data(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        assert isinstance(index.recency_data, dict)
        assert len(index.recency_data) > 0

    def test_has_graph(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        assert index.graph is not None
        assert "src/auth.py" in index.graph.nodes

    def test_has_fitted_embedding(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        assert index.embedding_signal._fitted is True

    def test_stores_mtimes(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        assert len(index.file_mtimes) == len(index.sources)
        for mtime in index.file_mtimes.values():
            assert isinstance(mtime, float)

    def test_stores_head_commit(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        assert isinstance(index.head_commit, str)
        assert len(index.head_commit) == 40

    def test_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            RepoIndex.build("/nonexistent/path/that/does/not/exist")


# -- RepoIndex.refresh tests --


class TestRepoIndexRefresh:
    def test_returns_self_when_unchanged(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        refreshed = index.refresh()
        assert refreshed is index

    def test_detects_file_change(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        auth_path = os.path.join(str(tmp_path), "src", "auth.py")
        with open(auth_path, "a") as f:
            f.write("\ndef logout():\n    return True\n")

        # Bump mtime to ensure the change is detected even on fast filesystems
        mtime = os.path.getmtime(auth_path)
        os.utime(auth_path, (mtime + 2, mtime + 2))

        refreshed = index.refresh()
        assert refreshed is not index

        refreshed_contents = {s.path: s.content for s in refreshed.sources}
        assert "logout" in refreshed_contents["src/auth.py"]

    def test_detects_new_file(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        new_path = os.path.join(str(tmp_path), "src", "new_module.py")
        with open(new_path, "w") as f:
            f.write("def new_feature():\n    pass\n")

        refreshed = index.refresh()
        assert refreshed is not index
        refreshed_paths = {s.path for s in refreshed.sources}
        assert "src/new_module.py" in refreshed_paths

    def test_detects_deleted_file(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        utils_path = os.path.join(str(tmp_path), "src", "utils.py")
        os.remove(utils_path)

        refreshed = index.refresh()
        assert refreshed is not index
        refreshed_paths = {s.path for s in refreshed.sources}
        assert "src/utils.py" not in refreshed_paths


# -- select_context tests --


class TestSelectContext:
    def test_with_string_task(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        result = select_context(index, "fix the login bug")
        selected_paths = {ss.source.path for ss in result.selected}
        assert "src/auth.py" in selected_paths

    def test_with_task_object(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        task = Task(
            title="fix the login bug",
            body="",
            symbols=frozenset(["login"]),
        )
        result = select_context(index, task)
        selected_paths = {ss.source.path for ss in result.selected}
        assert "src/auth.py" in selected_paths

    def test_respects_budget(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        result = select_context(index, "fix the login bug", budget=50)
        assert result.total_tokens <= 50

    def test_empty_sources(self, tmp_path):
        # Set up a repo that has no source files (only a markdown README),
        # so the indexer finds nothing and select_context returns an empty result.
        repo_dir = str(tmp_path)
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "T"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        readme = os.path.join(repo_dir, "README.md")
        with open(readme, "w") as f:
            f.write("# Hello\n")
        subprocess.run(
            ["git", "add", "."], cwd=repo_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        index = RepoIndex.build(repo_dir)
        result = select_context(index, "anything")
        assert result.selected == []
        assert result.total_tokens == 0

    def test_orders_results(self, tmp_path):
        _create_fake_repo(str(tmp_path))
        index = RepoIndex.build(str(tmp_path))

        result = select_context(index, "fix the login bug", budget=5000)
        paths = [ss.source.path for ss in result.selected]

        # If both test and non-test files are selected, tests should come after
        test_indices = [i for i, p in enumerate(paths) if "test" in p.lower()]
        non_test_indices = [i for i, p in enumerate(paths) if "test" not in p.lower()]
        if test_indices and non_test_indices:
            assert max(non_test_indices) < min(test_indices)


# -- select_context_from_repo tests --


class TestSelectContextFromRepo:
    def test_convenience_wrapper(self, tmp_path):
        _create_fake_repo(str(tmp_path))

        result = select_context_from_repo(str(tmp_path), "fix the login bug")
        assert result.total_tokens > 0
        selected_paths = {ss.source.path for ss in result.selected}
        assert "src/auth.py" in selected_paths
