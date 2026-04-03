import os
import subprocess
import tempfile
from unittest.mock import patch

from cognitive_cache.indexer.git_analyzer import GitAnalyzer


def _create_git_repo(tmpdir):
    """Helper: create a git repo with some history."""
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmpdir,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True
    )

    # First commit: create two files
    with open(os.path.join(tmpdir, "old.py"), "w") as f:
        f.write("x = 1\n")
    with open(os.path.join(tmpdir, "new.py"), "w") as f:
        f.write("y = 2\n")
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmpdir, capture_output=True)

    # Second commit: modify only new.py
    with open(os.path.join(tmpdir, "new.py"), "w") as f:
        f.write("y = 2\nz = 3\n")
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "update new.py"], cwd=tmpdir, capture_output=True
    )


def test_recency_scores_recent_file_higher():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_git_repo(tmpdir)
        analyzer = GitAnalyzer(tmpdir)
        scores = analyzer.recency_scores()

        assert "new.py" in scores
        assert "old.py" in scores
        assert scores["new.py"] > scores["old.py"]


def test_recency_scores_range_zero_to_one():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_git_repo(tmpdir)
        analyzer = GitAnalyzer(tmpdir)
        scores = analyzer.recency_scores()

        for path, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{path} score {score} out of range"


def test_change_frequency():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_git_repo(tmpdir)
        analyzer = GitAnalyzer(tmpdir)
        freq = analyzer.change_frequency()

        # new.py appears in 2 commits, old.py in 1
        assert freq["new.py"] > freq["old.py"]


def test_non_git_repo_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = GitAnalyzer(tmpdir)
        scores = analyzer.recency_scores()
        assert scores == {}


def test_recency_scores_uses_since_flag():
    analyzer = GitAnalyzer("/fake/repo")
    with patch.object(analyzer, "_run_git", return_value=None) as mock_run:
        analyzer.recency_scores()
        args = mock_run.call_args[0][0]
        assert any("since" in arg for arg in args), (
            f"Expected --since flag in git args, got: {args}"
        )


def test_change_frequency_uses_since_flag():
    analyzer = GitAnalyzer("/fake/repo")
    with patch.object(analyzer, "_run_git", return_value=None) as mock_run:
        analyzer.change_frequency()
        args = mock_run.call_args[0][0]
        assert any("since" in arg for arg in args), (
            f"Expected --since flag in git args, got: {args}"
        )


def test_is_shallow_returns_false_for_full_clone():
    analyzer = GitAnalyzer("/fake/repo")
    with patch.object(analyzer, "_run_git", return_value="false\n") as mock_run:
        assert analyzer.is_shallow() is False
        mock_run.assert_called_once_with(["rev-parse", "--is-shallow-repository"])


def test_is_shallow_returns_true_for_shallow_clone():
    analyzer = GitAnalyzer("/fake/repo")
    with patch.object(analyzer, "_run_git", return_value="true\n"):
        assert analyzer.is_shallow() is True


def test_is_shallow_returns_false_on_error():
    analyzer = GitAnalyzer("/fake/repo")
    with patch.object(analyzer, "_run_git", return_value=None):
        assert analyzer.is_shallow() is False
