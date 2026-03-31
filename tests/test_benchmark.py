from benchmark.dataset import BenchmarkIssue, load_dataset, save_dataset
from benchmark.evaluator import compute_file_recall
from benchmark.prompt_template import build_prompt


def test_benchmark_issue_structure():
    issue = BenchmarkIssue(
        repo="owner/repo",
        repo_url="https://github.com/owner/repo",
        issue_number=42,
        title="Fix login bug",
        body="Login crashes on empty password",
        fixed_files=["src/auth.py", "tests/test_auth.py"],
        fix_commit="abc123",
        base_commit="def456",
    )
    assert issue.repo == "owner/repo"
    assert len(issue.fixed_files) == 2


def test_file_recall_perfect():
    selected = {"src/auth.py", "tests/test_auth.py"}
    actual = {"src/auth.py", "tests/test_auth.py"}
    assert compute_file_recall(selected, actual) == 1.0


def test_file_recall_partial():
    selected = {"src/auth.py", "src/utils.py"}
    actual = {"src/auth.py", "tests/test_auth.py"}
    assert compute_file_recall(selected, actual) == 0.5


def test_file_recall_none():
    selected = {"src/utils.py"}
    actual = {"src/auth.py"}
    assert compute_file_recall(selected, actual) == 0.0


def test_prompt_template_includes_context():
    context_files = {"src/auth.py": "def login(): pass"}
    prompt = build_prompt(
        title="Fix login",
        body="It crashes",
        context_files=context_files,
    )
    assert "Fix login" in prompt
    assert "def login(): pass" in prompt
    assert "src/auth.py" in prompt


def test_dataset_roundtrip(tmp_path):
    issues = [
        BenchmarkIssue(
            repo="owner/repo",
            repo_url="https://github.com/owner/repo",
            issue_number=1,
            title="Bug",
            body="Details",
            fixed_files=["a.py"],
            fix_commit="abc",
            base_commit="def",
        )
    ]
    path = str(tmp_path / "dataset.json")
    save_dataset(issues, path)
    loaded = load_dataset(path)
    assert len(loaded) == 1
    assert loaded[0].title == "Bug"
