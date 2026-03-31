from cognitive_cache.models import Source, Task, SelectionResult, ScoredSource


def test_source_creation():
    source = Source(
        path="src/auth.py",
        content="def login(user, password): pass",
        token_count=10,
        language="python",
        symbols=frozenset({"login", "user", "password"}),
    )
    assert source.path == "src/auth.py"
    assert source.token_count == 10
    assert "login" in source.symbols


def test_task_creation():
    task = Task(
        title="Fix login bug",
        body="The login function crashes when password is empty",
        symbols=frozenset({"login", "password"}),
    )
    assert "login" in task.symbols


def test_scored_source():
    source = Source("a.py", "x = 1", 5, "python", frozenset({"x"}))
    scored = ScoredSource(
        source=source,
        score=0.85,
        signal_scores={"symbol_overlap": 0.9, "graph_distance": 0.8},
    )
    assert scored.score == 0.85
    assert scored.signal_scores["symbol_overlap"] == 0.9


def test_selection_result():
    source = Source("a.py", "x = 1", 5, "python", frozenset({"x"}))
    scored = ScoredSource(source, 0.85, {})
    result = SelectionResult(
        selected=[scored],
        total_tokens=5,
        budget=100,
    )
    assert result.budget_remaining == 95
