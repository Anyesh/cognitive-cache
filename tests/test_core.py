from cognitive_cache.models import Source, Task, ScoredSource
from cognitive_cache.core.value_function import ValueFunction
from cognitive_cache.core.selector import GreedySelector
from cognitive_cache.core.orderer import order_context
from cognitive_cache.core.chunker import chunk_source


def _make_source(path, content="x = 1", tokens=10, symbols=None, is_test=False):
    return Source(
        path=path,
        content=content,
        token_count=tokens,
        language="python",
        symbols=frozenset(symbols or []),
        is_test=is_test,
    )


def _make_task(symbols=None):
    return Task(
        title="Fix bug", body="login fails", symbols=frozenset(symbols or ["login"])
    )


# --- Value Function ---


def test_value_function_scores_relevant_higher():
    relevant = _make_source("auth.py", symbols=["login", "authenticate"])
    irrelevant = _make_source("plotting.py", symbols=["plot", "chart"])
    task = _make_task(symbols=["login"])

    vf = ValueFunction()
    score_relevant = vf.score(relevant, task, [])
    score_irrelevant = vf.score(irrelevant, task, [])

    assert score_relevant > score_irrelevant


def test_value_function_penalizes_redundancy():
    source1 = _make_source("auth.py", symbols=["login"])
    source2 = _make_source("auth2.py", symbols=["login"])
    task = _make_task(symbols=["login"])

    vf = ValueFunction()
    score_first = vf.score(source1, task, [])
    score_redundant = vf.score(source2, task, [source1])

    assert score_first > score_redundant


def test_value_function_returns_signal_breakdown():
    source = _make_source("auth.py", symbols=["login"])
    task = _make_task(symbols=["login"])

    vf = ValueFunction()
    score, breakdown = vf.score_with_breakdown(source, task, [])

    assert isinstance(breakdown, dict)
    assert "symbol_overlap" in breakdown
    assert score > 0


# --- Selector ---


def test_selector_respects_budget():
    sources = [
        _make_source("a.py", tokens=50, symbols=["login"]),
        _make_source("b.py", tokens=50, symbols=["login"]),
        _make_source("c.py", tokens=50, symbols=["login"]),
    ]
    task = _make_task(symbols=["login"])

    selector = GreedySelector(value_function=ValueFunction())
    result = selector.select(sources, task, budget=100)

    assert result.total_tokens <= 100
    assert len(result.selected) == 2


def test_selector_picks_highest_value():
    high_value = _make_source("auth.py", tokens=10, symbols=["login", "validate"])
    low_value = _make_source("readme.py", tokens=10, symbols=[])
    task = _make_task(symbols=["login", "validate"])

    selector = GreedySelector(value_function=ValueFunction())
    result = selector.select([low_value, high_value], task, budget=15)

    # auth.py should be picked first (highest score)
    assert result.selected[0].source.path == "auth.py"


def test_selector_prefers_value_per_token():
    small = _make_source("types.py", tokens=10, symbols=["login"])
    large = _make_source("big.py", tokens=100, symbols=["login"])
    task = _make_task(symbols=["login"])

    selector = GreedySelector(value_function=ValueFunction())
    result = selector.select([large, small], task, budget=20)

    assert result.selected[0].source.path == "types.py"


def test_selector_stops_at_threshold():
    sources = [
        _make_source("a.py", tokens=10, symbols=["login"]),
        _make_source("b.py", tokens=10, symbols=[]),
        _make_source("c.py", tokens=10, symbols=[]),
    ]
    task = _make_task(symbols=["login"])

    selector = GreedySelector(
        value_function=ValueFunction(), score_threshold=0.3, vpt_threshold=0.01
    )
    result = selector.select(sources, task, budget=1000)

    # b.py and c.py have no symbol overlap — below both thresholds
    assert len(result.selected) < len(sources)


# --- Orderer ---


def test_orderer_puts_highest_score_first():
    scored = [
        ScoredSource(_make_source("low.py"), score=0.3, signal_scores={}),
        ScoredSource(_make_source("high.py"), score=0.9, signal_scores={}),
        ScoredSource(_make_source("mid.py"), score=0.6, signal_scores={}),
    ]
    ordered = order_context(scored)
    assert ordered[0].source.path == "high.py"


def test_orderer_puts_tests_last():
    scored = [
        ScoredSource(
            _make_source("tests/test_auth.py", is_test=True),
            score=0.9,
            signal_scores={},
        ),
        ScoredSource(_make_source("auth.py"), score=0.9, signal_scores={}),
    ]
    ordered = order_context(scored)
    assert ordered[-1].source.path == "tests/test_auth.py"


def test_selector_excludes_test_files():
    sources = [
        _make_source("auth.py", tokens=10, symbols=["login"]),
        _make_source("tests/test_auth.py", tokens=10, symbols=["login"], is_test=True),
    ]
    task = _make_task(symbols=["login"])

    selector = GreedySelector(value_function=ValueFunction(), include_tests=False)
    result = selector.select(sources, task, budget=1000)

    paths = [ss.source.path for ss in result.selected]
    assert "auth.py" in paths
    assert "tests/test_auth.py" not in paths


def test_selector_includes_test_files_by_default():
    sources = [
        _make_source("auth.py", tokens=10, symbols=["login"]),
        _make_source("tests/test_auth.py", tokens=10, symbols=["login"], is_test=True),
    ]
    task = _make_task(symbols=["login"])

    selector = GreedySelector(value_function=ValueFunction())
    result = selector.select(sources, task, budget=1000)

    paths = [ss.source.path for ss in result.selected]
    assert "tests/test_auth.py" in paths


def test_selector_respects_min_score():
    sources = [
        _make_source("auth.py", tokens=10, symbols=["login"]),
        _make_source("unrelated.py", tokens=10, symbols=[]),
    ]
    task = _make_task(symbols=["login"])

    selector = GreedySelector(value_function=ValueFunction(), min_score=0.3)
    result = selector.select(sources, task, budget=1000)

    paths = [ss.source.path for ss in result.selected]
    assert "auth.py" in paths
    assert "unrelated.py" not in paths


def test_selector_respects_max_files():
    sources = [
        _make_source(f"file{i}.py", tokens=10, symbols=["login"]) for i in range(10)
    ]
    task = _make_task(symbols=["login"])

    selector = GreedySelector(value_function=ValueFunction(), max_files=3)
    result = selector.select(sources, task, budget=10000)

    assert len(result.selected) == 3


def test_chunker_preserves_is_test():
    large_content = "def test_login():\n" + "    assert True\n" * 500
    source = _make_source(
        "tests/test_auth.py", content=large_content, tokens=1000, is_test=True
    )
    task = _make_task(symbols=["login"])
    chunked = chunk_source(source, task, max_tokens=100)
    assert chunked.is_test is True
    assert chunked.token_count < source.token_count
