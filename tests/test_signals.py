from cognitive_cache.models import Source, Task
from cognitive_cache.signals.symbol_overlap import SymbolOverlapSignal
from cognitive_cache.signals.graph_distance import GraphDistanceSignal
from cognitive_cache.signals.change_recency import ChangeRecencySignal
from cognitive_cache.signals.redundancy import RedundancySignal
from cognitive_cache.signals.embedding_sim import EmbeddingSimilaritySignal
from cognitive_cache.signals.file_role_prior import FileRolePriorSignal
from cognitive_cache.indexer.graph_builder import DependencyGraph


def _make_source(path="a.py", content="x = 1", symbols=None):
    return Source(
        path=path,
        content=content,
        token_count=10,
        language="python",
        symbols=frozenset(symbols or []),
    )


def _make_task(title="Fix bug", body="The login function fails", symbols=None):
    return Task(title=title, body=body, symbols=frozenset(symbols or []))


# --- Symbol Overlap ---


def test_symbol_overlap_exact_match_scores_high():
    source = _make_source(symbols=["login", "authenticate"])
    task = _make_task(title="fix the login function", body="", symbols=["login"])
    signal = SymbolOverlapSignal()
    score = signal.score(source, task, [])
    assert score >= 0.5


def test_symbol_overlap_contains_match_scores_lower():
    source = _make_source(symbols=["test_login_success", "test_login_failure"])
    task = _make_task(title="fix the login function", body="", symbols=["login"])
    signal = SymbolOverlapSignal()
    score = signal.score(source, task, [])
    assert 0.0 < score < 0.5


def test_symbol_overlap_exact_beats_contains():
    exact_source = _make_source(path="auth.py", symbols=["login"])
    contains_source = _make_source(
        path="test_auth.py", symbols=["test_login_success", "test_login_failure"]
    )
    task = _make_task(title="fix the login function", body="", symbols=["login"])
    signal = SymbolOverlapSignal()
    exact_score = signal.score(exact_source, task, [])
    contains_score = signal.score(contains_source, task, [])
    assert exact_score > contains_score


def test_symbol_overlap_no_match():
    source = _make_source(symbols=["database", "query"])
    task = _make_task(title="fix the login function", body="", symbols=["login"])
    signal = SymbolOverlapSignal()
    score = signal.score(source, task, [])
    assert score == 0.0


def test_symbol_overlap_content_match():
    source = _make_source(
        symbols=["other_func"],
        content="# This module handles login for the app",
    )
    task = _make_task(title="fix the login function", body="", symbols=["login"])
    signal = SymbolOverlapSignal()
    score = signal.score(source, task, [])
    assert 0.0 < score <= 0.15


def test_symbol_overlap_fallback_to_task_symbols():
    source = _make_source(symbols=["login"])
    task = _make_task(title="fix", body="", symbols=["login"])
    signal = SymbolOverlapSignal()
    score = signal.score(source, task, [])
    assert score > 0.0


def test_symbol_overlap_empty_task():
    source = _make_source(symbols=["login"])
    task = _make_task(title="", body="", symbols=[])
    signal = SymbolOverlapSignal()
    score = signal.score(source, task, [])
    assert score == 0.0


# --- Graph Distance ---


def test_graph_distance_direct_neighbor():
    graph = DependencyGraph()
    graph.add_file("a.py")
    graph.add_file("b.py")
    graph.add_edge("a.py", "b.py")

    signal = GraphDistanceSignal(graph)
    source = _make_source(path="b.py")
    task = _make_task(symbols=["something"])
    score = signal.score(source, task, [], entry_points={"a.py"})
    assert score >= 0.5  # distance 1 -> 0.5^1 = 0.5


def test_graph_distance_no_connection():
    graph = DependencyGraph()
    graph.add_file("a.py")
    graph.add_file("z.py")

    signal = GraphDistanceSignal(graph)
    source = _make_source(path="z.py")
    task = _make_task()
    score = signal.score(source, task, [], entry_points={"a.py"})
    assert score == 0.0


# --- Change Recency ---


def test_change_recency_with_scores():
    recency_data = {"recent.py": 1.0, "old.py": 0.2}
    signal = ChangeRecencySignal(recency_data)

    recent = _make_source(path="recent.py")
    old = _make_source(path="old.py")
    task = _make_task()

    assert signal.score(recent, task, []) > signal.score(old, task, [])


def test_change_recency_unknown_file():
    signal = ChangeRecencySignal({})
    source = _make_source(path="unknown.py")
    assert signal.score(source, _make_task(), []) == 0.0


def test_change_recency_shallow_fallback():
    recency_data = {"recent.py": 1.0}
    signal = ChangeRecencySignal(recency_data, is_shallow=True)
    source = _make_source(path="stable_core.py")
    score = signal.score(source, _make_task(), [])
    assert score == 0.3


def test_change_recency_not_shallow_uses_zero():
    recency_data = {"recent.py": 1.0}
    signal = ChangeRecencySignal(recency_data, is_shallow=False)
    source = _make_source(path="unknown.py")
    score = signal.score(source, _make_task(), [])
    assert score == 0.0


# --- Redundancy ---


def test_redundancy_penalizes_similar():
    source = _make_source(path="auth2.py", symbols=["login", "validate"])
    already_selected = [_make_source(path="auth.py", symbols=["login", "validate"])]
    signal = RedundancySignal()
    score = signal.score(source, _make_task(), already_selected)
    assert score > 0.5


def test_redundancy_no_penalty_for_unique():
    source = _make_source(path="db.py", symbols=["query", "connect"])
    already_selected = [_make_source(path="auth.py", symbols=["login"])]
    signal = RedundancySignal()
    score = signal.score(source, _make_task(), already_selected)
    assert score < 0.2


def test_redundancy_empty_selection():
    signal = RedundancySignal()
    score = signal.score(_make_source(), _make_task(), [])
    assert score == 0.0


# --- Embedding Similarity ---


def test_embedding_similarity_related_content():
    signal = EmbeddingSimilaritySignal()
    source = _make_source(content="def login(user, password): authenticate(user)")
    task = _make_task(body="The login authentication is broken")
    score = signal.score(source, task, [])
    assert score > 0.0


def test_embedding_similarity_unrelated():
    signal = EmbeddingSimilaritySignal()
    source = _make_source(content="import matplotlib\nplt.plot(x, y)")
    task = _make_task(body="The login authentication is broken")
    related_source = _make_source(content="def login(user, password): pass")

    score_unrelated = signal.score(source, task, [])
    score_related = signal.score(related_source, task, [])
    assert score_related > score_unrelated


# --- File Role Prior ---


def test_file_role_test_file():
    signal = FileRolePriorSignal()
    source = _make_source(path="tests/test_auth.py")
    score = signal.score(source, _make_task(), [])
    assert score > 0.0


def test_file_role_config_file():
    signal = FileRolePriorSignal()
    source = _make_source(path="config.py")
    score = signal.score(source, _make_task(), [])
    assert score > 0.0


def test_file_role_regular_file():
    signal = FileRolePriorSignal()
    test_source = _make_source(path="tests/test_auth.py")
    regular_source = _make_source(path="src/utils.py")
    task = _make_task()
    assert signal.score(test_source, task, []) != signal.score(regular_source, task, [])


def test_file_role_source_beats_test():
    signal = FileRolePriorSignal()
    test_source = _make_source(path="tests/test_auth.py")
    regular_source = _make_source(path="src/auth.py")
    task = _make_task()
    assert signal.score(regular_source, task, []) > signal.score(test_source, task, [])
