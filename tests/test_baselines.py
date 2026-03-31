from cognitive_cache.models import Source, Task
from cognitive_cache.baselines.random_select import RandomStrategy
from cognitive_cache.baselines.full_stuff import FullStuffStrategy
from cognitive_cache.baselines.embedding_select import EmbeddingStrategy
from cognitive_cache.baselines.grep_select import GrepStrategy


def _make_sources(n=10):
    return [
        Source(
            path=f"file{i}.py",
            content=f"def func{i}(): pass\n" * 5,
            token_count=20,
            language="python",
            symbols=frozenset([f"func{i}"]),
        )
        for i in range(n)
    ]


def _make_task():
    return Task(title="Fix func3", body="func3 crashes", symbols=frozenset(["func3"]))


def test_random_strategy_respects_budget():
    sources = _make_sources(10)
    strategy = RandomStrategy(seed=42)
    result = strategy.select(sources, _make_task(), budget=50)
    assert result.total_tokens <= 50


def test_random_strategy_is_deterministic_with_seed():
    sources = _make_sources(10)
    result1 = RandomStrategy(seed=42).select(sources, _make_task(), budget=100)
    result2 = RandomStrategy(seed=42).select(sources, _make_task(), budget=100)
    paths1 = [s.source.path for s in result1.selected]
    paths2 = [s.source.path for s in result2.selected]
    assert paths1 == paths2


def test_full_stuff_fills_budget():
    sources = _make_sources(10)
    result = FullStuffStrategy().select(sources, _make_task(), budget=100)
    assert result.total_tokens <= 100
    assert result.total_tokens >= 80


def test_embedding_strategy_prefers_relevant():
    sources = [
        Source("auth.py", "def func3(): login()", 20, "python", frozenset(["func3"])),
        Source("plot.py", "import matplotlib; plt.show()", 20, "python", frozenset(["plot"])),
    ]
    result = EmbeddingStrategy().select(sources, _make_task(), budget=25)
    assert result.selected[0].source.path == "auth.py"


def test_grep_strategy_finds_symbol_matches():
    sources = _make_sources(10)
    result = GrepStrategy().select(sources, _make_task(), budget=100)
    paths = {s.source.path for s in result.selected}
    assert "file3.py" in paths


def test_all_strategies_return_selection_result():
    sources = _make_sources(5)
    task = _make_task()
    strategies = [
        RandomStrategy(seed=42),
        FullStuffStrategy(),
        EmbeddingStrategy(),
        GrepStrategy(),
    ]
    for strategy in strategies:
        result = strategy.select(sources, task, budget=60)
        assert result.total_tokens <= 60
        assert result.budget == 60
