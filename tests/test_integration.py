"""End-to-end test: index a fake repo, run all strategies, verify the pipeline works."""

import os
import subprocess
import tempfile

from cognitive_cache.indexer.repo_indexer import index_repo
from cognitive_cache.indexer.git_analyzer import GitAnalyzer
from cognitive_cache.indexer.graph_builder import build_dependency_graph
from cognitive_cache.models import Task
from cognitive_cache.core.value_function import ValueFunction
from cognitive_cache.core.selector import GreedySelector
from cognitive_cache.core.orderer import order_context
from cognitive_cache.signals.embedding_sim import EmbeddingSimilaritySignal
from cognitive_cache.baselines.random_select import RandomStrategy
from cognitive_cache.baselines.full_stuff import FullStuffStrategy
from cognitive_cache.baselines.embedding_select import EmbeddingStrategy
from cognitive_cache.baselines.grep_select import GrepStrategy
from benchmark.evaluator import compute_file_recall
from benchmark.prompt_template import build_prompt


def _create_fake_repo(tmpdir):
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)

    os.makedirs(os.path.join(tmpdir, "src"))
    os.makedirs(os.path.join(tmpdir, "tests"))

    files = {
        "src/auth.py": "from db import get_user\n\ndef login(username, password):\n    user = get_user(username)\n    if user and user.password == password:\n        return True\n    return False\n",
        "src/db.py": "class User:\n    def __init__(self, username, password):\n        self.username = username\n        self.password = password\n\ndef get_user(username):\n    return User(username, 'secret')\n",
        "src/utils.py": "def format_date(d):\n    return d.strftime('%Y-%m-%d')\n\ndef sanitize(text):\n    return text.strip()\n",
        "src/config.py": "DEBUG = True\nSECRET_KEY = 'dev-key'\n",
        "tests/test_auth.py": "from src.auth import login\n\ndef test_login_success():\n    assert login('admin', 'secret') == True\n\ndef test_login_failure():\n    assert login('admin', 'wrong') == False\n",
    }
    for path, content in files.items():
        full = os.path.join(tmpdir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)

    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmpdir, capture_output=True)


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)

        # Index
        sources = index_repo(tmpdir)
        assert len(sources) >= 4

        # Build infrastructure
        git_analyzer = GitAnalyzer(tmpdir)
        recency_data = git_analyzer.recency_scores()
        graph = build_dependency_graph(sources)
        embedding_signal = EmbeddingSimilaritySignal()
        embedding_signal.fit(sources)

        # Define task
        task = Task(
            title="Fix login to handle empty password",
            body="login() crashes when password is None instead of returning False",
            symbols=frozenset(["login", "password"]),
        )
        entry_points = {"src/auth.py"}

        # Run Cognitive Cache
        vf = ValueFunction(
            graph=graph,
            recency_data=recency_data,
            embedding_signal=embedding_signal,
            entry_points=entry_points,
        )
        selector = GreedySelector(value_function=vf)
        result = selector.select(sources, task, budget=500)
        ordered = order_context(result.selected)

        # auth.py should be selected
        selected_paths = {ss.source.path for ss in ordered}
        assert "src/auth.py" in selected_paths

        # File recall
        actual_fix_files = {"src/auth.py"}
        recall = compute_file_recall(selected_paths, actual_fix_files)
        assert recall == 1.0

        # Build prompt
        context = {ss.source.path: ss.source.content for ss in ordered}
        prompt = build_prompt(task.title, task.body, context)
        assert "login" in prompt
        assert "src/auth.py" in prompt

        # Run baselines
        for strategy in [RandomStrategy(seed=42), FullStuffStrategy(), EmbeddingStrategy(), GrepStrategy()]:
            baseline_result = strategy.select(sources, task, budget=500)
            assert baseline_result.total_tokens <= 500
            assert len(baseline_result.selected) > 0


def test_cognitive_cache_beats_random_on_recall():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)

        sources = index_repo(tmpdir)
        task = Task(
            title="Fix login bug",
            body="login function crashes",
            symbols=frozenset(["login"]),
        )
        actual_fix = {"src/auth.py"}

        git_analyzer = GitAnalyzer(tmpdir)
        graph = build_dependency_graph(sources)
        embedding_signal = EmbeddingSimilaritySignal()
        embedding_signal.fit(sources)

        vf = ValueFunction(
            graph=graph,
            recency_data=git_analyzer.recency_scores(),
            embedding_signal=embedding_signal,
            entry_points={"src/auth.py"},
        )
        cc_result = GreedySelector(value_function=vf).select(sources, task, budget=100)
        cc_recall = compute_file_recall(
            {s.source.path for s in cc_result.selected}, actual_fix
        )

        random_recalls = []
        for seed in range(10):
            rand_result = RandomStrategy(seed=seed).select(sources, task, budget=100)
            r = compute_file_recall(
                {s.source.path for s in rand_result.selected}, actual_fix
            )
            random_recalls.append(r)
        avg_random_recall = sum(random_recalls) / len(random_recalls)

        assert cc_recall >= avg_random_recall
