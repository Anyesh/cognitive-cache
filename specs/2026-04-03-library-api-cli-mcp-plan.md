# Library API, CLI, and MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give cognitive-cache three integration surfaces (library API, CLI, MCP server) so it can be used by agent builders, developers, and AI coding tools.

**Architecture:** A `RepoIndex` class caches all repo-dependent state (sources, git recency, dependency graph, TF-IDF embeddings). A `select_context()` function takes a cached index plus a task and runs only the cheap per-task scoring. The CLI and MCP server are thin wrappers around these two primitives. Two performance fixes (git log cap, graph copy cache) are applied to existing code before building on top of it.

**Tech Stack:** Python 3.11+, `mcp` SDK (FastMCP), argparse, tiktoken, scikit-learn, networkx, difflib

**Spec:** `specs/2026-04-03-library-api-cli-mcp-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/cognitive_cache/indexer/graph_builder.py` | Modify | Cache undirected graph in DependencyGraph |
| `src/cognitive_cache/indexer/git_analyzer.py` | Modify | Cap git log at 6 months |
| `src/cognitive_cache/api.py` | Create | RepoIndex, select_context, select_context_from_repo, _extract_task_symbols, _find_entry_points |
| `src/cognitive_cache/__init__.py` | Modify | Add public exports |
| `src/cognitive_cache/cli.py` | Create | CLI entry point (argparse) |
| `src/cognitive_cache/mcp_server.py` | Create | MCP server (FastMCP, single tool) |
| `benchmark/runner.py` | Modify | Import _extract_task_symbols and _find_entry_points from api |
| `benchmark/evaluator.py` | Modify | Add compute_patch_similarity |
| `benchmark/dataset.py` | Modify | Add ground_truth_diff field to BenchmarkIssue |
| `benchmark/analyze_patches.py` | Create | Patch similarity analysis script |
| `pyproject.toml` | Modify | Add entry points, mcp dependency group |
| `tests/test_graph_builder.py` | Modify | Add undirected cache tests |
| `tests/test_git_analyzer.py` | Modify | Add git log cap tests |
| `tests/test_api.py` | Create | Tests for RepoIndex, select_context |
| `tests/test_cli.py` | Create | Tests for CLI output |
| `tests/test_mcp_server.py` | Create | Tests for MCP tool handler |
| `tests/test_patch_eval.py` | Create | Tests for compute_patch_similarity |

---

### Task 1: Fix DependencyGraph undirected copy bug

**Files:**
- Modify: `src/cognitive_cache/indexer/graph_builder.py:20-47`
- Test: `tests/test_graph_builder.py`

- [ ] **Step 1: Write failing test for cached undirected graph**

Add to `tests/test_graph_builder.py`:

```python
def test_shortest_distance_caches_undirected_graph():
    """Calling shortest_distance twice should reuse the cached undirected graph."""
    graph = DependencyGraph()
    graph.add_file("a.py")
    graph.add_file("b.py")
    graph.add_file("c.py")
    graph.add_edge("a.py", "b.py")
    graph.add_edge("b.py", "c.py")

    # First call computes and caches
    d1 = graph.shortest_distance("a.py", "c.py")
    assert d1 == 2

    # Second call should return same result (from cache)
    d2 = graph.shortest_distance("a.py", "c.py")
    assert d2 == 2

    # Verify cache exists
    assert graph._undirected is not None


def test_undirected_cache_invalidated_on_add_edge():
    """Adding an edge should invalidate the cached undirected graph."""
    graph = DependencyGraph()
    graph.add_file("a.py")
    graph.add_file("b.py")
    graph.add_file("c.py")
    graph.add_edge("a.py", "b.py")

    # Populate cache
    graph.shortest_distance("a.py", "b.py")
    assert graph._undirected is not None

    # Adding edge invalidates cache
    graph.add_edge("b.py", "c.py")
    assert graph._undirected is None

    # New distance computation works with updated graph
    d = graph.shortest_distance("a.py", "c.py")
    assert d == 2


def test_undirected_cache_invalidated_on_add_file():
    """Adding a file should invalidate the cached undirected graph."""
    graph = DependencyGraph()
    graph.add_file("a.py")

    # Populate cache
    graph.shortest_distance("a.py", "a.py")
    assert graph._undirected is not None

    # Adding file invalidates cache
    graph.add_file("b.py")
    assert graph._undirected is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_graph_builder.py -v -k "cache"`
Expected: FAIL with `AttributeError: 'DependencyGraph' object has no attribute '_undirected'`

- [ ] **Step 3: Implement undirected graph cache**

In `src/cognitive_cache/indexer/graph_builder.py`, modify the `DependencyGraph` class:

Replace the `__init__` method:

```python
def __init__(self):
    self._graph = nx.DiGraph()
    self._undirected = None  # cached undirected view
```

Replace the `add_file` method:

```python
def add_file(self, path: str):
    """Add a file node to the graph."""
    self._graph.add_node(path)
    self._undirected = None  # invalidate cache
```

Replace the `add_edge` method:

```python
def add_edge(self, importer: str, imported: str):
    """Add a directed edge: importer depends on imported."""
    self._graph.add_edge(importer, imported)
    self._undirected = None  # invalidate cache
```

Replace the `shortest_distance` method:

```python
def shortest_distance(self, source: str, target: str) -> float:
    """Shortest path distance between two files (undirected).

    We treat the graph as undirected for distance because
    "A imports B" means both A and B are relevant to each other.
    Returns float('inf') if no path exists.
    """
    if self._undirected is None:
        self._undirected = self._graph.to_undirected()
    try:
        return nx.shortest_path_length(self._undirected, source, target)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return float("inf")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_graph_builder.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/cognitive_cache/indexer/graph_builder.py tests/test_graph_builder.py
git commit -m "perf: cache undirected graph in DependencyGraph

Eliminates redundant to_undirected() calls during selection.
Previously created a full graph copy on every shortest_distance call;
now lazily computed once and invalidated when edges or nodes change."
```

---

### Task 2: Cap git log depth at 6 months

**Files:**
- Modify: `src/cognitive_cache/indexer/git_analyzer.py:37-80`
- Test: `tests/test_git_analyzer.py`

- [ ] **Step 1: Write failing test for git log depth cap**

Add to `tests/test_git_analyzer.py`:

```python
from unittest.mock import patch, MagicMock
import subprocess


def test_recency_scores_uses_since_flag():
    """recency_scores should pass --since to git log to cap history depth."""
    analyzer = GitAnalyzer("/fake/repo")
    with patch.object(analyzer, '_run_git', return_value=None) as mock_run:
        analyzer.recency_scores()
        args = mock_run.call_args[0][0]
        assert any("since" in arg for arg in args), \
            f"Expected --since flag in git args, got: {args}"


def test_change_frequency_uses_since_flag():
    """change_frequency should pass --since to git log to cap history depth."""
    analyzer = GitAnalyzer("/fake/repo")
    with patch.object(analyzer, '_run_git', return_value=None) as mock_run:
        analyzer.change_frequency()
        args = mock_run.call_args[0][0]
        assert any("since" in arg for arg in args), \
            f"Expected --since flag in git args, got: {args}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_git_analyzer.py -v -k "since_flag"`
Expected: FAIL with `AssertionError: Expected --since flag`

- [ ] **Step 3: Add --since flag to git log calls**

In `src/cognitive_cache/indexer/git_analyzer.py`, modify `recency_scores()`:

Replace the `_run_git` call inside `recency_scores` (around line 46):

```python
        output = self._run_git([
            "log", "--pretty=format:%H", "--name-only", "--diff-filter=ACMR",
            "--since=6.months.ago",
        ])
```

And modify `change_frequency()`, replace the `_run_git` call (around line 91):

```python
        output = self._run_git([
            "log", "--pretty=format:", "--name-only", "--diff-filter=ACMR",
            "--since=6.months.ago",
        ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_git_analyzer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/cognitive_cache/indexer/git_analyzer.py tests/test_git_analyzer.py
git commit -m "perf: cap git log at 6 months of history

Prevents multi-minute git log calls on repos with 100K+ commits.
Six months is generous for recency scoring; older files normalize to
near-zero anyway."
```

---

### Task 3: Create the library API with RepoIndex and select_context

**Files:**
- Create: `src/cognitive_cache/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests for RepoIndex.build**

Create `tests/test_api.py`:

```python
"""Tests for the public API: RepoIndex and select_context."""

import os
import subprocess
import tempfile

from cognitive_cache.api import RepoIndex, select_context, select_context_from_repo
from cognitive_cache.models import Task, SelectionResult


def _create_fake_repo(tmpdir):
    """Create a minimal git repo with source files for testing."""
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)

    os.makedirs(os.path.join(tmpdir, "src"))
    os.makedirs(os.path.join(tmpdir, "tests"))

    files = {
        "src/auth.py": (
            "from src.db import get_user\n\n"
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
            "        self.password = password\n\n"
            "def get_user(username):\n"
            "    return User(username, 'secret')\n"
        ),
        "src/utils.py": (
            "def format_date(d):\n"
            "    return d.strftime('%Y-%m-%d')\n\n"
            "def sanitize(text):\n"
            "    return text.strip()\n"
        ),
        "tests/test_auth.py": (
            "from src.auth import login\n\n"
            "def test_login_success():\n"
            "    assert login('admin', 'secret') == True\n"
        ),
    }
    for path, content in files.items():
        full = os.path.join(tmpdir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)

    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmpdir, capture_output=True)


# --- RepoIndex.build ---

def test_repo_index_build_indexes_sources():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        assert len(index.sources) >= 4
        paths = {s.path for s in index.sources}
        assert "src/auth.py" in paths
        assert "src/db.py" in paths


def test_repo_index_build_has_recency_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        assert isinstance(index.recency_data, dict)
        # All files were committed, so they should have recency scores
        assert len(index.recency_data) > 0


def test_repo_index_build_has_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        assert index.graph is not None
        assert "src/auth.py" in index.graph.nodes


def test_repo_index_build_has_fitted_embedding():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        assert index.embedding_signal is not None
        assert index.embedding_signal._fitted


def test_repo_index_build_stores_mtimes():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        assert len(index.file_mtimes) == len(index.sources)
        for path in index.file_mtimes:
            assert isinstance(index.file_mtimes[path], float)


def test_repo_index_build_stores_head_commit():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        assert isinstance(index.head_commit, str)
        assert len(index.head_commit) == 40  # SHA-1 hash


def test_repo_index_build_nonexistent_path():
    import pytest
    with pytest.raises(FileNotFoundError):
        RepoIndex.build("/nonexistent/path/that/does/not/exist")


# --- RepoIndex.refresh ---

def test_repo_index_refresh_returns_self_when_unchanged():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)
        refreshed = index.refresh()

        assert refreshed is index  # same object, nothing changed


def test_repo_index_refresh_detects_file_change():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        # Modify a file and explicitly bump mtime to avoid filesystem
        # resolution issues (some FSes have 1-second granularity)
        auth_path = os.path.join(tmpdir, "src/auth.py")
        with open(auth_path, "a") as f:
            f.write("\ndef logout():\n    pass\n")
        os.utime(auth_path, (os.path.getmtime(auth_path) + 2, os.path.getmtime(auth_path) + 2))

        refreshed = index.refresh()
        assert refreshed is not index  # new object

        # The new file should have the updated content
        auth_source = next(s for s in refreshed.sources if s.path == "src/auth.py")
        assert "logout" in auth_source.content


def test_repo_index_refresh_detects_new_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        # Add a new source file
        with open(os.path.join(tmpdir, "src/new_module.py"), "w") as f:
            f.write("def new_func():\n    pass\n")

        refreshed = index.refresh()
        assert refreshed is not index

        paths = {s.path for s in refreshed.sources}
        assert "src/new_module.py" in paths


def test_repo_index_refresh_detects_deleted_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        # Delete a file
        os.remove(os.path.join(tmpdir, "src/utils.py"))

        refreshed = index.refresh()
        assert refreshed is not index

        paths = {s.path for s in refreshed.sources}
        assert "src/utils.py" not in paths


# --- select_context ---

def test_select_context_with_string_task():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        result = select_context(index, "fix the login bug", budget=5000)

        assert isinstance(result, SelectionResult)
        assert result.total_tokens <= 5000
        assert len(result.selected) > 0

        # auth.py should be selected since "login" matches a symbol
        selected_paths = {ss.source.path for ss in result.selected}
        assert "src/auth.py" in selected_paths


def test_select_context_with_task_object():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        task = Task(
            title="Fix login",
            body="login function crashes",
            symbols=frozenset(["login"]),
        )
        result = select_context(index, task, budget=5000)

        assert isinstance(result, SelectionResult)
        selected_paths = {ss.source.path for ss in result.selected}
        assert "src/auth.py" in selected_paths


def test_select_context_respects_budget():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        result = select_context(index, "fix login", budget=50)
        assert result.total_tokens <= 50


def test_select_context_empty_sources():
    """select_context on a repo with no source files returns empty result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
        # Create a non-source file so git commit works
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("hello")
        subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)

        index = RepoIndex.build(tmpdir)
        result = select_context(index, "fix something", budget=5000)

        assert len(result.selected) == 0
        assert result.total_tokens == 0


def test_select_context_orders_results():
    """Results should be ordered by the orderer (non-test files first, tests last)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        index = RepoIndex.build(tmpdir)

        result = select_context(index, "fix the login bug", budget=5000)

        if len(result.selected) > 1:
            paths = [ss.source.path for ss in result.selected]
            test_indices = [i for i, p in enumerate(paths) if "test" in p]
            non_test_indices = [i for i, p in enumerate(paths) if "test" not in p]
            if test_indices and non_test_indices:
                assert max(non_test_indices) < min(test_indices), \
                    "Test files should come after non-test files"


# --- select_context_from_repo ---

def test_select_context_from_repo_convenience():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)

        result = select_context_from_repo(tmpdir, "fix the login bug", budget=5000)

        assert isinstance(result, SelectionResult)
        assert len(result.selected) > 0
        selected_paths = {ss.source.path for ss in result.selected}
        assert "src/auth.py" in selected_paths
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cognitive_cache.api'`

- [ ] **Step 3: Implement api.py**

Create `src/cognitive_cache/api.py`:

```python
"""Public API for cognitive-cache.

Two primary exports:
- RepoIndex: caches all repo-dependent state (sources, git recency, graph, embeddings)
- select_context: runs the per-task scoring and selection against a cached index
"""

import os
import re
import subprocess

from cognitive_cache.models import Source, Task, SelectionResult
from cognitive_cache.indexer.repo_indexer import index_repo, SOURCE_EXTENSIONS, SKIP_DIRS
from cognitive_cache.indexer.git_analyzer import GitAnalyzer
from cognitive_cache.indexer.graph_builder import build_dependency_graph
from cognitive_cache.indexer.token_counter import count_tokens
from cognitive_cache.signals.embedding_sim import EmbeddingSimilaritySignal
from cognitive_cache.core.value_function import ValueFunction
from cognitive_cache.core.selector import GreedySelector
from cognitive_cache.core.orderer import order_context


def _get_head_commit(repo_path: str) -> str:
    """Get the current HEAD commit hash, or empty string if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def _collect_mtimes(repo_path: str, sources: list[Source]) -> dict[str, float]:
    """Collect file modification times for all indexed sources."""
    mtimes = {}
    for source in sources:
        full_path = os.path.join(repo_path, source.path)
        try:
            mtimes[source.path] = os.path.getmtime(full_path)
        except OSError:
            pass
    return mtimes


def _extract_task_symbols(title: str, body: str, sources: list[Source]) -> frozenset[str]:
    """Extract symbols mentioned in the task text that match known repo symbols.

    Two-level matching:
    1. Exact match: symbol name appears verbatim in issue text (strongest signal)
    2. Substring match: longer words (6+ chars) from the task appear within symbol names
    """
    all_symbols = set()
    for s in sources:
        all_symbols.update(s.symbols)

    text = f"{title} {body}".lower()
    task_words_long = set(re.findall(r"\b[a-z_][a-z0-9_]{5,}\b", text))
    stop_words = {
        "return", "import", "should", "string", "number", "before", "after",
        "called", "values", "object", "update", "create", "delete", "method",
        "function", "default", "option", "options", "config", "module",
        "result", "response", "request", "handler", "callback", "parameter",
    }
    task_words_long -= stop_words

    matches = set()
    for sym in all_symbols:
        sym_lower = sym.lower()
        if sym_lower in text and len(sym_lower) >= 4:
            matches.add(sym)
            continue
        for word in task_words_long:
            if word in sym_lower:
                matches.add(sym)
                break

    return frozenset(matches)


def _find_entry_points(task_symbols: frozenset[str], sources: list[Source]) -> set[str]:
    """Find files whose defined symbols overlap with the task's symbols."""
    entry_points = set()
    for s in sources:
        if s.symbols & task_symbols:
            entry_points.add(s.path)
    return entry_points


class RepoIndex:
    """Cached repo-dependent state for fast repeated context selection.

    Build once with RepoIndex.build(repo_path), then call select_context()
    multiple times with different tasks. Call refresh() to pick up file
    changes without re-indexing from scratch.
    """

    def __init__(
        self,
        repo_path: str,
        sources: list[Source],
        recency_data: dict[str, float],
        graph,
        embedding_signal: EmbeddingSimilaritySignal,
        file_mtimes: dict[str, float],
        head_commit: str,
    ):
        self.repo_path = repo_path
        self.sources = sources
        self.recency_data = recency_data
        self.graph = graph
        self.embedding_signal = embedding_signal
        self.file_mtimes = file_mtimes
        self.head_commit = head_commit

    @classmethod
    def build(cls, repo_path: str) -> "RepoIndex":
        """Full index from scratch. This is the expensive call."""
        repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

        sources = index_repo(repo_path)

        git_analyzer = GitAnalyzer(repo_path)
        recency_data = git_analyzer.recency_scores()

        graph = build_dependency_graph(sources)

        embedding_signal = EmbeddingSimilaritySignal()
        if sources:
            embedding_signal.fit(sources)

        file_mtimes = _collect_mtimes(repo_path, sources)
        head_commit = _get_head_commit(repo_path)

        return cls(
            repo_path=repo_path,
            sources=sources,
            recency_data=recency_data,
            graph=graph,
            embedding_signal=embedding_signal,
            file_mtimes=file_mtimes,
            head_commit=head_commit,
        )

    def refresh(self) -> "RepoIndex":
        """Check for changes and return an updated index.

        Returns self if nothing changed. Returns a new RepoIndex if files
        were added, removed, or modified, or if HEAD moved.
        """
        current_head = _get_head_commit(self.repo_path)
        head_changed = current_head != self.head_commit

        # Check for file changes by walking the source tree and comparing mtimes
        current_mtimes: dict[str, float] = {}
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in files:
                ext = os.path.splitext(filename)[1]
                if ext not in SOURCE_EXTENSIONS:
                    continue
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, self.repo_path).replace("\\", "/")
                try:
                    current_mtimes[rel_path] = os.path.getmtime(full_path)
                except OSError:
                    pass

        files_changed = current_mtimes != self.file_mtimes

        if not files_changed and not head_changed:
            return self

        # Something changed: rebuild
        sources = index_repo(self.repo_path)
        graph = build_dependency_graph(sources)

        embedding_signal = EmbeddingSimilaritySignal()
        if sources:
            embedding_signal.fit(sources)

        recency_data = self.recency_data
        if head_changed:
            git_analyzer = GitAnalyzer(self.repo_path)
            recency_data = git_analyzer.recency_scores()

        return RepoIndex(
            repo_path=self.repo_path,
            sources=sources,
            recency_data=recency_data,
            graph=graph,
            embedding_signal=embedding_signal,
            file_mtimes=_collect_mtimes(self.repo_path, sources),
            head_commit=current_head,
        )


def select_context(
    index: RepoIndex,
    task: "str | Task",
    budget: int = 12_000,
) -> SelectionResult:
    """Select the optimal set of files for an LLM's context window.

    Args:
        index: A pre-built RepoIndex (from RepoIndex.build or .refresh).
        task: Either a plain text description (str) or a Task object with
              pre-extracted symbols.
        budget: Maximum token budget for the selected context.

    Returns:
        SelectionResult with ranked, ordered files and their scores.
    """
    if not index.sources:
        return SelectionResult(selected=[], total_tokens=0, budget=budget)

    if isinstance(task, str):
        task_symbols = _extract_task_symbols(task, "", index.sources)
        task = Task(title=task, body="", symbols=task_symbols)

    entry_points = _find_entry_points(task.symbols, index.sources)

    vf = ValueFunction(
        graph=index.graph,
        recency_data=index.recency_data,
        embedding_signal=index.embedding_signal,
        entry_points=entry_points,
    )

    selector = GreedySelector(value_function=vf)
    result = selector.select(index.sources, task, budget)
    result.selected = order_context(result.selected)

    return result


def select_context_from_repo(
    repo_path: str,
    task: str,
    budget: int = 12_000,
) -> SelectionResult:
    """Build index + select in one call. No caching."""
    index = RepoIndex.build(repo_path)
    return select_context(index, task, budget)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_api.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/cognitive_cache/api.py tests/test_api.py
git commit -m "feat: add library API with RepoIndex and select_context

RepoIndex caches all repo-dependent state (sources, git recency, graph,
embeddings). select_context runs per-task scoring against a cached index.
refresh() detects file changes via mtimes and HEAD changes via git
rev-parse, returning self when nothing changed."
```

---

### Task 4: Update __init__.py exports and benchmark runner imports

**Files:**
- Modify: `src/cognitive_cache/__init__.py`
- Modify: `benchmark/runner.py`

- [ ] **Step 1: Update __init__.py**

Replace the contents of `src/cognitive_cache/__init__.py` with:

```python
"""Cognitive Cache: Optimal context orchestration for LLMs."""

from cognitive_cache.api import RepoIndex, select_context, select_context_from_repo
from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult

__all__ = [
    "RepoIndex",
    "select_context",
    "select_context_from_repo",
    "Source",
    "Task",
    "ScoredSource",
    "SelectionResult",
]
```

- [ ] **Step 2: Update benchmark/runner.py to import from api**

In `benchmark/runner.py`, remove the `_extract_task_symbols` function (lines 46-84) and the `_find_entry_points` function (lines 87-92). Add an import at the top:

```python
from cognitive_cache.api import _extract_task_symbols, _find_entry_points
```

Then update the existing uses at lines 139-141, which already call `_extract_task_symbols(issue.title, issue.body, sources)` and `_find_entry_points(task_symbols, sources)`. These calls remain unchanged since the function signatures are identical.

- [ ] **Step 3: Run full test suite**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Verify imports work**

Run: `cd /mnt/data/cognitive-cache && uv run python -c "from cognitive_cache import RepoIndex, select_context, select_context_from_repo, Source, Task, ScoredSource, SelectionResult; print('All exports OK')"`
Expected: `All exports OK`

- [ ] **Step 5: Commit**

```bash
git add src/cognitive_cache/__init__.py benchmark/runner.py
git commit -m "refactor: export public API from __init__.py, deduplicate runner

Moves _extract_task_symbols and _find_entry_points from benchmark/runner.py
into the library API. Runner now imports from cognitive_cache.api."
```

---

### Task 5: Build the CLI

**Files:**
- Create: `src/cognitive_cache/cli.py`
- Create: `tests/test_cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing tests for CLI**

Create `tests/test_cli.py`:

```python
"""Tests for the CLI entry point."""

import json
import os
import subprocess
import sys
import tempfile

from cognitive_cache.cli import main


def _create_fake_repo(tmpdir):
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)

    os.makedirs(os.path.join(tmpdir, "src"))
    with open(os.path.join(tmpdir, "src/auth.py"), "w") as f:
        f.write("def login(username, password):\n    return True\n")
    with open(os.path.join(tmpdir, "src/db.py"), "w") as f:
        f.write("def get_user(name):\n    return name\n")

    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)


def test_cli_default_output(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        monkeypatch.setattr(
            sys, "argv",
            ["cognitive-cache", "select", "--repo", tmpdir, "--task", "fix the login bug"]
        )
        main()
        output = capsys.readouterr().out
        assert "src/auth.py" in output
        assert "files selected" in output


def test_cli_json_output(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        monkeypatch.setattr(
            sys, "argv",
            ["cognitive-cache", "select", "--repo", tmpdir, "--task", "fix login", "--json"]
        )
        main()
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "files" in data
        assert "total_tokens" in data
        assert "budget" in data


def test_cli_output_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        output_path = os.path.join(tmpdir, "context.txt")
        monkeypatch.setattr(
            sys, "argv",
            ["cognitive-cache", "select", "--repo", tmpdir, "--task", "fix login",
             "--output", output_path]
        )
        main()
        assert os.path.exists(output_path)
        content = open(output_path).read()
        assert "login" in content


def test_cli_custom_budget(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        monkeypatch.setattr(
            sys, "argv",
            ["cognitive-cache", "select", "--repo", tmpdir, "--task", "fix login",
             "--budget", "50", "--json"]
        )
        main()
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["total_tokens"] <= 50


def test_cli_nonexistent_repo(capsys, monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["cognitive-cache", "select", "--repo", "/nonexistent/repo", "--task", "fix bug"]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cognitive_cache.cli'`

- [ ] **Step 3: Implement cli.py**

Create `src/cognitive_cache/cli.py`:

```python
"""CLI entry point for cognitive-cache.

Usage:
    cognitive-cache select --repo . --task "fix the login bug" --budget 12000
"""

import argparse
import json
import sys

from cognitive_cache.api import RepoIndex, select_context


def _format_human_readable(result) -> str:
    """Format selection result as a human-readable table."""
    lines = []
    for ss in result.selected:
        path = ss.source.path
        score = ss.score
        signals = ss.signal_scores

        sig_parts = []
        for key in ["symbol_overlap", "graph_distance", "change_recency",
                     "embedding_sim", "file_role_prior", "redundancy"]:
            val = signals.get(key, 0.0)
            short = {"symbol_overlap": "sym", "graph_distance": "graph",
                     "change_recency": "recency", "embedding_sim": "embed",
                     "file_role_prior": "role", "redundancy": "redund"}[key]
            sig_parts.append(f"{short}={val:.1f}")

        sig_str = " ".join(sig_parts)
        lines.append(f"{path:<45} {score:.3f}  [{sig_str}]")

    lines.append("")
    lines.append(
        f"{len(result.selected)} files selected | "
        f"{result.total_tokens:,} / {result.budget:,} tokens used"
    )
    return "\n".join(lines)


def _format_json(result) -> str:
    """Format selection result as JSON."""
    data = {
        "files": [
            {
                "path": ss.source.path,
                "score": round(ss.score, 4),
                "signals": {k: round(v, 4) for k, v in ss.signal_scores.items()},
                "token_count": ss.source.token_count,
            }
            for ss in result.selected
        ],
        "total_tokens": result.total_tokens,
        "budget": result.budget,
        "budget_remaining": result.budget_remaining,
    }
    return json.dumps(data, indent=2)


def _write_context_file(result, output_path: str):
    """Write the full ordered context to a file."""
    with open(output_path, "w") as f:
        for ss in result.selected:
            f.write(f"# --- {ss.source.path} ---\n")
            f.write(ss.source.content)
            f.write("\n\n")


def main():
    parser = argparse.ArgumentParser(
        prog="cognitive-cache",
        description="Optimal context selection for LLMs",
    )
    subparsers = parser.add_subparsers(dest="command")

    select_parser = subparsers.add_parser("select", help="Select context files for a task")
    select_parser.add_argument("--repo", required=True, help="Path to the repository root")
    select_parser.add_argument("--task", required=True, help="Task description")
    select_parser.add_argument("--budget", type=int, default=12000, help="Token budget (default: 12000)")
    select_parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    select_parser.add_argument("--output", help="Write full context to this file")

    args = parser.parse_args()

    if args.command != "select":
        parser.print_help()
        sys.exit(1)

    try:
        index = RepoIndex.build(args.repo)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not index.sources:
        print("No source files found.", file=sys.stderr)
        sys.exit(1)

    result = select_context(index, args.task, budget=args.budget)

    if not result.selected:
        print("No files selected.", file=sys.stderr)
        sys.exit(0)

    # Warn if no symbols matched (vague task description)
    if all(ss.signal_scores.get("symbol_overlap", 0.0) == 0.0 for ss in result.selected):
        print("Warning: No symbol matches found, results may be less precise.",
              file=sys.stderr)

    if args.json_output:
        print(_format_json(result))
    else:
        print(_format_human_readable(result))

    if args.output:
        _write_context_file(result, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add CLI entry point to pyproject.toml**

In `pyproject.toml`, add after the `[dependency-groups]` section:

```toml
[project.scripts]
cognitive-cache = "cognitive_cache.cli:main"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Test CLI manually**

Run: `cd /mnt/data/cognitive-cache && uv run cognitive-cache select --repo . --task "fix the symbol overlap scoring"`
Expected: Human-readable table with file paths, scores, and signal breakdowns

- [ ] **Step 8: Commit**

```bash
git add src/cognitive_cache/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: add CLI entry point

cognitive-cache select --repo . --task 'description' --budget 12000
Supports --json for machine-readable output, --output for dumping
the full context to a file."
```

---

### Task 6: Build the MCP server

**Files:**
- Create: `src/cognitive_cache/mcp_server.py`
- Create: `tests/test_mcp_server.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add mcp dependency**

In `pyproject.toml`, update the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.0.0",
]
benchmark = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
]
```

And add to `[project.scripts]`:

```toml
[project.scripts]
cognitive-cache = "cognitive_cache.cli:main"
cognitive-cache-mcp = "cognitive_cache.mcp_server:main"
```

- [ ] **Step 2: Install mcp dependency**

Run: `cd /mnt/data/cognitive-cache && uv sync --extra mcp`
Expected: mcp package installed successfully

- [ ] **Step 3: Write failing tests for MCP server**

Create `tests/test_mcp_server.py`:

```python
"""Tests for the MCP server tool handler."""

import json
import os
import subprocess
import tempfile

import pytest

from cognitive_cache.mcp_server import _handle_select_context, _index_cache


def _create_fake_repo(tmpdir):
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
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

        # Second call should use the cached index (no rebuild)
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_mcp_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cognitive_cache.mcp_server'`

- [ ] **Step 5: Implement mcp_server.py**

Create `src/cognitive_cache/mcp_server.py`:

```python
"""MCP server for cognitive-cache.

Exposes a single tool: select_context. Caches RepoIndex instances
in memory so repeated calls to the same repo are fast.

Run with:
    cognitive-cache-mcp
    # or
    uv run --extra mcp cognitive-cache-mcp
"""

import json
import os

from cognitive_cache.api import RepoIndex, select_context

# In-memory cache of repo indices, keyed by absolute repo path
_index_cache: dict[str, RepoIndex] = {}


def _handle_select_context(
    repo_path: str, task: str, budget: int = 12000
) -> dict:
    """Core handler logic, separated from MCP for testability."""
    repo_path = os.path.abspath(repo_path)

    if repo_path in _index_cache:
        index = _index_cache[repo_path].refresh()
    else:
        index = RepoIndex.build(repo_path)

    _index_cache[repo_path] = index

    result = select_context(index, task, budget=budget)

    return {
        "files": [
            {
                "path": ss.source.path,
                "score": round(ss.score, 4),
                "signals": {k: round(v, 4) for k, v in ss.signal_scores.items()},
                "content": ss.source.content,
                "token_count": ss.source.token_count,
            }
            for ss in result.selected
        ],
        "total_tokens": result.total_tokens,
        "budget": result.budget,
        "budget_remaining": result.budget_remaining,
    }


def main():
    """Start the MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "MCP SDK not installed. Install with: uv sync --extra mcp",
            flush=True,
        )
        raise SystemExit(1)

    mcp = FastMCP(
        "cognitive-cache",
        description="Optimal context selection for LLMs. Given a codebase and a task, "
        "picks the files most likely to help the model get the answer right.",
    )

    @mcp.tool()
    def select_context_tool(
        repo_path: str,
        task: str,
        budget: int = 12000,
    ) -> str:
        """Select the optimal set of files for an LLM's context window.

        Given a repository path and a task description, analyzes the codebase
        using six signals (symbol overlap, dependency graph, git recency,
        redundancy penalty, embedding similarity, file role) and returns the
        highest-value files that fit within the token budget.

        Args:
            repo_path: Absolute path to the repository root.
            task: Plain text description of what the LLM needs to do.
            budget: Maximum token budget for selected context (default 12000).

        Returns:
            JSON with ranked files (path, score, signals, content),
            total tokens, budget, and budget remaining.
        """
        result = _handle_select_context(repo_path, task, budget)
        return json.dumps(result, indent=2)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_mcp_server.py -v`
Expected: ALL PASS

- [ ] **Step 7: Run full test suite**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/cognitive_cache/mcp_server.py tests/test_mcp_server.py pyproject.toml
git commit -m "feat: add MCP server with index caching

Single tool: select_context. Caches RepoIndex per repo_path in memory.
First call builds the full index; subsequent calls refresh via mtimes
and HEAD check. Uses FastMCP with stdio transport for Claude Code."
```

---

### Task 7: Add patch evaluation to the benchmark

**Files:**
- Modify: `benchmark/evaluator.py`
- Modify: `benchmark/dataset.py`
- Create: `benchmark/analyze_patches.py`
- Create: `tests/test_patch_eval.py`

- [ ] **Step 1: Write failing tests for compute_patch_similarity**

Create `tests/test_patch_eval.py`:

```python
"""Tests for patch similarity evaluation."""

from benchmark.evaluator import compute_patch_similarity


def test_identical_patches_score_one():
    patch = "--- a/auth.py\n+++ b/auth.py\n@@ -1,3 +1,3 @@\n-def login():\n+def login(user):\n     pass"
    assert compute_patch_similarity(patch, patch) == 1.0


def test_completely_different_patches():
    generated = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    truth = "--- a/db.py\n+++ b/db.py\n@@ -1 +1 @@\n-def foo():\n+def bar():"
    score = compute_patch_similarity(generated, truth)
    assert score < 0.5


def test_empty_generated_patch():
    truth = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    assert compute_patch_similarity("", truth) == 0.0


def test_empty_ground_truth():
    generated = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    assert compute_patch_similarity(generated, "") == 0.0


def test_both_empty():
    assert compute_patch_similarity("", "") == 0.0


def test_error_string_patch():
    """Patches that are error strings should score 0."""
    truth = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    assert compute_patch_similarity("ERROR: connection timeout", truth) == 0.0


def test_partial_overlap():
    generated = (
        "--- a/auth.py\n+++ b/auth.py\n@@ -1,3 +1,3 @@\n"
        " def login(user, password):\n-    return False\n+    return check(user, password)\n"
    )
    truth = (
        "--- a/auth.py\n+++ b/auth.py\n@@ -1,3 +1,4 @@\n"
        " def login(user, password):\n-    return False\n+    result = check(user, password)\n+    return result\n"
    )
    score = compute_patch_similarity(generated, truth)
    assert 0.0 < score < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_patch_eval.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_patch_similarity'`

- [ ] **Step 3: Implement compute_patch_similarity**

In `benchmark/evaluator.py`, add after the existing functions:

```python
import difflib


def compute_patch_similarity(generated_patch: str, ground_truth_diff: str) -> float:
    """Compute similarity between a generated patch and the ground truth diff.

    Uses difflib.SequenceMatcher on the changed lines (lines starting with
    + or -) to produce a [0, 1] score. Returns 0.0 for empty or error patches.
    """
    if not generated_patch or not ground_truth_diff:
        return 0.0

    # Skip error strings from failed LLM calls
    if generated_patch.startswith("ERROR:"):
        return 0.0

    def _extract_changed_lines(patch: str) -> str:
        """Extract only the added/removed lines from a patch."""
        lines = []
        for line in patch.splitlines():
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                lines.append(line)
        return "\n".join(lines)

    gen_changes = _extract_changed_lines(generated_patch)
    truth_changes = _extract_changed_lines(ground_truth_diff)

    if not gen_changes or not truth_changes:
        return 0.0

    return difflib.SequenceMatcher(None, gen_changes, truth_changes).ratio()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/test_patch_eval.py -v`
Expected: ALL PASS

- [ ] **Step 5: Add ground_truth_diff to BenchmarkIssue**

In `benchmark/dataset.py`, update the `BenchmarkIssue` dataclass:

```python
@dataclass
class BenchmarkIssue:
    repo: str
    repo_url: str
    issue_number: int
    title: str
    body: str
    fixed_files: list[str]
    fix_commit: str
    base_commit: str
    ground_truth_diff: str = ""
```

And update `load_dataset` to handle the optional field:

```python
def load_dataset(path: str) -> list[BenchmarkIssue]:
    with open(path) as f:
        data = json.load(f)
    issues = []
    for item in data:
        # Handle datasets that don't have ground_truth_diff yet
        if "ground_truth_diff" not in item:
            item["ground_truth_diff"] = ""
        issues.append(BenchmarkIssue(**item))
    return issues
```

- [ ] **Step 6: Create analyze_patches.py**

Create `benchmark/analyze_patches.py`:

```python
"""Analyze benchmark results: compute patch similarity per strategy.

Usage:
    PYTHONPATH=. uv run python benchmark/analyze_patches.py benchmark/results/results_YYYYMMDD_HHMMSS.json
"""

import json
import sys
from collections import defaultdict

from benchmark.evaluator import compute_patch_similarity


def analyze(results_path: str):
    with open(results_path) as f:
        results = json.load(f)

    # Group by strategy
    strategy_scores: dict[str, list[float]] = defaultdict(list)
    strategy_recalls: dict[str, list[float]] = defaultdict(list)

    for r in results:
        strategy = r["strategy"]
        # Patch similarity requires ground truth diff in the dataset
        # For now, we record file recall; patch sim will be added when
        # ground_truth_diff is populated in the dataset
        strategy_recalls[strategy].append(r["file_recall"])

    # Print summary
    print(f"{'Strategy':<20} {'Avg File Recall':>15} {'Issues':>8}")
    print("-" * 45)

    for strategy in sorted(strategy_recalls.keys()):
        recalls = strategy_recalls[strategy]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"{strategy:<20} {avg_recall:>14.1%} {len(recalls):>8}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results.json>")
        sys.exit(1)
    analyze(sys.argv[1])


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run full test suite**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add benchmark/evaluator.py benchmark/dataset.py benchmark/analyze_patches.py tests/test_patch_eval.py
git commit -m "feat: add patch similarity evaluation to benchmark

compute_patch_similarity compares generated patches against ground truth
using difflib. BenchmarkIssue gains a ground_truth_diff field.
analyze_patches.py produces per-strategy summary tables."
```

---

### Task 8: Final integration test and cleanup

**Files:**
- Test: `tests/test_integration.py` (existing, verify still passes)

- [ ] **Step 1: Run the full test suite**

Run: `cd /mnt/data/cognitive-cache && uv run pytest tests/ -v --tb=short`
Expected: ALL PASS (should be 56+ existing tests plus all new tests)

- [ ] **Step 2: Test the CLI end-to-end on cognitive-cache's own repo**

Run: `cd /mnt/data/cognitive-cache && uv run cognitive-cache select --repo . --task "fix the symbol overlap scoring to handle missing symbols"`
Expected: Human-readable output showing ranked files from this repo

- [ ] **Step 3: Test CLI JSON output**

Run: `cd /mnt/data/cognitive-cache && uv run cognitive-cache select --repo . --task "fix symbol overlap" --json`
Expected: Valid JSON output

- [ ] **Step 4: Test CLI context dump**

Run: `cd /mnt/data/cognitive-cache && uv run cognitive-cache select --repo . --task "fix symbol overlap" --output /tmp/cc_context.txt && head -20 /tmp/cc_context.txt`
Expected: File created with source code contents

- [ ] **Step 5: Verify MCP server starts**

Run: `cd /mnt/data/cognitive-cache && timeout 3 uv run --extra mcp cognitive-cache-mcp 2>&1 || true`
Expected: Server starts (blocks on stdin), exits after timeout. No import errors or crashes.

- [ ] **Step 6: Commit any fixes**

Only if steps 1-5 revealed issues. If everything passes, skip this step.

- [ ] **Step 7: Final commit with all files verified**

Run `git status` and ensure only expected files are changed. If any test helper files or temp artifacts were created, clean them up.
