"""Public API for Cognitive Cache: RepoIndex and select_context.

This module is the single entry point for library consumers. The CLI
and MCP server are thin wrappers around RepoIndex and select_context;
everything else (indexing, scoring, selection) is internal machinery.
"""

import os
import re
import subprocess
from dataclasses import dataclass

from cognitive_cache.models import Source, Task, SelectionResult
from cognitive_cache.indexer.repo_indexer import (
    index_repo,
    SOURCE_EXTENSIONS,
    SKIP_DIRS,
)
from cognitive_cache.indexer.git_analyzer import GitAnalyzer
from cognitive_cache.indexer.graph_builder import build_dependency_graph
from cognitive_cache.signals.embedding_sim import EmbeddingSimilaritySignal
from cognitive_cache.core.value_function import ValueFunction, WeightConfig
from cognitive_cache.core.selector import GreedySelector
from cognitive_cache.core.orderer import order_context

_TEST_KEYWORDS = {"test", "spec", "testing", "coverage", "fixture", "mock", "stub"}


def _extract_task_symbols(
    title: str, body: str, sources: list[Source]
) -> frozenset[str]:
    """Extract symbols from task text using exact and selective substring matching.

    Strategy:
    1. Exact match: symbol name appears verbatim in issue text (strongest signal).
    2. Substring match: issue words (6+ chars) appear in symbol names.
       Short common words like 'error', 'class', 'test' cause too many false
       matches, so we only use longer words.
    """
    all_symbols = set()
    for s in sources:
        all_symbols.update(s.symbols)

    text = f"{title} {body}".lower()
    task_words_long = set(re.findall(r"\b[a-z_][a-z0-9_]{5,}\b", text))
    stop_words = {
        "return",
        "import",
        "should",
        "string",
        "number",
        "before",
        "after",
        "called",
        "values",
        "object",
        "update",
        "create",
        "delete",
        "method",
        "function",
        "default",
        "option",
        "options",
        "config",
        "module",
        "result",
        "response",
        "request",
        "handler",
        "callback",
        "parameter",
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

    # Vague-task fallback: when no symbols matched, use long task words as
    # pseudo-symbols so that symbol_overlap and graph_distance don't both
    # zero out (which collapses 55% of scoring weight)
    if not matches and task_words_long:
        matches = set(list(task_words_long)[:10])

    return frozenset(matches)


def _find_entry_points(task_symbols: frozenset[str], sources: list[Source]) -> set[str]:
    entry_points = set()
    for s in sources:
        if s.symbols & task_symbols:
            entry_points.add(s.path)
    return entry_points


def _get_head_commit(repo_path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _collect_mtimes(repo_path: str, sources: list[Source]) -> dict[str, float]:
    mtimes: dict[str, float] = {}
    for s in sources:
        full_path = os.path.join(repo_path, s.path)
        try:
            mtimes[s.path] = os.path.getmtime(full_path)
        except OSError:
            pass
    return mtimes


@dataclass
class RepoIndex:
    """An indexed snapshot of a repository, ready for context selection.

    Build one with ``RepoIndex.build(repo_path)`` and reuse it across
    multiple ``select_context`` calls. Call ``refresh()`` to cheaply detect
    whether the repo has changed since the last index.
    """

    repo_path: str
    sources: list[Source]
    recency_data: dict[str, float]
    graph: object  # DependencyGraph
    embedding_signal: EmbeddingSimilaritySignal
    file_mtimes: dict[str, float]
    head_commit: str

    @classmethod
    def build(cls, repo_path: str) -> "RepoIndex":
        """Build a full index from scratch.

        Raises FileNotFoundError if repo_path does not exist.
        """
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

        sources = index_repo(repo_path)
        graph = build_dependency_graph(sources, repo_path=repo_path)

        git_analyzer = GitAnalyzer(repo_path)
        recency_data = git_analyzer.recency_scores()

        embedding_signal = EmbeddingSimilaritySignal()
        if sources:
            embedding_signal.fit(sources)

        return cls(
            repo_path=repo_path,
            sources=sources,
            recency_data=recency_data,
            graph=graph,
            embedding_signal=embedding_signal,
            file_mtimes=_collect_mtimes(repo_path, sources),
            head_commit=_get_head_commit(repo_path),
        )

    def refresh(self) -> "RepoIndex":
        """Check whether the repo has changed and rebuild if needed.

        Returns ``self`` if nothing changed. Returns a new RepoIndex if
        files were modified or HEAD moved. Only re-fetches git recency
        data when HEAD has actually moved, since that's the expensive part.
        """
        current_head = _get_head_commit(self.repo_path)
        head_changed = current_head != self.head_commit

        # Walk the source tree and compare mtimes against our snapshot.
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

        # Something changed, so rebuild the index.
        sources = index_repo(self.repo_path)
        graph = build_dependency_graph(sources, repo_path=self.repo_path)
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


def _task_mentions_testing(task: Task) -> bool:
    return any(kw in task.full_text.lower() for kw in _TEST_KEYWORDS)


def select_context(
    index: RepoIndex,
    task: str | Task,
    budget: int = 12_000,
    weights: WeightConfig | None = None,
    include_tests: bool | None = None,
    max_files: int = 15,
    min_score: float = 0.0,
) -> SelectionResult:
    """Select the most valuable context for a task from an indexed repo.

    Args:
        index: Pre-built repo index.
        task: Plain text task description or Task object.
        budget: Maximum token budget for selected context.
        weights: Custom signal weights. None uses defaults.
        include_tests: True=always include, False=always exclude,
            None=auto-detect from task text (include only if task mentions testing).
        max_files: Maximum number of files to return.
        min_score: Minimum score threshold for returned files.
    """
    if not index.sources:
        return SelectionResult(selected=[], total_tokens=0, budget=budget)

    if isinstance(task, str):
        task_symbols = _extract_task_symbols(task, "", index.sources)
        task = Task(title=task, body="", symbols=task_symbols)

    if include_tests is None:
        include_tests = _task_mentions_testing(task)

    entry_points = _find_entry_points(task.symbols, index.sources)
    vf = ValueFunction(
        weights=weights,
        graph=index.graph,
        recency_data=index.recency_data,
        embedding_signal=index.embedding_signal,
        entry_points=entry_points,
    )
    selector = GreedySelector(
        value_function=vf,
        include_tests=include_tests,
        max_files=max_files,
        min_score=min_score,
    )
    result = selector.select(index.sources, task, budget)
    result.selected = order_context(result.selected)
    return result


def select_context_from_repo(
    repo_path: str,
    task: str | Task,
    budget: int = 12_000,
    weights: WeightConfig | None = None,
    include_tests: bool | None = None,
    max_files: int = 15,
    min_score: float = 0.0,
) -> SelectionResult:
    """Convenience wrapper that builds an index and selects context in one call.

    Useful for one-shot usage where you don't need to reuse the index.
    For repeated queries against the same repo, build a RepoIndex once
    and call select_context directly.
    """
    index = RepoIndex.build(repo_path)
    return select_context(
        index,
        task,
        budget,
        weights=weights,
        include_tests=include_tests,
        max_files=max_files,
        min_score=min_score,
    )
