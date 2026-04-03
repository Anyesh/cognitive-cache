"""Benchmark runner: orchestrates the full experiment.

For each issue: index the repo, run all strategies, call LLMs, record results.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime

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
from cognitive_cache.baselines.llm_triage import LLMTriageStrategy
from cognitive_cache.llm.adapter import LLMAdapter
from cognitive_cache.api import _extract_task_symbols, _find_entry_points

from benchmark.dataset import BenchmarkIssue
from benchmark.prompt_template import build_prompt
from benchmark.evaluator import compute_file_recall


@dataclass
class RunResult:
    issue_repo: str
    issue_number: int
    strategy: str
    model: str
    selected_files: list[str]
    file_recall: float
    tokens_used: int
    generated_patch: str
    timestamp: str


def _clone_at_commit(repo_url: str, commit: str, target_dir: str):
    subprocess.run(
        ["git", "clone", "--quiet", repo_url, target_dir],
        capture_output=True,
        check=True,
        timeout=120,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", commit],
        cwd=target_dir,
        capture_output=True,
        check=True,
    )


def run_benchmark(
    issues: list[BenchmarkIssue],
    llm_adapters: dict[str, LLMAdapter],
    budget: int = 12000,
    output_dir: str = "benchmark/results",
) -> list[RunResult]:
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for issue in issues:
        print(f"\n{'=' * 60}", flush=True)
        print(
            f"Issue: {issue.repo}#{issue.issue_number} - {issue.title}".encode(
                "ascii", "replace"
            ).decode(),
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = os.path.join(tmpdir, "repo")
            try:
                _clone_at_commit(issue.repo_url, issue.base_commit, repo_dir)
            except subprocess.CalledProcessError as e:
                print(f"  SKIP: Failed to clone {issue.repo_url}: {e}", flush=True)
                continue

            sources = index_repo(repo_dir)
            if not sources:
                print(f"  SKIP: No source files found", flush=True)
                continue

            git_analyzer = GitAnalyzer(repo_dir)
            recency_data = git_analyzer.recency_scores()
            graph = build_dependency_graph(sources)
            embedding_signal = EmbeddingSimilaritySignal()
            embedding_signal.fit(sources)

            task_symbols = _extract_task_symbols(issue.title, issue.body, sources)
            task = Task(title=issue.title, body=issue.body, symbols=task_symbols)
            entry_points = _find_entry_points(task_symbols, sources)

            vf = ValueFunction(
                graph=graph,
                recency_data=recency_data,
                embedding_signal=embedding_signal,
                entry_points=entry_points,
            )

            strategies: dict[str, object] = {
                "random": RandomStrategy(seed=42),
                "full_stuff": FullStuffStrategy(),
                "embedding": EmbeddingStrategy(),
                "grep": GrepStrategy(),
                "cognitive_cache": GreedySelector(value_function=vf),
            }

            actual_files = set(issue.fixed_files)

            for strategy_name, strategy in strategies.items():
                if strategy_name == "cognitive_cache":
                    result = strategy.select(sources, task, budget)
                    ordered = order_context(result.selected)
                    selected_scored = ordered
                else:
                    result = strategy.select(sources, task, budget)
                    selected_scored = result.selected

                selected_paths = {ss.source.path for ss in selected_scored}
                file_recall = compute_file_recall(selected_paths, actual_files)

                context_files = {
                    ss.source.path: ss.source.content for ss in selected_scored
                }
                prompt = build_prompt(issue.title, issue.body, context_files)

                for model_name, adapter in llm_adapters.items():
                    print(
                        f"  [{strategy_name}] [{model_name}] recall={file_recall:.2f} tokens={result.total_tokens}",
                        flush=True,
                    )

                    try:
                        patch = adapter.complete(
                            prompt, max_tokens=4096, temperature=0.0
                        )
                    except Exception as e:
                        print(f"    ERROR: {e}", flush=True)
                        patch = f"ERROR: {e}"

                    all_results.append(
                        RunResult(
                            issue_repo=issue.repo,
                            issue_number=issue.issue_number,
                            strategy=strategy_name,
                            model=model_name,
                            selected_files=list(selected_paths),
                            file_recall=file_recall,
                            tokens_used=result.total_tokens,
                            generated_patch=patch,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

            # LLM-triage baseline (needs LLM adapter)
            for model_name, adapter in llm_adapters.items():
                triage = LLMTriageStrategy(llm_adapter=adapter)
                try:
                    result = triage.select(sources, task, budget)
                except Exception as e:
                    print(f"  [llm_triage] [{model_name}] ERROR: {e}", flush=True)
                    continue

                selected_paths = {ss.source.path for ss in result.selected}
                file_recall = compute_file_recall(selected_paths, actual_files)
                context_files = {
                    ss.source.path: ss.source.content for ss in result.selected
                }
                prompt = build_prompt(issue.title, issue.body, context_files)

                print(
                    f"  [llm_triage] [{model_name}] recall={file_recall:.2f} tokens={result.total_tokens}",
                    flush=True,
                )

                try:
                    patch = adapter.complete(prompt, max_tokens=4096, temperature=0.0)
                except Exception as e:
                    patch = f"ERROR: {e}"

                all_results.append(
                    RunResult(
                        issue_repo=issue.repo,
                        issue_number=issue.issue_number,
                        strategy="llm_triage",
                        model=model_name,
                        selected_files=list(selected_paths),
                        file_recall=file_recall,
                        tokens_used=result.total_tokens,
                        generated_patch=patch,
                        timestamp=datetime.now().isoformat(),
                    )
                )

    results_path = os.path.join(
        output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {results_path}", flush=True)

    return all_results
