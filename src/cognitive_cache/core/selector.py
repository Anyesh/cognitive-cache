"""Greedy Submodular Selector: picks the optimal context set under a token budget.

Two-phase selection:
1. First, select high-value files by absolute score (regardless of size).
   These are the files most likely to contain the fix.
2. Then, fill remaining budget with supporting files by value-per-token.
   These provide context around the core files.

This hybrid approach avoids the trap of pure VPT scoring, which favors
tiny files over important-but-large core files like app.py.
"""

from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult
from cognitive_cache.core.value_function import ValueFunction
from cognitive_cache.core.chunker import chunk_source


class GreedySelector:
    def __init__(
        self,
        value_function: ValueFunction,
        score_threshold: float = 0.10,
        vpt_threshold: float = 0.0002,
        max_files: int = 15,
        include_tests: bool = True,
        min_score: float = 0.0,
    ):
        self._vf = value_function
        self._score_threshold = score_threshold
        self._vpt_threshold = vpt_threshold
        self._max_files = max_files
        self._include_tests = include_tests
        self._min_score = min_score

    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        candidates = []
        for i, source in enumerate(sources):
            if source.token_count == 0:
                continue
            if not self._include_tests and source.is_test:
                continue
            score, breakdown = self._vf.score_with_breakdown(source, task, [])
            candidates.append((i, source, score, breakdown))

        selected_sources: list[Source] = []
        selected_scored: list[ScoredSource] = []
        used_indices = set()
        remaining_budget = budget

        candidates.sort(key=lambda x: x[2], reverse=True)

        for idx, source, initial_score, _ in candidates:
            if len(selected_scored) >= self._max_files:
                break
            if remaining_budget <= 0:
                break

            score, breakdown = self._vf.score_with_breakdown(
                source, task, selected_sources
            )

            if score < self._score_threshold:
                continue
            if score < self._min_score:
                continue

            if source.token_count > remaining_budget:
                source = chunk_source(source, task, max_tokens=remaining_budget)
                if source.token_count > remaining_budget or source.token_count == 0:
                    continue

            selected_sources.append(source)
            selected_scored.append(
                ScoredSource(
                    source=source,
                    score=score,
                    signal_scores=breakdown,
                )
            )
            used_indices.add(idx)
            remaining_budget -= source.token_count

        remaining_candidates = [
            (i, s, sc, bd) for i, s, sc, bd in candidates if i not in used_indices
        ]

        for idx, source, _, _ in remaining_candidates:
            if len(selected_scored) >= self._max_files:
                break
            if remaining_budget <= 0:
                break
            if source.token_count > remaining_budget:
                continue

            score, breakdown = self._vf.score_with_breakdown(
                source, task, selected_sources
            )
            vpt = score / source.token_count

            if score < self._score_threshold:
                continue
            if vpt < self._vpt_threshold:
                continue
            if score < self._min_score:
                continue

            selected_sources.append(source)
            selected_scored.append(
                ScoredSource(
                    source=source,
                    score=score,
                    signal_scores=breakdown,
                )
            )
            remaining_budget -= source.token_count

        total_tokens = sum(ss.source.token_count for ss in selected_scored)
        return SelectionResult(
            selected=selected_scored,
            total_tokens=total_tokens,
            budget=budget,
        )
