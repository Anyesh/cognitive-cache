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
    """Selects the highest-value context set within a token budget."""

    def __init__(
        self,
        value_function: ValueFunction,
        score_threshold: float = 0.015,
        vpt_threshold: float = 0.0001,
        max_files: int = 15,
    ):
        self._vf = value_function
        self._score_threshold = score_threshold  # min absolute score for phase 1
        self._vpt_threshold = vpt_threshold      # min value-per-token for phase 2
        self._max_files = max_files

    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        """Select the optimal context set using two-phase greedy selection."""
        # Score all candidates once
        candidates = []
        for i, source in enumerate(sources):
            if source.token_count == 0:
                continue
            score, breakdown = self._vf.score_with_breakdown(source, task, [])
            candidates.append((i, source, score, breakdown))

        # Phase 1: pick high-value files by absolute score (greedy with redundancy)
        selected_sources: list[Source] = []
        selected_scored: list[ScoredSource] = []
        used_indices = set()
        remaining_budget = budget

        # Sort by absolute score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        for idx, source, initial_score, _ in candidates:
            if len(selected_scored) >= self._max_files:
                break
            if remaining_budget <= 0:
                break

            # Re-score with current selection (accounts for redundancy)
            score, breakdown = self._vf.score_with_breakdown(
                source, task, selected_sources
            )

            if score < self._score_threshold:
                continue

            # If the file is too large, chunk it to extract relevant portions
            if source.token_count > remaining_budget:
                source = chunk_source(source, task, max_tokens=remaining_budget)
                if source.token_count > remaining_budget or source.token_count == 0:
                    continue

            selected_sources.append(source)
            selected_scored.append(ScoredSource(
                source=source, score=score, signal_scores=breakdown,
            ))
            used_indices.add(idx)
            remaining_budget -= source.token_count

        # Phase 2: fill remaining budget with supporting files by VPT
        remaining_candidates = [
            (i, s, sc, bd) for i, s, sc, bd in candidates
            if i not in used_indices
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

            if vpt < self._vpt_threshold:
                continue

            selected_sources.append(source)
            selected_scored.append(ScoredSource(
                source=source, score=score, signal_scores=breakdown,
            ))
            remaining_budget -= source.token_count

        total_tokens = sum(ss.source.token_count for ss in selected_scored)
        return SelectionResult(
            selected=selected_scored,
            total_tokens=total_tokens,
            budget=budget,
        )
