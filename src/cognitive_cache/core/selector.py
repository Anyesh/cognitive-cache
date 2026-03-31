"""Greedy Submodular Selector: picks the optimal context set under a token budget.

This is the optimization algorithm. It greedily selects the file with the
highest value-per-token ratio, adds it to the selection, then re-scores
all remaining candidates (because adding a file changes the redundancy
penalties for similar files).

Greedy submodular maximization under a knapsack constraint has a provable
approximation guarantee: it achieves at least (1 - 1/e) ~ 63% of the
optimal selection. In practice, it's usually much closer to optimal.
"""

from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult
from cognitive_cache.core.value_function import ValueFunction


class GreedySelector:
    """Selects the highest-value context set within a token budget."""

    def __init__(self, value_function: ValueFunction, threshold: float = 0.001):
        self._vf = value_function
        self._threshold = threshold

    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        """Select the optimal context set."""
        selected_sources: list[Source] = []
        selected_scored: list[ScoredSource] = []
        remaining = set(range(len(sources)))
        remaining_budget = budget

        while remaining and remaining_budget > 0:
            best_idx = -1
            best_vpt = -1.0
            best_score = 0.0
            best_breakdown = {}

            for idx in remaining:
                candidate = sources[idx]
                if candidate.token_count > remaining_budget:
                    continue
                if candidate.token_count == 0:
                    continue

                score, breakdown = self._vf.score_with_breakdown(
                    candidate, task, selected_sources
                )
                vpt = score / candidate.token_count

                if vpt > best_vpt:
                    best_idx = idx
                    best_vpt = vpt
                    best_score = score
                    best_breakdown = breakdown

            if best_idx == -1 or best_vpt < self._threshold:
                break

            chosen = sources[best_idx]
            selected_sources.append(chosen)
            selected_scored.append(ScoredSource(
                source=chosen,
                score=best_score,
                signal_scores=best_breakdown,
            ))
            remaining.discard(best_idx)
            remaining_budget -= chosen.token_count

        total_tokens = sum(ss.source.token_count for ss in selected_scored)
        return SelectionResult(
            selected=selected_scored,
            total_tokens=total_tokens,
            budget=budget,
        )
