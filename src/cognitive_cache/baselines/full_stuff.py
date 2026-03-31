"""Baseline 2: Full Stuff. Cram as many files as fit, in filesystem order."""

from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult
from cognitive_cache.baselines.base import BaselineStrategy


class FullStuffStrategy(BaselineStrategy):
    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        sorted_sources = sorted(sources, key=lambda s: s.path)

        selected = []
        total = 0
        for s in sorted_sources:
            if total + s.token_count > budget:
                continue
            selected.append(ScoredSource(source=s, score=0.0))
            total += s.token_count

        return SelectionResult(selected=selected, total_tokens=total, budget=budget)
