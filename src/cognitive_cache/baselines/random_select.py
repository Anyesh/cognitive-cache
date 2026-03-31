"""Baseline 1: Random Selection. Lower bound."""

import random as random_module

from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult
from cognitive_cache.baselines.base import BaselineStrategy


class RandomStrategy(BaselineStrategy):
    def __init__(self, seed: int | None = None):
        self._seed = seed

    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        rng = random_module.Random(self._seed)
        shuffled = list(sources)
        rng.shuffle(shuffled)

        selected = []
        total = 0
        for s in shuffled:
            if total + s.token_count > budget:
                continue
            selected.append(ScoredSource(source=s, score=0.0))
            total += s.token_count

        return SelectionResult(selected=selected, total_tokens=total, budget=budget)
