"""Baseline 4: Grep Strategy (simulates Claude Code / Copilot symbol search)."""

from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult
from cognitive_cache.baselines.base import BaselineStrategy


class GrepStrategy(BaselineStrategy):
    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        if not task.symbols:
            keywords = set(task.full_text.lower().split())
        else:
            keywords = {s.lower() for s in task.symbols}

        scored = []
        for source in sources:
            content_lower = source.content.lower()
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                scored.append((source, matches))

        scored.sort(key=lambda x: x[1], reverse=True)

        selected = []
        total = 0
        for source, matches in scored:
            if total + source.token_count > budget:
                continue
            score = matches / max(len(keywords), 1)
            selected.append(ScoredSource(source=source, score=score))
            total += source.token_count

        return SelectionResult(selected=selected, total_tokens=total, budget=budget)
