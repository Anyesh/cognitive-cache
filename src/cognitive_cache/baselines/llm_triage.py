"""Baseline 5: LLM Triage (let the brain pick its own context).

Two-pass strategy — requires an LLM adapter. Only used during benchmark runs.
"""

from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult
from cognitive_cache.baselines.base import BaselineStrategy


class LLMTriageStrategy(BaselineStrategy):
    def __init__(self, llm_adapter=None):
        self._llm = llm_adapter

    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        if self._llm is None:
            raise RuntimeError("LLMTriageStrategy requires an LLM adapter")

        file_listing = "\n".join(s.path for s in sources)
        triage_prompt = (
            f"You are helping fix a bug. Here is the issue:\n\n"
            f"Title: {task.title}\n"
            f"Body: {task.body}\n\n"
            f"Here are all the files in the repository:\n\n"
            f"{file_listing}\n\n"
            f"Which files are most likely relevant to this issue? "
            f"List just the file paths, one per line. Pick at most 20 files."
        )

        response = self._llm.complete(triage_prompt, max_tokens=500)

        source_by_path = {s.path: s for s in sources}
        selected = []
        total = 0
        for line in response.strip().split("\n"):
            path = line.strip().strip("- ").strip("`")
            if path in source_by_path:
                source = source_by_path[path]
                if total + source.token_count <= budget:
                    selected.append(ScoredSource(source=source, score=1.0))
                    total += source.token_count

        return SelectionResult(selected=selected, total_tokens=total, budget=budget)
