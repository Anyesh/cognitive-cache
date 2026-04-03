"""Signal 3: Change Recency.

Scores a file by how recently it was modified in git history.
The intuition: bugs are more likely in recently changed code.

Takes pre-computed recency data from GitAnalyzer so we don't
re-run git commands for every candidate file.
"""

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal


class ChangeRecencySignal(Signal):
    def __init__(self, recency_data: dict[str, float]):
        self._recency = recency_data

    def score(self, source: Source, task: Task, selected: list[Source], **kwargs) -> float:
        return self._recency.get(source.path, 0.0)
