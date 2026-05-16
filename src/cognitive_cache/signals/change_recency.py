"""Signal 3: Change Recency.

Scores a file by how recently it was modified in git history.
Gated on structural relevance: if a file's max(symbol_overlap, embedding_sim)
is below _RELEVANCE_FLOOR, recency returns 0.0 to prevent recently-changed-but-
unrelated files from leaking into results. TF-IDF and embedding scores rarely
hit exactly 0.0, so a threshold catches noise-level matches that the old exact
comparison missed.
"""

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal

_RELEVANCE_FLOOR = 0.10


class ChangeRecencySignal(Signal):
    def __init__(self, recency_data: dict[str, float]):
        self._recency = recency_data

    def score(
        self, source: Source, task: Task, selected: list[Source], **kwargs
    ) -> float:
        symbol_score: float = kwargs.get("symbol_overlap_score", 1.0)
        embedding_score: float = kwargs.get("embedding_sim_score", 1.0)
        if max(symbol_score, embedding_score) < _RELEVANCE_FLOOR:
            return 0.0
        return self._recency.get(source.path, 0.0)
