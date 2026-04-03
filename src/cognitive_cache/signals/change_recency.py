"""Signal 3: Change Recency.

Scores a file by how recently it was modified in git history.
The intuition: bugs are more likely in recently changed code.

Takes pre-computed recency data from GitAnalyzer so we don't
re-run git commands for every candidate file.

On shallow clones, files outside the cloned history get a neutral
fallback score (0.3) instead of zero, since their absence from
the history is an artifact of the clone depth.
"""

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal

_SHALLOW_FALLBACK = 0.3


class ChangeRecencySignal(Signal):
    def __init__(self, recency_data: dict[str, float], is_shallow: bool = False):
        self._recency = recency_data
        self._default = _SHALLOW_FALLBACK if is_shallow else 0.0

    def score(
        self, source: Source, task: Task, selected: list[Source], **kwargs
    ) -> float:
        return self._recency.get(source.path, self._default)
