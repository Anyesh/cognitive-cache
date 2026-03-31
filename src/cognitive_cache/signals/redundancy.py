"""Signal 4: Redundancy Penalty.

Measures how much a candidate file overlaps with already-selected files.
High redundancy = high score (this gets SUBTRACTED in the value function).

Uses Jaccard similarity on symbol sets: |A intersection B| / |A union B|.
If multiple files are selected, we take the MAX similarity to any of them,
because even one highly similar file makes this candidate redundant.
"""

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal


class RedundancySignal(Signal):
    def score(self, source: Source, task: Task, selected: list[Source], **kwargs) -> float:
        if not selected or not source.symbols:
            return 0.0

        max_sim = 0.0
        for s in selected:
            if not s.symbols:
                continue
            intersection = len(source.symbols & s.symbols)
            union = len(source.symbols | s.symbols)
            if union > 0:
                max_sim = max(max_sim, intersection / union)

        return max_sim
