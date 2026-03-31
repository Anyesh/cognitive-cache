"""Base interface for all scoring signals.

Every signal takes a candidate source, the task, and the already-selected
sources, and returns a float in [0, 1]. Higher = more valuable (or for
redundancy: higher = more redundant).
"""

from abc import ABC, abstractmethod

from cognitive_cache.models import Source, Task


class Signal(ABC):
    @abstractmethod
    def score(self, source: Source, task: Task, selected: list[Source], **kwargs) -> float:
        """Score a candidate source.

        Args:
            source: The candidate file to score.
            task: The task description.
            selected: Sources already selected (for redundancy checking).
            **kwargs: Signal-specific context (e.g., entry_points for graph distance).

        Returns:
            Float in [0, 1].
        """
        ...
