"""Base interface for context selection strategies (baselines + our algorithm)."""

from abc import ABC, abstractmethod

from cognitive_cache.models import Source, Task, SelectionResult


class BaselineStrategy(ABC):
    @abstractmethod
    def select(self, sources: list[Source], task: Task, budget: int) -> SelectionResult:
        ...
