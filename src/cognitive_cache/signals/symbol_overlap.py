"""Signal 1: Symbol Overlap.

Scores a file by how many of the task's mentioned symbols appear in the file.
This is the simplest and fastest signal — direct string matching on identifiers.

Example: if the task mentions "login" and "validate", and a file defines both,
it scores 1.0. If it defines only "login", it scores 0.5.
"""

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal


class SymbolOverlapSignal(Signal):
    def score(self, source: Source, task: Task, selected: list[Source], **kwargs) -> float:
        if not task.symbols:
            return 0.0
        overlap = source.symbols & task.symbols
        return len(overlap) / len(task.symbols)
