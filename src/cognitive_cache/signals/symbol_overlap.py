"""Signal 1: Symbol Overlap.

Scores a file by how many of the task's mentioned symbols appear in the file.
Two-level matching:
  1. Defined symbols: task symbols that match function/class names in the file
  2. Content keywords: task symbols that appear anywhere in the file content

Content matching is weighted lower (0.5x) since it's noisier.
"""

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal


class SymbolOverlapSignal(Signal):
    def score(self, source: Source, task: Task, selected: list[Source], **kwargs) -> float:
        if not task.symbols:
            return 0.0

        # Level 1: exact symbol match (function/class names)
        symbol_hits = len(source.symbols & task.symbols)

        # Level 2: content keyword match (symbol appears anywhere in file)
        content_lower = source.content.lower()
        content_hits = 0
        for sym in task.symbols:
            if sym not in source.symbols and sym.lower() in content_lower:
                content_hits += 1

        # Combine: symbol hits count full, content hits count half
        effective_hits = symbol_hits + (content_hits * 0.5)
        return min(1.0, effective_hits / len(task.symbols))
