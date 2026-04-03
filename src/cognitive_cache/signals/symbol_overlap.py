"""Signal 1: Symbol Overlap (core-word tier scoring).

Scores a file by how well it covers the core concepts in the task.

Extracts "core words" from the task text and scores each word by
the quality of its best match in the file's defined symbols:
  - Exact match (symbol == word): 1.0
  - Contains match (word in symbol): 0.4
  - Content match (word in file text): 0.15
  - No match: 0.0

This prevents test files (which define many compound symbols like
test_login_success) from outscoring source files that define the
actual symbol (login) the task is about.
"""

import re

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal

_STOP_WORDS = {
    "should",
    "would",
    "could",
    "their",
    "there",
    "these",
    "those",
    "where",
    "while",
    "being",
    "about",
    "which",
    "doesn",
    "after",
    "before",
    "called",
    "every",
    "first",
    "other",
    "still",
    "since",
    "using",
    "when",
    "then",
    "than",
    "that",
    "this",
    "with",
    "from",
    "into",
    "have",
    "been",
    "were",
    "also",
    "some",
    "each",
    "what",
    "will",
    "more",
    "very",
    "just",
    "like",
    "only",
    "most",
    "such",
}

_TIER_EXACT = 1.0
_TIER_CONTAINS = 0.4
_TIER_CONTENT = 0.15


def _extract_core_words(text: str) -> list[str]:
    words = set(re.findall(r"\b[a-z_][a-z0-9_]{3,}\b", text.lower()))
    return sorted(words - _STOP_WORDS)


class SymbolOverlapSignal(Signal):
    def score(
        self, source: Source, task: Task, selected: list[Source], **kwargs
    ) -> float:
        core_words = _extract_core_words(task.full_text)

        if len(core_words) < 2 and task.symbols:
            extra = {s.lower() for s in task.symbols if len(s) >= 4}
            core_words = sorted(set(core_words) | extra)

        if not core_words:
            return 0.0

        symbols_lower = {s.lower() for s in source.symbols}
        content_lower = source.content.lower()

        total = 0.0
        for word in core_words:
            best = 0.0
            for sym in symbols_lower:
                if sym == word:
                    best = _TIER_EXACT
                    break
                if word in sym and best < _TIER_CONTAINS:
                    best = _TIER_CONTAINS
            if best == 0.0 and word in content_lower:
                best = _TIER_CONTENT
            total += best

        return min(1.0, total / len(core_words))
