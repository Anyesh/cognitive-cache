"""Context Orderer: arranges selected files for optimal LLM attention.

LLMs have attention biases:
- Primacy: they pay more attention to content at the start
- Recency: they pay more attention to content at the end
- Lost in the middle: content in the middle gets less attention

Our strategy:
1. Highest-scored files first (primacy — most important context up front)
2. Test files last (recency — expected behavior anchors the output)
3. Everything else in the middle, sorted by score
"""

from cognitive_cache.models import ScoredSource


def order_context(scored_sources: list[ScoredSource]) -> list[ScoredSource]:
    tests = [s for s in scored_sources if s.source.is_test]
    non_tests = [s for s in scored_sources if not s.source.is_test]

    non_tests.sort(key=lambda s: s.score, reverse=True)
    tests.sort(key=lambda s: s.score, reverse=True)

    return non_tests + tests
