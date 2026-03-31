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


def _is_test_file(path: str) -> bool:
    return ("test" in path.lower() and
            any(path.endswith(ext) for ext in (".py", ".js", ".ts", ".jsx", ".tsx")))


def order_context(scored_sources: list[ScoredSource]) -> list[ScoredSource]:
    """Order selected sources for optimal LLM context placement."""
    tests = [s for s in scored_sources if _is_test_file(s.source.path)]
    non_tests = [s for s in scored_sources if not _is_test_file(s.source.path)]

    non_tests.sort(key=lambda s: s.score, reverse=True)
    tests.sort(key=lambda s: s.score, reverse=True)

    return non_tests + tests
