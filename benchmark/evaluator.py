"""Benchmark evaluator: measures how well each strategy performed.

Three metrics:
1. File recall: did the strategy select the files that were actually modified?
2. Patch correctness: does the generated patch apply and pass tests? (manual/CI)
3. Token efficiency: correctness divided by tokens used.
"""


def compute_file_recall(selected_paths: set[str], actual_paths: set[str]) -> float:
    """What fraction of the actually-modified files were in the selected context?"""
    if not actual_paths:
        return 1.0
    hits = selected_paths & actual_paths
    return len(hits) / len(actual_paths)


def compute_token_efficiency(correct: bool, tokens_used: int) -> float:
    """Score that rewards correctness at lower token cost."""
    if not correct or tokens_used == 0:
        return 0.0
    return 1000.0 / tokens_used
