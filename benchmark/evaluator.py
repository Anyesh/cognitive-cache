"""Benchmark evaluator: measures how well each strategy performed.

Three metrics:
1. File recall: did the strategy select the files that were actually modified?
2. Patch correctness: does the generated patch apply and pass tests? (manual/CI)
3. Token efficiency: correctness divided by tokens used.
"""

import difflib


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


def compute_patch_similarity(generated_patch: str, ground_truth_diff: str) -> float:
    """Compute similarity between a generated patch and the ground truth diff.

    Uses difflib.SequenceMatcher on the changed lines (lines starting with
    + or -) to produce a [0, 1] score. Returns 0.0 for empty or error patches.
    """
    if not generated_patch or not ground_truth_diff:
        return 0.0

    if generated_patch.startswith("ERROR:"):
        return 0.0

    def _extract_changed_lines(patch: str) -> str:
        lines = []
        for line in patch.splitlines():
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                lines.append(line)
        return "\n".join(lines)

    gen_changes = _extract_changed_lines(generated_patch)
    truth_changes = _extract_changed_lines(ground_truth_diff)

    if not gen_changes or not truth_changes:
        return 0.0

    return difflib.SequenceMatcher(None, gen_changes, truth_changes).ratio()
