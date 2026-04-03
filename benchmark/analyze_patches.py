"""Analyze benchmark results: compute patch similarity per strategy.

Usage:
    PYTHONPATH=. uv run python benchmark/analyze_patches.py benchmark/results/results_YYYYMMDD_HHMMSS.json
"""

import json
import sys
from collections import defaultdict

from benchmark.evaluator import compute_patch_similarity


def analyze(results_path: str):
    with open(results_path) as f:
        results = json.load(f)

    strategy_recalls: dict[str, list[float]] = defaultdict(list)

    for r in results:
        strategy = r["strategy"]
        strategy_recalls[strategy].append(r["file_recall"])

    print(f"{'Strategy':<20} {'Avg File Recall':>15} {'Issues':>8}")
    print("-" * 45)

    for strategy in sorted(strategy_recalls.keys()):
        recalls = strategy_recalls[strategy]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"{strategy:<20} {avg_recall:>14.1%} {len(recalls):>8}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results.json>")
        sys.exit(1)
    analyze(sys.argv[1])


if __name__ == "__main__":
    main()
