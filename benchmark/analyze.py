"""Analyze benchmark results and print a summary table.

Run: uv run python -m benchmark.analyze benchmark/results/<file>.json
"""

import json
import sys
from collections import defaultdict


def analyze(results_path: str):
    with open(results_path) as f:
        results = json.load(f)

    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    print(f"\n{'Strategy':<20} {'Avg Recall':>12} {'Avg Tokens':>12} {'N Runs':>8}")
    print("-" * 56)

    for strategy in ["random", "full_stuff", "embedding", "grep", "llm_triage", "cognitive_cache"]:
        runs = by_strategy.get(strategy, [])
        if not runs:
            continue
        avg_recall = sum(r["file_recall"] for r in runs) / len(runs)
        avg_tokens = sum(r["tokens_used"] for r in runs) / len(runs)
        print(f"{strategy:<20} {avg_recall:>11.1%} {avg_tokens:>11.0f} {len(runs):>8}")

    print(f"\n{'Strategy':<20} {'Model':<15} {'Avg Recall':>12}")
    print("-" * 50)
    for strategy in ["random", "full_stuff", "embedding", "grep", "llm_triage", "cognitive_cache"]:
        runs = by_strategy.get(strategy, [])
        by_model = defaultdict(list)
        for r in runs:
            by_model[r["model"]].append(r)
        for model, model_runs in sorted(by_model.items()):
            avg_recall = sum(r["file_recall"] for r in model_runs) / len(model_runs)
            print(f"{strategy:<20} {model:<15} {avg_recall:>11.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python -m benchmark.analyze <results.json>")
        sys.exit(1)
    analyze(sys.argv[1])
