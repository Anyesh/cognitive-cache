"""Run the benchmark using local llama.cpp models.

Usage:
  uv run python benchmark/run_local.py

Uses models running on localhost:8081 — zero API cost.
"""

from cognitive_cache.llm.llamacpp_adapter import LlamaCppAdapter
from benchmark.dataset import load_dataset
from benchmark.runner import run_benchmark


def main():
    # Load dataset
    issues = load_dataset("benchmark/dataset/issues.json")
    print(f"Loaded {len(issues)} issues")

    # Set up local model adapters
    adapters = {
        "qwen3.5-9b": LlamaCppAdapter(),
    }

    # Run benchmark with 12K token budget
    results = run_benchmark(
        issues=issues,
        llm_adapters=adapters,
        budget=12000,
        output_dir="benchmark/results",
    )

    print(f"\nTotal runs: {len(results)}")
    print(
        "Analyze with: uv run python -m benchmark.analyze benchmark/results/<file>.json"
    )


if __name__ == "__main__":
    main()
