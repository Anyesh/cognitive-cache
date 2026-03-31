"""Quick test run: 3 issues only, to verify the full pipeline works."""

from cognitive_cache.llm.llamacpp_adapter import LlamaCppAdapter
from benchmark.dataset import load_dataset
from benchmark.runner import run_benchmark


def main():
    issues = load_dataset("benchmark/dataset/issues.json")[:3]
    print(f"Running test with {len(issues)} issues:")
    for i in issues:
        print(f"  {i.repo}#{i.issue_number} — {i.title[:50]}")

    adapters = {
        "qwen3.5-9b": LlamaCppAdapter(model="Qwen3.5-9B-Q4_K_M"),
    }

    results = run_benchmark(
        issues=issues,
        llm_adapters=adapters,
        budget=12000,
        output_dir="benchmark/results",
    )

    print(f"\nDone. {len(results)} runs completed.")
    print("Analyze: uv run python -m benchmark.analyze benchmark/results/<latest>.json")


if __name__ == "__main__":
    main()
