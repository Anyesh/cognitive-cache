"""CLI entry point for cognitive-cache.

Usage:
    cognitive-cache select --repo . --task "fix the login bug" --budget 12000
"""

import argparse
import json
import sys

from cognitive_cache.api import RepoIndex, select_context


def _format_human_readable(result) -> str:
    lines = []
    for ss in result.selected:
        path = ss.source.path
        score = ss.score
        signals = ss.signal_scores

        sig_parts = []
        for key in [
            "symbol_overlap",
            "graph_distance",
            "change_recency",
            "embedding_sim",
            "file_role_prior",
            "redundancy",
        ]:
            val = signals.get(key, 0.0)
            short = {
                "symbol_overlap": "sym",
                "graph_distance": "graph",
                "change_recency": "recency",
                "embedding_sim": "embed",
                "file_role_prior": "role",
                "redundancy": "redund",
            }[key]
            sig_parts.append(f"{short}={val:.1f}")

        sig_str = " ".join(sig_parts)
        lines.append(f"{path:<45} {score:.3f}  [{sig_str}]")

    lines.append("")
    lines.append(
        f"{len(result.selected)} files selected | "
        f"{result.total_tokens:,} / {result.budget:,} tokens used"
    )
    return "\n".join(lines)


def _format_json(result) -> str:
    data = {
        "files": [
            {
                "path": ss.source.path,
                "score": round(ss.score, 4),
                "signals": {k: round(v, 4) for k, v in ss.signal_scores.items()},
                "token_count": ss.source.token_count,
            }
            for ss in result.selected
        ],
        "total_tokens": result.total_tokens,
        "budget": result.budget,
        "budget_remaining": result.budget_remaining,
    }
    return json.dumps(data, indent=2)


def _write_context_file(result, output_path: str):
    with open(output_path, "w") as f:
        for ss in result.selected:
            f.write(f"# --- {ss.source.path} ---\n")
            f.write(ss.source.content)
            f.write("\n\n")


def main():
    parser = argparse.ArgumentParser(
        prog="cognitive-cache",
        description="Optimal context selection for LLMs",
    )
    subparsers = parser.add_subparsers(dest="command")

    select_parser = subparsers.add_parser(
        "select", help="Select context files for a task"
    )
    select_parser.add_argument(
        "--repo", required=True, help="Path to the repository root"
    )
    select_parser.add_argument("--task", required=True, help="Task description")
    select_parser.add_argument(
        "--budget", type=int, default=12000, help="Token budget (default: 12000)"
    )
    select_parser.add_argument(
        "--max-files",
        type=int,
        default=15,
        help="Maximum files to return (default: 15)",
    )
    select_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score threshold (default: 0.0)",
    )
    select_parser.add_argument(
        "--include-tests",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Include test files: auto (detect from task), yes, no (default: auto)",
    )
    select_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Output as JSON"
    )
    select_parser.add_argument("--output", help="Write full context to this file")

    args = parser.parse_args()

    if args.command != "select":
        parser.print_help()
        sys.exit(1)

    try:
        index = RepoIndex.build(args.repo)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not index.sources:
        print("No source files found.", file=sys.stderr)
        sys.exit(1)

    include_tests_map = {"auto": None, "yes": True, "no": False}
    include_tests = include_tests_map[args.include_tests]

    result = select_context(
        index,
        args.task,
        budget=args.budget,
        include_tests=include_tests,
        max_files=args.max_files,
        min_score=args.min_score,
    )

    if not result.selected:
        print("No files selected.", file=sys.stderr)
        sys.exit(0)

    # Warn if no symbols matched (vague task description)
    if all(
        ss.signal_scores.get("symbol_overlap", 0.0) == 0.0 for ss in result.selected
    ):
        print(
            "Warning: No symbol matches found, results may be less precise.",
            file=sys.stderr,
        )

    if args.json_output:
        print(_format_json(result))
    else:
        print(_format_human_readable(result))

    if args.output:
        _write_context_file(result, args.output)


if __name__ == "__main__":
    main()
