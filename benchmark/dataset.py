"""Benchmark dataset: curated GitHub issues with known fixes.

Each issue records which repo, the issue text, which files were modified
in the fix (ground truth), and commit SHAs to check out the right state.
Stored as JSON for easy inspection and editing.
"""

import json
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkIssue:
    repo: str
    repo_url: str
    issue_number: int
    title: str
    body: str
    fixed_files: list[str]
    fix_commit: str
    base_commit: str
    ground_truth_diff: str = ""


def save_dataset(issues: list[BenchmarkIssue], path: str):
    data = [asdict(issue) for issue in issues]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset(path: str) -> list[BenchmarkIssue]:
    with open(path) as f:
        data = json.load(f)
    issues = []
    for item in data:
        if "ground_truth_diff" not in item:
            item["ground_truth_diff"] = ""
        issues.append(BenchmarkIssue(**item))
    return issues
