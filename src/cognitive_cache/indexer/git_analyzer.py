"""Git history analyzer for change recency scoring.

Extracts two signals from git history:
1. Recency: when was each file last modified? (normalized to [0, 1])
2. Frequency: how many commits touched each file? (normalized to [0, 1])

Uses subprocess to call git directly — simple, no dependencies,
and works with any git version.
"""

import subprocess
from collections import Counter


class GitAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _run_git(self, args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def is_shallow(self) -> bool:
        output = self._run_git(["rev-parse", "--is-shallow-repository"])
        if output is None:
            return False
        return output.strip() == "true"

    def recency_scores(self) -> dict[str, float]:
        """Score each file by how recently it was modified.

        The most recently modified file gets 1.0, the oldest gets a score
        approaching 0.0. Files not in git history are not included.
        """
        output = self._run_git(
            [
                "log",
                "--pretty=format:%H",
                "--name-only",
                "--diff-filter=ACMR",
                "--since=6.months.ago",
            ]
        )
        if not output:
            return {}

        file_order = {}
        order_counter = 0
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                order_counter += 1
                continue
            path = line.replace("\\", "/")
            if path not in file_order:
                file_order[path] = order_counter

        if not file_order:
            return {}

        max_order = max(file_order.values())
        if max_order <= 1:
            return {path: 1.0 for path in file_order}

        return {
            path: 1.0 - ((order - 1) / (max_order - 1))
            for path, order in file_order.items()
        }

    def change_frequency(self) -> dict[str, float]:
        """Score each file by how frequently it's been modified.

        More commits = higher score. Normalized to [0, 1].
        """
        output = self._run_git(
            [
                "log",
                "--pretty=format:",
                "--name-only",
                "--diff-filter=ACMR",
                "--since=6.months.ago",
            ]
        )
        if not output:
            return {}

        counts = Counter()
        for line in output.strip().split("\n"):
            line = line.strip()
            if line:
                counts[line.replace("\\", "/")] += 1

        if not counts:
            return {}

        max_count = max(counts.values())
        if max_count <= 1:
            return {path: 1.0 for path in counts}

        return {path: count / max_count for path, count in counts.items()}
