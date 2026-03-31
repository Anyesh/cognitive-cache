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
    """Analyzes git history for a repository."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _run_git(self, args: list[str]) -> str | None:
        """Run a git command and return stdout, or None if it fails."""
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

    def recency_scores(self) -> dict[str, float]:
        """Score each file by how recently it was modified.

        The most recently modified file gets 1.0, the oldest gets a score
        approaching 0.0. Files not in git history are not included.

        Returns:
            Dict mapping relative file path to recency score in [0, 1].
        """
        # git log: one line per commit per file, ordered newest-first
        output = self._run_git([
            "log", "--pretty=format:%H", "--name-only", "--diff-filter=ACMR",
        ])
        if not output:
            return {}

        # Parse: track the order each file first appears (= its most recent commit)
        file_order = {}
        order_counter = 0
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Skip commit hashes (40 hex chars)
            if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                order_counter += 1
                continue
            # It's a file path
            path = line.replace("\\", "/")
            if path not in file_order:
                file_order[path] = order_counter

        if not file_order:
            return {}

        # Normalize: file that appeared first (order=1) gets highest score
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

        Returns:
            Dict mapping relative file path to frequency score in [0, 1].
        """
        output = self._run_git([
            "log", "--pretty=format:", "--name-only", "--diff-filter=ACMR",
        ])
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

        return {
            path: count / max_count
            for path, count in counts.items()
        }
