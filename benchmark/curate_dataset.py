"""Dataset curation script: finds good benchmark issues from GitHub.

Searches for closed bug-fix PRs in well-chosen repos, extracts the
files changed, and filters to non-trivial fixes (2+ source files).

Usage:
  # Without token (60 req/hour limit):
  uv run python benchmark/curate_dataset.py

  # With token (5000 req/hour):
  GITHUB_TOKEN=ghp_xxx uv run python benchmark/curate_dataset.py

Output: benchmark/dataset/issues.json
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

# Repos selected for:
# - Good test suites (verifiable fixes)
# - Moderate size (500-5000 source files)
# - Active issue trackers with bug labels
# - Python or JavaScript/TypeScript
CANDIDATE_REPOS = [
    # Python repos
    "pallets/flask",
    "encode/httpx",
    "psf/requests",
    "pallets/click",
    "Textualize/rich",
    # JS/TS repos
    "expressjs/express",
    "fastify/fastify",
]

SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx"}


def _github_get(endpoint: str, token: str | None = None) -> dict | list | None:
    """Make a GitHub API request."""
    url = f"https://api.github.com{endpoint}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"  Rate limited. Waiting 60s...")
            time.sleep(60)
            return _github_get(endpoint, token)
        print(f"  HTTP {e.code} for {endpoint}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def _is_source_file(path: str) -> bool:
    return any(path.endswith(ext) for ext in SOURCE_EXTENSIONS)


def find_bug_fix_prs(repo: str, token: str | None = None, max_prs: int = 10) -> list[dict]:
    """Find closed PRs that fix bugs in a repo."""
    # Search for merged PRs with "fix" or "bug" in title
    results = []

    # Strategy: search for recently merged PRs, filter for bug fixes
    prs = _github_get(
        f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc&per_page=50",
        token,
    )
    if not prs:
        return []

    for pr in prs:
        if not pr.get("merged_at"):
            continue

        title = pr.get("title", "").lower()
        # Filter for bug-fix-like PRs
        if not any(kw in title for kw in ["fix", "bug", "crash", "error", "broken", "issue"]):
            continue

        # Get files changed in this PR
        files = _github_get(f"/repos/{repo}/pulls/{pr['number']}/files", token)
        if not files:
            continue

        source_files = [f["filename"] for f in files if _is_source_file(f["filename"])]

        # We want non-trivial fixes: 2+ source files changed
        if len(source_files) < 2:
            continue

        # Get the base and merge commits
        base_sha = pr.get("base", {}).get("sha", "")
        merge_sha = pr.get("merge_commit_sha", "")

        # Try to find linked issue
        body = pr.get("body", "") or ""
        issue_number = pr["number"]  # Use PR number as issue reference

        results.append({
            "repo": repo,
            "repo_url": f"https://github.com/{repo}.git",
            "issue_number": pr["number"],
            "title": pr["title"],
            "body": body[:500],  # Truncate long bodies
            "fixed_files": source_files,
            "fix_commit": merge_sha,
            "base_commit": base_sha,
        })

        if len(results) >= max_prs:
            break

    return results


def curate_dataset(token: str | None = None) -> list[dict]:
    """Build the benchmark dataset from GitHub."""
    all_issues = []

    for repo in CANDIDATE_REPOS:
        print(f"\nSearching {repo}...")
        issues = find_bug_fix_prs(repo, token, max_prs=8)
        print(f"  Found {len(issues)} qualifying PRs")
        all_issues.extend(issues)

        # Check rate limit
        rate = _github_get("/rate_limit", token)
        if rate:
            remaining = rate.get("rate", {}).get("remaining", 0)
            print(f"  API calls remaining: {remaining}")
            if remaining < 5:
                print("  Low on API calls, stopping.")
                break

    return all_issues


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        print("Using authenticated requests (5000/hour)")
    else:
        print("No GITHUB_TOKEN set — using unauthenticated requests (60/hour)")
        print("Set GITHUB_TOKEN=ghp_xxx for better results\n")

    issues = curate_dataset(token)

    # Save dataset
    os.makedirs("benchmark/dataset", exist_ok=True)
    output_path = "benchmark/dataset/issues.json"
    with open(output_path, "w") as f:
        json.dump(issues, f, indent=2)

    print(f"\nSaved {len(issues)} issues to {output_path}")
    print(f"\nBreakdown by repo:")
    repos = {}
    for issue in issues:
        repos[issue["repo"]] = repos.get(issue["repo"], 0) + 1
    for repo, count in sorted(repos.items()):
        print(f"  {repo}: {count}")


if __name__ == "__main__":
    main()
