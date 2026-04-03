"""MCP server for cognitive-cache.

Exposes a single tool: select_context. Caches RepoIndex instances
in memory so repeated calls to the same repo are fast.

Run with:
    cognitive-cache-mcp
    uv run --extra mcp cognitive-cache-mcp
"""

import json
import os

from cognitive_cache.api import RepoIndex, select_context

try:
    from mcp.server.fastmcp import FastMCP as _FastMCP  # noqa: PLC0415

    _MCP_AVAILABLE = True
except ImportError:
    _FastMCP = None
    _MCP_AVAILABLE = False

# In-memory cache of repo indices, keyed by absolute repo path
_index_cache: dict[str, RepoIndex] = {}


def _handle_select_context(repo_path: str, task: str, budget: int = 12000) -> dict:
    repo_path = os.path.abspath(repo_path)

    if repo_path in _index_cache:
        index = _index_cache[repo_path].refresh()
    else:
        index = RepoIndex.build(repo_path)

    _index_cache[repo_path] = index

    result = select_context(index, task, budget=budget)

    return {
        "files": [
            {
                "path": ss.source.path,
                "score": round(ss.score, 4),
                "signals": {k: round(v, 4) for k, v in ss.signal_scores.items()},
                "content": ss.source.content,
                "token_count": ss.source.token_count,
            }
            for ss in result.selected
        ],
        "total_tokens": result.total_tokens,
        "budget": result.budget,
        "budget_remaining": result.budget_remaining,
    }


def main():
    if not _MCP_AVAILABLE:
        print(
            "MCP SDK not installed. Install with: uv sync --extra mcp",
            flush=True,
        )
        raise SystemExit(1)

    mcp = _FastMCP(
        "cognitive-cache",
        description="Optimal context selection for LLMs. Given a codebase and a task, "
        "picks the files most likely to help the model get the answer right.",
    )

    @mcp.tool()
    def select_context_tool(
        repo_path: str,
        task: str,
        budget: int = 12000,
    ) -> str:
        """Select the optimal set of files for an LLM's context window.

        Given a repository path and a task description, analyzes the codebase
        using six signals (symbol overlap, dependency graph, git recency,
        redundancy penalty, embedding similarity, file role) and returns the
        highest-value files that fit within the token budget.

        Args:
            repo_path: Absolute path to the repository root.
            task: Plain text description of what the LLM needs to do.
            budget: Maximum token budget for selected context (default 12000).

        Returns:
            JSON with ranked files (path, score, signals, content),
            total tokens, budget, and budget remaining.
        """
        result = _handle_select_context(repo_path, task, budget)
        return json.dumps(result, indent=2)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
