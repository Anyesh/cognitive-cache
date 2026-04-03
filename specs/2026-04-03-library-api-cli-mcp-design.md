# Design: Library API, CLI, and MCP Server

**Date:** 2026-04-03
**Status:** Draft
**Approach:** B (RepoIndex with built-in caching)

## Problem

cognitive-cache has a working context selection algorithm that beats all baselines on a 23-issue benchmark, but no way for anything to actually call it. The only orchestration lives in `benchmark/runner.py`, tangled with benchmark-specific logic. There is no public API, no CLI, and no server. The project needs integration surfaces before it can find real-world adoption.

## Target Users

**Primary:** Developers building coding agents with the Claude API, OpenAI API, or local models. They control the context window directly, have no existing context selection solution, and need a `pip install` + three-line integration.

**Secondary:** Users of AI coding tools (Claude Code, Cursor) who add cognitive-cache as an MCP server for an explicit "which files matter for this task?" signal.

**Tertiary:** Tool builders (Cursor, Continue, Aider teams) who might integrate the library if public benchmarks demonstrate superiority over their current heuristics.

## Design

### 1. Library API (`src/cognitive_cache/api.py`)

Two public symbols: `RepoIndex` and `select_context`.

#### `RepoIndex`

Holds everything that is repo-dependent and task-independent:

- `sources: list[Source]` from `index_repo`
- `recency_data: dict[str, float]` from `GitAnalyzer.recency_scores`
- `graph: DependencyGraph` from `build_dependency_graph`
- `embedding_signal: EmbeddingSimilaritySignal` (fitted)
- `file_mtimes: dict[str, float]` for staleness detection
- `head_commit: str` from `git rev-parse HEAD`
- `repo_path: str`

**`RepoIndex.build(repo_path: str) -> RepoIndex`**

Class method. Full index from scratch. Calls `index_repo`, `GitAnalyzer.recency_scores`, `build_dependency_graph`, `EmbeddingSimilaritySignal.fit`. Stores the mtime for each indexed file and the current HEAD commit. This is the expensive call, expected to take 3 to 6 seconds on a typical repo (100 to 2000 source files).

**`RepoIndex.refresh() -> RepoIndex`**

Returns a new `RepoIndex` (does not mutate self). Compares stored mtimes against current disk state:

- Changed/added files: re-read, re-tokenize, update sources list
- Deleted files: remove from sources list
- If any files changed: rebuild graph, re-fit TF-IDF (both fast enough that incremental updates aren't worth the complexity)
- Check `git rev-parse HEAD` against stored `head_commit`. Only re-fetch git recency if HEAD moved. This avoids a 0.5 to 2 second subprocess call on every file save.

If nothing changed (no mtime differences, HEAD unchanged), returns `self` to avoid unnecessary copies. Otherwise returns a new instance so the MCP server can swap the reference atomically without race conditions on overlapping async requests.

#### `select_context`

```python
def select_context(
    index: RepoIndex,
    task: str | Task,
    budget: int = 12_000,
) -> SelectionResult:
```

Accepts either a plain string (symbols extracted automatically) or a `Task` object (for callers who want to provide their own symbols, e.g., from AST analysis).

When given a string, it:
1. Extracts task symbols by matching the text against all known symbols in the index (moved from `runner.py:_extract_task_symbols`)
2. Finds entry points (files whose symbols overlap with task symbols)
3. Builds a `ValueFunction` with the index's graph, recency data, embedding signal, and entry points
4. Runs `GreedySelector.select`
5. Orders results via `order_context` (primacy/recency-aware placement)

When given a `Task` object, it skips step 1 and uses the provided symbols directly.

#### Convenience wrapper

```python
def select_context_from_repo(
    repo_path: str,
    task: str,
    budget: int = 12_000,
) -> SelectionResult:
    """Build index + select in one call. No caching."""
    index = RepoIndex.build(repo_path)
    return select_context(index, task, budget)
```

#### Public exports (`__init__.py`)

```python
from cognitive_cache.api import RepoIndex, select_context, select_context_from_repo
from cognitive_cache.models import Source, Task, ScoredSource, SelectionResult
```

#### Symbol extraction location

`_extract_task_symbols` and `_find_entry_points` both move from `benchmark/runner.py` into `api.py` as private functions. They're only called from `select_context()` and don't warrant their own module. The benchmark runner will import from the API instead of having its own copies.

### 2. Performance Fixes (existing code)

#### Git log depth cap (`src/cognitive_cache/indexer/git_analyzer.py`)

Add `--since=6.months.ago` to the `git log` calls in both `recency_scores()` and `change_frequency()`. Six months is generous; recency scoring beyond that normalizes to near-zero anyway. This caps the worst case from minutes to seconds on repos with long history (100K+ commits).

#### Graph undirected copy cache (`src/cognitive_cache/indexer/graph_builder.py`)

Add a cached `_undirected` attribute to `DependencyGraph`. Lazily computed on first call to `shortest_distance`, reused on subsequent calls. Invalidated when `add_edge` or `add_file` is called (which only happens during index build/refresh, not during selection).

Currently `shortest_distance` calls `self._graph.to_undirected()` on every invocation. For a repo with 500 files and 10 entry points, that's 5,000 full graph copies per task. This fix eliminates all of them.

### 3. CLI (`src/cognitive_cache/cli.py`)

Argparse-based command-line interface.

#### Usage

```
cognitive-cache select --repo . --task "fix the login redirect bug" --budget 12000
```

#### Default output (human-readable)

```
src/auth/handler.py          0.482  [sym=0.8 graph=0.5 recency=0.3 embed=0.2 role=0.1 redund=0.0]
src/auth/middleware.py        0.331  [sym=0.4 graph=1.0 recency=0.5 embed=0.3 role=0.1 redund=0.2]
tests/test_auth.py            0.215  [sym=0.2 graph=0.3 recency=0.8 embed=0.1 role=0.3 redund=0.0]

3 files selected | 4,821 / 12,000 tokens used
```

#### Flags

- `--repo PATH` (required): path to the repository root
- `--task TEXT` (required): plain text task description
- `--budget N` (optional, default 12000): token budget
- `--json` (optional): output as JSON instead of human-readable table
- `--output PATH` (optional): dump the full ordered context (file contents, ready to paste into a prompt) to a file

#### Entry point

```toml
[project.scripts]
cognitive-cache = "cognitive_cache.cli:main"
```

### 4. MCP Server (`src/cognitive_cache/mcp_server.py`)

Single-tool MCP server using the official `mcp` Python SDK.

#### Tool: `select_context`

**Parameters:**
- `repo_path` (string, required): absolute path to the repository root
- `task` (string, required): plain text task description
- `budget` (integer, optional, default 12000): token budget

**Returns:** JSON object with:
- `files`: array of `{path, score, signals: {symbol_overlap, graph_distance, change_recency, redundancy, embedding_sim, file_role_prior}, content}`
- `total_tokens`: total tokens across all selected files
- `budget`: the budget that was requested
- `budget_remaining`: unused tokens

#### Index caching

The server holds a `dict[str, RepoIndex]` in memory, keyed by `repo_path`.

On each tool call:
1. If no index exists for `repo_path`: call `RepoIndex.build()` (3 to 6 seconds, first call only)
2. If an index exists: call `refresh()` which checks HEAD + mtimes (fast, typically < 100ms)
3. Store the (possibly new) index back in the dict
4. Run `select_context()` with the index and task

#### Dependency

Added as an optional dependency group so the core library stays lightweight:

```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0.0"]
benchmark = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
]
```

#### Entry point

```toml
[project.scripts]
cognitive-cache = "cognitive_cache.cli:main"
cognitive-cache-mcp = "cognitive_cache.mcp_server:main"
```

#### Claude Code configuration

Users add to their MCP settings:

```json
{
  "mcpServers": {
    "cognitive-cache": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "cognitive-cache-mcp"]
    }
  }
}
```

### 5. Patch Evaluation (benchmark extension)

#### Motivation

The benchmark currently measures file recall (did we pick the right files?) but not whether better file selection leads to better LLM output. The runner already generates patches for every strategy+issue combination and saves them to JSON. We need to close the loop by evaluating those patches.

#### `compute_patch_similarity` (`benchmark/evaluator.py`)

New function alongside the existing `compute_file_recall`:

```python
def compute_patch_similarity(generated_patch: str, ground_truth_diff: str) -> float:
```

Compares the generated patch against the ground-truth PR diff using `difflib.SequenceMatcher`. Returns a similarity score in [0, 1]. This is a continuous metric rather than binary correct/incorrect, which gives us a gradient to optimize against.

The ground truth diff needs to be stored in the benchmark dataset. `BenchmarkIssue` gets a new field `ground_truth_diff: str` populated by `curate_dataset.py` (which already has GitHub API access and can fetch the PR diff).

#### `benchmark/analyze_patches.py`

New script that loads benchmark results JSON, computes patch similarity per strategy, and produces a summary table:

```
Strategy        Avg Patch Sim   Avg File Recall
cognitive-cache 0.42            34.7%
llm-triage      0.38            25.7%
embedding       0.31            25.9%
grep            0.28            22.8%
```

This is the evidence that either validates the project's core assumption (better file selection leads to better output) or tells us we need to go back and rethink.

No new dependencies; `difflib` is in the standard library.

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `src/cognitive_cache/api.py` | Create | RepoIndex, select_context, select_context_from_repo |
| `src/cognitive_cache/cli.py` | Create | CLI entry point |
| `src/cognitive_cache/mcp_server.py` | Create | MCP server |
| `src/cognitive_cache/__init__.py` | Edit | Add public exports |
| `src/cognitive_cache/indexer/git_analyzer.py` | Edit | Cap git log at 6 months |
| `src/cognitive_cache/indexer/graph_builder.py` | Edit | Cache undirected graph |
| `benchmark/evaluator.py` | Edit | Add compute_patch_similarity |
| `benchmark/dataset.py` | Edit | Add ground_truth_diff field |
| `benchmark/analyze_patches.py` | Create | Patch similarity analysis script |
| `benchmark/runner.py` | Edit | Import _extract_task_symbols from api instead of local copy |
| `pyproject.toml` | Edit | Add entry points, mcp dependency group |

## Error Handling

Errors at system boundaries are handled gracefully, not with crashes:

- **`repo_path` doesn't exist:** `RepoIndex.build()` raises `FileNotFoundError`. The CLI prints a clear message and exits with code 1. The MCP server returns an error response.
- **Not a git repo:** `GitAnalyzer` already returns empty dicts when git commands fail. The recency and graph distance signals simply contribute zero. No error, just degraded scoring.
- **Zero source files after filtering:** `select_context` returns an empty `SelectionResult` (no files selected, zero tokens used). The CLI prints "No source files found" and exits cleanly.
- **Vague task description (no symbol matches):** `select_context` still runs; it falls back to embedding similarity, recency, and file role prior. The CLI prints a warning: "No symbol matches found, results may be less precise."

## What's Explicitly Out of Scope

- Disk-based index persistence (the `RepoIndex` is in-memory only for v1)
- Language support beyond Python, JS, TS
- File watcher / automatic re-indexing
- Neural embeddings (TF-IDF stays)
- Weight tuning (hand-tuned weights stay)
- Adaptive replanning mid-conversation
