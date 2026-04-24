# cognitive-cache

Every LLM tool right now (Cursor, Claude Code, Copilot, all of them) decides what to put in the context window using heuristics: grep for some symbols, embed and cosine-similarity search, or just cram as many files as will fit. Nobody has an actual algorithm for this.

This project is an attempt to build one.

**No any LLM calls or API keys or cloud. It runs entirely local.**

![cognitive-cache finding the right files for a real GitHub issue](demo/demo.gif)

## the problem: context as an os-level resource

Think of it this way:

| Classic OS | LLM Equivalent | Current State |
|---|---|---|
| RAM | Context window | Manually managed |
| Virtual memory / page swaps | Context eviction + retrieval | Crude summarization |
| Process scheduler | Agent orchestration | Hand-coded loops |
| File system cache | Knowledge retrieval | Cosine similarity |
| Memory allocator | Token budget allocation | Nobody does this |

Early computers had programmers manually managing memory addresses. Then virtual memory was invented, a single algorithm, and it unlocked everything we know as modern computing.

**The LLM ecosystem is at the "manual memory management" stage right now.** Context is the single most important resource where reasoning happens, yet every tool manages it with heuristics instead of optimization.

Cognitive-cache is building the "virtual memory" for LLM reasoning.

## what it does

Given a task (like a GitHub issue) and a codebase, cognitive-cache picks which files to include in the LLM's context window. It combines multiple signals (symbol matching, dependency graph distance, git recency, semantic similarity, redundancy penalties) into a scoring function and runs greedy submodular optimization to select the highest-value set of files that fits within a token budget.

The key insight is treating context selection as a **constrained optimization problem** rather than a retrieval problem. RAG systems ask "what's most similar to the query?" but what you actually want is "what maximizes the chance the model gets this right?" Those are different questions.

## benchmark results

Benchmarked on **23 real bug-fix PRs** across 8 open-source repositories. For each issue, we run every strategy with a 12K token budget and measure **file recall**: did the algorithm select the files that were actually modified in the fix?

### overall performance

| Strategy | Avg Recall | Median | Head-to-head vs cognitive-cache |
|---|---|---|---|
| **cognitive-cache** | **34.7%** | **40%** | |
| llm-triage | 25.7% | 20% | 8W / 11T / 4L |
| embedding (RAG) | 25.9% | 0% | 9W / 9T / 5L |
| grep | 22.8% | 0% | 7W / 15T / 1L |
| random | 4.7% | 0% | 12W / 11T / 0L |
| full-stuff | 2.2% | 0% | 13W / 10T / 0L |

cognitive-cache **never loses to random or full-stuff** and has a winning record against every baseline, including llm-triage (asking the LLM to pick its own files), which is the hardest one to beat.

### per-repo breakdown

| Repository | cognitive-cache | llm-triage | embedding | grep | Issues |
|---|---|---|---|---|---|
| Textualize/rich | 100% | 0% | 50% | 100% | 1 |
| pallets/werkzeug | 62% | 38% | 44% | 44% | 4 |
| psf/requests | 50% | 25% | 25% | 25% | 2 |
| pallets/flask | 38% | 36% | 23% | 22% | 3 |
| pallets/click | 33% | 33% | 0% | 33% | 1 |
| pallets/jinja | 20% | 40% | 20% | 20% | 5 |
| fastify/fastify | 17% | 0% | 25% | 0% | 6 |
| tiangolo/fastapi | 0% | 50% | 0% | 0% | 1 |

### per-issue detail

Every issue, every strategy, full transparency:

| Issue | CC | LLM | Emb | Grep | Rand | Full |
|---|---|---|---|---|---|---|
| Textualize/rich#4006 | **100%** | 0% | 50% | 100% | 0% | 0% |
| fastify/fastify#6013 | 0% | 0% | 0% | 0% | 0% | 0% |
| fastify/fastify#6021 | **50%** | 0% | 0% | 0% | 0% | 0% |
| fastify/fastify#6026 | 0% | 0% | 0% | 0% | 0% | 0% |
| fastify/fastify#6030 | 0% | 0% | **50%** | 0% | 0% | 0% |
| fastify/fastify#6064 | 0% | 0% | 0% | 0% | 0% | 0% |
| fastify/fastify#6613 | 50% | 0% | **100%** | 0% | 0% | 0% |
| pallets/click#3225 | 33% | 33% | 0% | 33% | 33% | 0% |
| pallets/flask#5899 | 50% | 50% | 50% | 0% | 0% | 0% |
| pallets/flask#5917 | **40%** | 20% | 20% | 40% | 0% | 0% |
| pallets/flask#5928 | 25% | **38%** | 0% | 25% | 0% | 0% |
| pallets/jinja#1663 | 0% | **100%** | 50% | 50% | 0% | 0% |
| pallets/jinja#1665 | 0% | 0% | 0% | 0% | 0% | 0% |
| pallets/jinja#1706 | 0% | 0% | **50%** | 0% | 0% | 0% |
| pallets/jinja#1852 | 0% | **50%** | 0% | 0% | 0% | 0% |
| pallets/jinja#2061 | **100%** | 50% | 0% | 50% | 0% | 0% |
| pallets/werkzeug#3038 | 50% | 50% | **75%** | 25% | 25% | 0% |
| pallets/werkzeug#3078 | 50% | 50% | 50% | 50% | 0% | 0% |
| pallets/werkzeug#3080 | **50%** | 0% | 0% | 0% | 0% | 0% |
| pallets/werkzeug#3129 | **100%** | 50% | 50% | 100% | 0% | 0% |
| psf/requests#7308 | 50% | 50% | 0% | 0% | 0% | 50% |
| psf/requests#7309 | 50% | 0% | 50% | 50% | 50% | 0% |
| tiangolo/fastapi#15139 | 0% | **50%** | 0% | 0% | 0% | 0% |

Bold = best recall for that issue. All runs use Qwen 3.5 9B (Q4_K_M) via llama.cpp at zero API cost.

### what the baselines do

- **random**: picks files at random until the token budget is full
- **full-stuff**: crams files alphabetically until the budget is full, which is roughly what most tools approximate
- **embedding (RAG)**: TF-IDF cosine similarity (scikit-learn, up to 5K features) between the issue text and file contents. This uses TF-IDF rather than neural embeddings, so it's a simpler version of what most RAG tools do; real-world RAG with a proper embedding model would likely score somewhat higher
- **grep**: searches for symbols and identifiers mentioned in the issue
- **llm-triage**: gives the LLM (same Qwen 3.5 9B) the issue plus a full file listing and asks it to pick the most relevant files. This simulates what tools like Cursor and Claude Code do when they decide what to read, and is the hardest baseline to beat

## how it works

Six signals score each file:

- **symbol overlap**: does this file define or mention identifiers from the task? Strongest signal when it fires, so it gets the highest weight
- **graph distance**: how many imports away is this file from the ones the task mentions? Built with networkx from the dependency graph
- **change recency**: files changed recently in git tend to be more relevant, and more likely to contain the bug
- **redundancy penalty**: if you already selected a file with similar symbols, this one becomes less valuable, which prevents the algorithm from blowing the budget on five files from the same directory
- **embedding similarity**: TF-IDF cosine similarity, essentially what RAG does. Weighted low because we're trying to beat this, not replicate it
- **file role prior**: test files, type definitions, and config files get small baseline boosts since they tend to be informative

These get combined into a weighted score, and a greedy selector picks files one at a time, re-evaluating redundancy after each pick. If a file is too large to fit (like a 13K token app.py when your budget is 12K), it gets chunked to extract just the relevant functions.

## using it

### as a library

```python
from cognitive_cache import select_context_from_repo

result = select_context_from_repo(".", "fix the login redirect bug")
for item in result.selected:
    print(f"{item.source.path} (score: {item.score:.3f})")
```

For repeated queries against the same repo, build the index once and reuse it:

```python
from cognitive_cache import RepoIndex, select_context

index = RepoIndex.build(".")
r1 = select_context(index, "fix the login bug")
r2 = select_context(index, "add rate limiting to the API")
```

### as a CLI

```
uv sync
uv run cognitive-cache select --repo . --task "fix the login redirect bug"
uv run cognitive-cache select --repo . --task "fix login" --json          # machine-readable
uv run cognitive-cache select --repo . --task "fix login" --output ctx.txt # dump context to file
```

### as an MCP server (for Claude Code, Cursor, etc.)

**Claude Code** — registers the server at user scope (available in all projects):

```bash
claude mcp add --scope user cognitive-cache -- uvx --from "cognitive-cache[mcp]" cognitive-cache-mcp
```

**Cursor / Windsurf / other editors** — add to your MCP config file:

```json
{
  "mcpServers": {
    "cognitive-cache": {
      "command": "uvx",
      "args": ["--from", "cognitive-cache[mcp]", "cognitive-cache-mcp"]
    }
  }
}
```

#### telling the model when to use it

The tool is most useful when the files relevant to a task aren't obvious — cross-cutting bugs, unfamiliar parts of a large codebase, tasks that span multiple layers. For small repos or tasks where you already know which files to touch, it doesn't add much over grep.

Add this to your `CLAUDE.md` (or equivalent system prompt / rules file):

```markdown
## context selection

When a task spans multiple files or you're not sure where to look, call `select_context_tool`
from the `cognitive-cache` MCP server before reading files manually:

- `repo_path`: absolute path to the repo root
- `task`: specific description of the task — the more precise, the better the results
- `budget`: token budget for returned context (default 12000; raise for complex tasks)

The tool returns file contents directly. Use them instead of separate file reads.
Call it at the start of investigation; the index is cached so follow-up calls are fast.

Skip it when you already know which files to read.
```

A precise task description outperforms a vague one because symbol overlap and embedding similarity both depend on the exact words used. "users get 401 after OIDC callback when session token is present" will score better than "fix auth bug".

### as a GitHub Action

The included workflow (`.github/workflows/context-suggest.yml`) automatically comments on new issues with the most relevant files. It runs entirely in CI with no API keys required.

## running the benchmark

```
uv sync --dev
uv run pytest tests/  # 97 tests
```

To run the benchmark with a local llama.cpp server:

```
LLAMACPP_BASE_URL=http://localhost:8080 PYTHONPATH=. uv run python benchmark/run_local.py   # full 23-issue benchmark
PYTHONPATH=. uv run python benchmark/run_test.py                                            # quick 3-issue test
```

Configure the llama.cpp connection with environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLAMACPP_BASE_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLAMACPP_MODEL` | `Qwen3.5-9B-Q4_K_M` | Model name to request |

To expand the benchmark dataset (needs a GitHub token for API access):

```
GITHUB_TOKEN=ghp_xxx uv run python benchmark/curate_dataset.py
```

## project structure

```
src/cognitive_cache/
    api.py              # public API: RepoIndex, select_context
    cli.py              # CLI entry point
    mcp_server.py       # MCP server for Claude Code / Cursor
    models.py           # core types (Source, Task, ScoredSource, SelectionResult)
    indexer/             # turns a repo directory into a list of Source objects
    signals/             # the six scoring signals
    core/                # value function, greedy selector, file chunker
    baselines/           # the five baseline strategies we benchmark against
    llm/                 # adapters for calling LLMs (claude, openai, llama.cpp)
benchmark/
    dataset/             # curated github issues with known fixes
    runner.py            # orchestrates benchmark runs
    evaluator.py         # computes recall and efficiency metrics
```

## whats next

- Weight tuning on a larger dataset, since the signal weights are currently hand-tuned
- Improving the test-vs-source ranking for bug-fix tasks (the algorithm sometimes ranks test files above the source files that need fixing)
- Adaptive replanning that re-optimizes context mid-conversation after the model calls a tool or asks a followup
- Task-aware compression that goes beyond chunking to actually compress file content while preserving what's relevant

## why this matters

Context windows are getting bigger (1M tokens, soon more) but bigger doesn't mean the problem goes away. More capacity means more choices, and just stuffing everything in wastes compute while actively degrading output quality because the model's attention gets diluted. The optimization becomes more valuable as windows grow, not less.

If there's going to be one algorithm at the center of how LLM tools work, it should probably be this one. Or something like it.

## license

Apache 2.0. See [LICENSE](LICENSE).
