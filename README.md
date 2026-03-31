# cognitive-cache

Every LLM tool right now -- Cursor, Claude Code, Copilot, all of them -- decides what to put in the context window using heuristics. Grep for some symbols, embed and cosine-similarity search, or just cram as many files as will fit. Nobody has an actual algorithm for this.

This project is an attempt to build one.

## what it does

Given a task (like a github issue) and a codebase, cognitive-cache picks which files to include in the LLM's context window. It uses a scoring function with multiple signals -- symbol matching, dependency graph distance, git recency, semantic similarity, redundancy penalties -- and runs greedy submodular optimization to select the highest-value set of files that fits within a token budget.

The key insight is treating context selection as a constrained optimization problem rather than a retrieval problem. RAG systems ask "whats most similar to the query?" but what you actually want is "what maximizes the chance the model gets this right?" Those are different questions.

## does it work?

Early benchmarks on real github issues with known fixes (we check if the algorithm selects the files that were actually modified in the fix):

| strategy | avg file recall | what it does |
|---|---|---|
| cognitive-cache | ~40-50% | our algorithm |
| llm-triage | ~27% | ask the LLM to pick its own files |
| embedding/RAG | ~24% | cosine similarity, what most tools do |
| grep | ~21-35% | search for mentioned symbols |
| random | ~5% | random files up to budget |
| full-stuff | ~7% | cram everything alphabetically |

Benchmarked against 23 bug-fix PRs across Flask, Requests, FastAPI, Jinja, Werkzeug, Click, Rich, and Fastify. Using a local Qwen 3.5 9B model via llama.cpp so the whole thing runs at zero cost.

The numbers are still moving as we tune, but the important thing is that its consistently beating the baselines including the "let the LLM pick its own context" approach which is the hardest one to beat.

## how it works

Six signals score each file:

- **symbol overlap** -- does this file define or mention identifiers from the task? This is the strongest signal when it fires, weighted highest
- **graph distance** -- how many imports away is this file from the files the task mentions? Uses networkx to build the dependency graph
- **change recency** -- files changed recently in git are more likely to be relevant (and more likely to contain the bug)
- **redundancy penalty** -- if you already selected a file with similar symbols, this one is less valuable. Prevents the algorithm from blowing the budget on 5 files from the same directory
- **embedding similarity** -- TF-IDF cosine similarity, basically what RAG does. Weighted low because were trying to beat this, not replicate it
- **file role prior** -- test files, type definitions, and config files get small baseline boosts since they tend to be informative

These get combined into a weighted score, and then a greedy selector picks files by score, re-evaluating redundancy after each pick. For files that are too large to fit (like a 13K token app.py when your budget is 12K), it chunks them to extract just the relevant functions.

## running it

```
uv sync --dev
uv run pytest tests/  # 56 tests
```

To run the benchmark with a local llama.cpp server:

```
PYTHONPATH=. uv run python benchmark/run_test.py    # quick 3-issue test
PYTHONPATH=. uv run python benchmark/run_local.py   # full 23-issue benchmark
```

Expects a llama.cpp server on localhost:8081 with an OpenAI-compatible API. If yours is on a different port just edit `benchmark/run_local.py`.

To expand the benchmark dataset (needs a github token for API access):

```
GITHUB_TOKEN=ghp_xxx uv run python benchmark/curate_dataset.py
```

## project structure

```
src/cognitive_cache/
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

The algorithm works but theres a lot of room to improve. Some things were thinking about:

- weight tuning on a larger dataset (right now the signal weights are hand-tuned)
- adaptive replanning -- re-optimize context mid-conversation after the model calls a tool or asks a followup
- task-aware compression beyond chunking, actually compressing file content to preserve whats relevant
- packaging this as a library other tools can integrate

## why this matters

Context windows are getting bigger (1M tokens, soon more) but bigger doesn't mean the problem goes away. More capacity means more choices and just stuffing everything in wastes compute and actually degrades output quality. The models attention gets diluted. So the optimization becomes more valuable as windows grow, not less.

If theres going to be one algorithm at the center of how LLM tools work, it should probably be this one. Or something like it.
