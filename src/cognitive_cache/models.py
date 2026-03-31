"""Core data types for Cognitive Cache.

These types flow through the entire system:
- Source: a file in the repo (what we might put in the context window)
- Task: what the user wants done (the GitHub issue)
- ScoredSource: a Source with its computed value score
- SelectionResult: the final set of sources selected for the context window
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """A file in the repository that could be included in the context window."""

    path: str
    content: str
    token_count: int
    language: str
    symbols: frozenset[str] = field(default_factory=frozenset)

    def __hash__(self):
        return hash(self.path)


@dataclass(frozen=True)
class Task:
    """The task the LLM needs to perform (e.g., a GitHub issue)."""

    title: str
    body: str
    symbols: frozenset[str] = field(default_factory=frozenset)

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.body}"


@dataclass
class ScoredSource:
    """A Source with its computed value score and per-signal breakdown."""

    source: Source
    score: float
    signal_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """The final context selection: which sources, in what order, at what cost."""

    selected: list[ScoredSource]
    total_tokens: int
    budget: int

    @property
    def budget_remaining(self) -> int:
        return self.budget - self.total_tokens
