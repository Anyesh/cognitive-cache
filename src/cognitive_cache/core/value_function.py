"""The Value Function: combines all 6 signals into a single score.

This is the core intellectual property of Cognitive Cache.
It answers: "how valuable is this file for this task?"

The formula:
  V(s | T, S) = w1 * symbol_overlap
              + w2 * graph_distance
              + w3 * change_recency
              - w4 * redundancy          (subtracted — penalty)
              + w5 * embedding_sim
              + w6 * file_role_prior

Weights are configurable and will be tuned against benchmark results.
"""

from dataclasses import dataclass

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.symbol_overlap import SymbolOverlapSignal
from cognitive_cache.signals.graph_distance import GraphDistanceSignal
from cognitive_cache.signals.change_recency import ChangeRecencySignal
from cognitive_cache.signals.redundancy import RedundancySignal
from cognitive_cache.signals.embedding_sim import EmbeddingSimilaritySignal
from cognitive_cache.signals.file_role_prior import FileRolePriorSignal
from cognitive_cache.indexer.graph_builder import DependencyGraph


@dataclass
class WeightConfig:
    """Weights for each signal. These are the tunable knobs."""

    symbol_overlap: float = 0.35
    graph_distance: float = 0.20
    change_recency: float = 0.15
    redundancy: float = 0.20
    embedding_sim: float = 0.05
    file_role_prior: float = 0.05


class ValueFunction:
    """Combines all scoring signals into a single value score."""

    def __init__(
        self,
        weights: WeightConfig | None = None,
        graph: DependencyGraph | None = None,
        recency_data: dict[str, float] | None = None,
        embedding_signal: EmbeddingSimilaritySignal | None = None,
        entry_points: set[str] | None = None,
    ):
        self.weights = weights or WeightConfig()
        self._symbol = SymbolOverlapSignal()
        self._graph = GraphDistanceSignal(graph) if graph else None
        self._recency = ChangeRecencySignal(recency_data or {})
        self._redundancy = RedundancySignal()
        self._embedding = embedding_signal or EmbeddingSimilaritySignal()
        self._role = FileRolePriorSignal()
        self._entry_points = entry_points or set()

    def score(self, source: Source, task: Task, selected: list[Source]) -> float:
        """Compute the combined value score for a candidate source."""
        score, _ = self.score_with_breakdown(source, task, selected)
        return score

    def score_with_breakdown(
        self, source: Source, task: Task, selected: list[Source]
    ) -> tuple[float, dict[str, float]]:
        """Compute value score and return per-signal breakdown."""
        w = self.weights
        breakdown = {}

        breakdown["symbol_overlap"] = self._symbol.score(source, task, selected)

        if self._graph:
            breakdown["graph_distance"] = self._graph.score(
                source, task, selected, entry_points=self._entry_points
            )
        else:
            breakdown["graph_distance"] = 0.0

        breakdown["change_recency"] = self._recency.score(source, task, selected)
        breakdown["redundancy"] = self._redundancy.score(source, task, selected)
        breakdown["embedding_sim"] = self._embedding.score(source, task, selected)
        breakdown["file_role_prior"] = self._role.score(source, task, selected)

        combined = (
            w.symbol_overlap * breakdown["symbol_overlap"]
            + w.graph_distance * breakdown["graph_distance"]
            + w.change_recency * breakdown["change_recency"]
            - w.redundancy * breakdown["redundancy"]
            + w.embedding_sim * breakdown["embedding_sim"]
            + w.file_role_prior * breakdown["file_role_prior"]
        )

        return max(0.0, combined), breakdown
