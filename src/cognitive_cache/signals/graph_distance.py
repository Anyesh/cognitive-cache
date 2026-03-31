"""Signal 2: Dependency Graph Distance.

Scores a file by how close it is (in the import graph) to the task's entry points.
Entry points are files that contain task-mentioned symbols.

The score decays exponentially with distance:
  distance 0 (the file IS an entry point) -> 1.0
  distance 1 (direct import) -> 0.5
  distance 2 -> 0.25
  no connection -> 0.0

Why exponential decay: each hop in the import graph makes a file less likely
to be directly relevant. But even 2-3 hops away can matter (e.g., the database
layer called by the service layer called by the handler with the bug).
"""

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal
from cognitive_cache.indexer.graph_builder import DependencyGraph


class GraphDistanceSignal(Signal):
    def __init__(self, graph: DependencyGraph):
        self._graph = graph

    def score(self, source: Source, task: Task, selected: list[Source], **kwargs) -> float:
        entry_points: set[str] = kwargs.get("entry_points", set())
        if not entry_points:
            return 0.0

        # Find minimum distance to any entry point
        min_distance = float("inf")
        for ep in entry_points:
            d = self._graph.shortest_distance(ep, source.path)
            min_distance = min(min_distance, d)

        if min_distance == float("inf"):
            return 0.0

        # Exponential decay: 0.5^distance
        return 0.5 ** min_distance
