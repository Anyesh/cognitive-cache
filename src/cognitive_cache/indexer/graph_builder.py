"""Dependency graph builder: maps import relationships between files.

Parses import statements in Python and JavaScript/TypeScript to build
a directed graph of file dependencies. Used by the graph_distance signal
to score files by their structural proximity to task-relevant files.

Uses networkx for graph storage and shortest-path computation.
"""

import os
import re

import networkx as nx

from cognitive_cache.models import Source


class DependencyGraph:
    """A directed graph of file import relationships."""

    def __init__(self):
        self._graph = nx.DiGraph()
        self._undirected = None

    def add_file(self, path: str):
        self._graph.add_node(path)
        self._undirected = None

    def add_edge(self, importer: str, imported: str):
        self._graph.add_edge(importer, imported)
        self._undirected = None

    def has_edge(self, source: str, target: str) -> bool:
        return self._graph.has_edge(source, target)

    def shortest_distance(self, source: str, target: str) -> float:
        """Shortest path distance between two files (undirected).

        We treat the graph as undirected for distance because
        "A imports B" means both A and B are relevant to each other.
        Returns float('inf') if no path exists.
        """
        if self._undirected is None:
            self._undirected = self._graph.to_undirected()
        try:
            return nx.shortest_path_length(self._undirected, source, target)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return float("inf")

    @property
    def nodes(self) -> set[str]:
        return set(self._graph.nodes)


def _resolve_python_import(
    module_name: str, source_path: str, all_paths: set[str]
) -> str | None:
    """Resolve a Python module name to a file path.

    'from utils import x' -> 'utils.py'
    'from auth.service import login' -> 'auth/service.py'
    """
    parts = module_name.split(".")
    # Try as a direct file
    candidate = "/".join(parts) + ".py"
    if candidate in all_paths:
        return candidate
    # Try as a package __init__
    candidate = "/".join(parts) + "/__init__.py"
    if candidate in all_paths:
        return candidate
    # Try relative to the source file's directory
    source_dir = os.path.dirname(source_path)
    if source_dir:
        candidate = source_dir + "/" + "/".join(parts) + ".py"
        if candidate in all_paths:
            return candidate
    return None


def _resolve_js_import(
    import_path: str, source_path: str, all_paths: set[str]
) -> str | None:
    """Resolve a JS/TS import path to a file path.

    './utils' -> 'src/utils.js' (relative to importer)
    """
    if not import_path.startswith("."):
        return None  # Skip external packages

    source_dir = os.path.dirname(source_path)
    # Normalize the relative path
    resolved = os.path.normpath(os.path.join(source_dir, import_path)).replace(
        "\\", "/"
    )

    # Try with various extensions
    for ext in ("", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"):
        candidate = resolved + ext
        if candidate in all_paths:
            return candidate
    return None


def build_dependency_graph(sources: list[Source]) -> DependencyGraph:
    """Build a dependency graph from a list of indexed source files.

    Args:
        sources: List of Source objects from the repo indexer.

    Returns:
        A DependencyGraph with edges representing import relationships.
    """
    graph = DependencyGraph()
    all_paths = {s.path for s in sources}

    for source in sources:
        graph.add_file(source.path)

    for source in sources:
        if source.language == "python":
            # Match: import X, from X import Y
            for match in re.finditer(r"(?:from|import)\s+([\w.]+)", source.content):
                module = match.group(1)
                target = _resolve_python_import(module, source.path, all_paths)
                if target and target != source.path:
                    graph.add_edge(source.path, target)

        elif source.language in ("javascript", "typescript"):
            # Match: require('./path'), import X from './path', import('./path')
            for match in re.finditer(
                r"""(?:require|from|import)\s*\(?\s*['"]([^'"]+)['"]""", source.content
            ):
                import_path = match.group(1)
                target = _resolve_js_import(import_path, source.path, all_paths)
                if target and target != source.path:
                    graph.add_edge(source.path, target)

    return graph
