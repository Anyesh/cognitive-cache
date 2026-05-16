"""Dependency graph builder: maps import relationships between files.

Parses import statements in Python and JavaScript/TypeScript to build
a directed graph of file dependencies. Used by the graph_distance signal
to score files by their structural proximity to task-relevant files.

Uses networkx for graph storage and shortest-path computation.
"""

import json
import os
import re

import networkx as nx

from cognitive_cache.models import Source


class DependencyGraph:
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
        """Shortest undirected path distance between two files, because
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
    parts = module_name.split(".")
    candidate = "/".join(parts) + ".py"
    if candidate in all_paths:
        return candidate
    candidate = "/".join(parts) + "/__init__.py"
    if candidate in all_paths:
        return candidate
    source_dir = os.path.dirname(source_path)
    if source_dir:
        candidate = source_dir + "/" + "/".join(parts) + ".py"
        if candidate in all_paths:
            return candidate
    return None


def _load_ts_aliases(repo_path: str | None) -> dict[str, str]:
    """Read tsconfig.json/jsconfig.json compilerOptions.paths, because TS projects
    use path aliases like @/ and ~/ that cannot be resolved from the import string alone.
    Falls back to common conventions when no config is found.
    """
    if repo_path is None:
        return {"@/": "src/", "~/": "src/"}

    for config_name in ("tsconfig.json", "jsconfig.json"):
        config_path = os.path.join(repo_path, config_name)
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.loads(f.read())
            paths = config.get("compilerOptions", {}).get("paths", {})
            aliases: dict[str, str] = {}
            for alias, targets in paths.items():
                if not targets:
                    continue
                clean_alias = alias.rstrip("*")
                clean_target = targets[0].rstrip("*")
                if not clean_alias:
                    continue
                aliases[clean_alias] = clean_target
            if aliases:
                return aliases
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    return {"@/": "src/", "~/": "src/"}


def _resolve_js_import(
    import_path: str,
    source_path: str,
    all_paths: set[str],
    aliases: dict[str, str] | None = None,
) -> str | None:
    resolved_path = import_path

    if not import_path.startswith("."):
        if aliases:
            for alias, target_dir in aliases.items():
                if import_path.startswith(alias):
                    for ext in (
                        "",
                        ".js",
                        ".ts",
                        ".jsx",
                        ".tsx",
                        "/index.js",
                        "/index.ts",
                    ):
                        candidate = (
                            target_dir + import_path[len(alias) :] + ext
                        ).replace("\\", "/")
                        if candidate in all_paths:
                            return candidate
                    # Alias matched but no file found — don't fall through to
                    # relative resolution, because the import was clearly aliased
                    return None
        return None

    source_dir = os.path.dirname(source_path)
    resolved = os.path.normpath(os.path.join(source_dir, resolved_path)).replace(
        "\\", "/"
    )

    for ext in ("", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"):
        candidate = resolved + ext
        if candidate in all_paths:
            return candidate
    return None


def _resolve_go_import(import_path: str, all_paths: set[str]) -> list[str]:
    """Returns all files in matching package directory, because Go imports are
    package-level — all .go files in the same directory share the package.
    """
    parts = import_path.strip('"').strip("/").split("/")
    matches = []
    for length in range(len(parts), 0, -1):
        candidate_dir = "/".join(parts[-length:])
        for p in all_paths:
            if os.path.dirname(p) == candidate_dir or p.startswith(candidate_dir + "/"):
                dir_of_p = os.path.dirname(p)
                if dir_of_p.endswith(candidate_dir):
                    matches.append(p)
        if matches:
            break
    return matches


def _resolve_rust_mod(
    mod_name: str, source_path: str, all_paths: set[str]
) -> str | None:
    source_dir = os.path.dirname(source_path)
    candidates = [
        f"{source_dir}/{mod_name}.rs" if source_dir else f"{mod_name}.rs",
        f"{source_dir}/{mod_name}/mod.rs" if source_dir else f"{mod_name}/mod.rs",
    ]
    for c in candidates:
        if c in all_paths:
            return c
    return None


def _resolve_java_import(class_path: str, all_paths: set[str]) -> str | None:
    parts = class_path.split(".")
    candidate = "/".join(parts) + ".java"
    if candidate in all_paths:
        return candidate
    class_name = parts[-1] + ".java"
    for p in all_paths:
        if p.endswith("/" + class_name) or p == class_name:
            return p
    return None


def _resolve_ruby_require(
    require_path: str, source_path: str, all_paths: set[str], is_relative: bool
) -> str | None:
    if is_relative:
        source_dir = os.path.dirname(source_path)
        resolved = os.path.normpath(os.path.join(source_dir, require_path)).replace(
            "\\", "/"
        )
        for ext in ("", ".rb"):
            candidate = resolved + ext
            if candidate in all_paths:
                return candidate
    else:
        for ext in ("", ".rb"):
            candidate = require_path + ext
            if candidate in all_paths:
                return candidate
        for ext in ("", ".rb"):
            candidate = "lib/" + require_path + ext
            if candidate in all_paths:
                return candidate
    return None


def _resolve_c_include(
    include_path: str, source_path: str, all_paths: set[str]
) -> str | None:
    source_dir = os.path.dirname(source_path)
    if source_dir:
        candidate = os.path.normpath(os.path.join(source_dir, include_path)).replace(
            "\\", "/"
        )
        if candidate in all_paths:
            return candidate
    if include_path in all_paths:
        return include_path
    return None


def build_dependency_graph(
    sources: list[Source], repo_path: str | None = None
) -> DependencyGraph:
    graph = DependencyGraph()
    all_paths = {s.path for s in sources}

    for source in sources:
        graph.add_file(source.path)

    ts_aliases: dict[str, str] | None = None

    for source in sources:
        if source.language == "python":
            for match in re.finditer(r"(?:from|import)\s+([\w.]+)", source.content):
                module = match.group(1)
                target = _resolve_python_import(module, source.path, all_paths)
                if target and target != source.path:
                    graph.add_edge(source.path, target)

        elif source.language in ("javascript", "typescript"):
            if ts_aliases is None:
                ts_aliases = _load_ts_aliases(repo_path)
            for match in re.finditer(
                r"""(?:require|from|import)\s*\(?\s*['"]([^'"]+)['"]""",
                source.content,
            ):
                import_path = match.group(1)
                target = _resolve_js_import(
                    import_path, source.path, all_paths, ts_aliases
                )
                if target and target != source.path:
                    graph.add_edge(source.path, target)

        elif source.language == "go":
            for match in re.finditer(r'import\s+"([^"]+)"', source.content):
                targets = _resolve_go_import(match.group(1), all_paths)
                for target in targets:
                    if target != source.path:
                        graph.add_edge(source.path, target)
            for match in re.finditer(r"import\s+\((.*?)\)", source.content, re.DOTALL):
                for pkg_match in re.findall(r'"([^"]+)"', match.group(1)):
                    targets = _resolve_go_import(pkg_match, all_paths)
                    for target in targets:
                        if target != source.path:
                            graph.add_edge(source.path, target)

        elif source.language == "rust":
            for match in re.finditer(r"mod\s+(\w+)\s*;", source.content):
                target = _resolve_rust_mod(match.group(1), source.path, all_paths)
                if target and target != source.path:
                    graph.add_edge(source.path, target)
            for match in re.finditer(r"use\s+(?:crate|super)::(\w+)", source.content):
                target = _resolve_rust_mod(match.group(1), source.path, all_paths)
                if target and target != source.path:
                    graph.add_edge(source.path, target)

        elif source.language == "java":
            for match in re.finditer(r"import\s+([\w.]+)\s*;", source.content):
                target = _resolve_java_import(match.group(1), all_paths)
                if target and target != source.path:
                    graph.add_edge(source.path, target)

        elif source.language == "ruby":
            for match in re.finditer(
                r"require_relative\s+['\"]([^'\"]+)['\"]", source.content
            ):
                target = _resolve_ruby_require(
                    match.group(1), source.path, all_paths, is_relative=True
                )
                if target and target != source.path:
                    graph.add_edge(source.path, target)
            for match in re.finditer(r"require\s+['\"]([^'\"]+)['\"]", source.content):
                path = match.group(1)
                if not path.startswith("."):
                    target = _resolve_ruby_require(
                        path, source.path, all_paths, is_relative=False
                    )
                    if target and target != source.path:
                        graph.add_edge(source.path, target)

        elif source.language in ("c", "cpp"):
            for match in re.finditer(r'#include\s+"([^"]+)"', source.content):
                target = _resolve_c_include(match.group(1), source.path, all_paths)
                if target and target != source.path:
                    graph.add_edge(source.path, target)

    return graph
