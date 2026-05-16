import json
import os
import tempfile

from cognitive_cache.indexer.graph_builder import (
    DependencyGraph,
    build_dependency_graph,
)
from cognitive_cache.indexer.repo_indexer import index_repo


def test_python_imports_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "main.py"), "w") as f:
            f.write("from utils import helper\nimport config\n")
        with open(os.path.join(tmpdir, "utils.py"), "w") as f:
            f.write("def helper(): pass\n")
        with open(os.path.join(tmpdir, "config.py"), "w") as f:
            f.write("DEBUG = True\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources)

        assert graph.has_edge("main.py", "utils.py")
        assert graph.has_edge("main.py", "config.py")


def test_js_imports_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "app.js"), "w") as f:
            f.write(
                "const utils = require('./utils');\nimport config from './config';\n"
            )
        with open(os.path.join(tmpdir, "src", "utils.js"), "w") as f:
            f.write("module.exports = {}\n")
        with open(os.path.join(tmpdir, "src", "config.js"), "w") as f:
            f.write("export default {}\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources)

        assert graph.has_edge("src/app.js", "src/utils.js")
        assert graph.has_edge("src/app.js", "src/config.js")


def test_graph_distance():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "a.py"), "w") as f:
            f.write("from b import x\n")
        with open(os.path.join(tmpdir, "b.py"), "w") as f:
            f.write("from c import y\n")
        with open(os.path.join(tmpdir, "c.py"), "w") as f:
            f.write("y = 1\n")
        with open(os.path.join(tmpdir, "unrelated.py"), "w") as f:
            f.write("z = 99\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources)

        assert graph.shortest_distance("a.py", "b.py") == 1
        assert graph.shortest_distance("a.py", "c.py") == 2
        assert graph.shortest_distance("a.py", "unrelated.py") == float("inf")


def test_shortest_distance_caches_undirected_graph():
    graph = DependencyGraph()
    graph.add_file("a.py")
    graph.add_file("b.py")
    graph.add_file("c.py")
    graph.add_edge("a.py", "b.py")
    graph.add_edge("b.py", "c.py")

    # First call computes and caches
    d1 = graph.shortest_distance("a.py", "c.py")
    assert d1 == 2

    # Second call should return same result (from cache)
    d2 = graph.shortest_distance("a.py", "c.py")
    assert d2 == 2

    # Verify cache exists
    assert graph._undirected is not None


def test_undirected_cache_invalidated_on_add_edge():
    graph = DependencyGraph()
    graph.add_file("a.py")
    graph.add_file("b.py")
    graph.add_file("c.py")
    graph.add_edge("a.py", "b.py")

    # Populate cache
    graph.shortest_distance("a.py", "b.py")
    assert graph._undirected is not None

    # Adding edge invalidates cache
    graph.add_edge("b.py", "c.py")
    assert graph._undirected is None

    # New distance computation works with updated graph
    d = graph.shortest_distance("a.py", "c.py")
    assert d == 2


def test_undirected_cache_invalidated_on_add_file():
    graph = DependencyGraph()
    graph.add_file("a.py")

    # Populate cache
    graph.shortest_distance("a.py", "a.py")
    assert graph._undirected is not None

    # Adding file invalidates cache
    graph.add_file("b.py")
    assert graph._undirected is None


def test_ts_alias_resolved_from_tsconfig():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "tsconfig.json"), "w") as f:
            json.dump({"compilerOptions": {"paths": {"@/*": ["src/*"]}}}, f)
        with open(os.path.join(tmpdir, "src", "app.ts"), "w") as f:
            f.write('import { helper } from "@/utils";\n')
        with open(os.path.join(tmpdir, "src", "utils.ts"), "w") as f:
            f.write("export function helper() {}\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources, repo_path=tmpdir)

        assert graph.has_edge("src/app.ts", "src/utils.ts")


def test_ts_fallback_alias_without_tsconfig():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "app.ts"), "w") as f:
            f.write('import { helper } from "@/utils";\n')
        with open(os.path.join(tmpdir, "src", "utils.ts"), "w") as f:
            f.write("export function helper() {}\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources, repo_path=tmpdir)

        assert graph.has_edge("src/app.ts", "src/utils.ts")


def test_go_imports_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "auth"))
        with open(os.path.join(tmpdir, "main.go"), "w") as f:
            f.write('package main\n\nimport "myapp/auth"\n\nfunc main() {}\n')
        with open(os.path.join(tmpdir, "auth", "service.go"), "w") as f:
            f.write("package auth\n\nfunc Authenticate() {}\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources)

        assert graph.has_edge("main.go", "auth/service.go")


def test_rust_mod_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "main.rs"), "w") as f:
            f.write("mod auth;\n\nfn main() {}\n")
        with open(os.path.join(tmpdir, "auth.rs"), "w") as f:
            f.write("pub fn login() {}\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources)

        assert graph.has_edge("main.rs", "auth.rs")


def test_java_import_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "com", "example"))
        with open(os.path.join(tmpdir, "App.java"), "w") as f:
            f.write("import com.example.UserService;\n\npublic class App {}\n")
        with open(os.path.join(tmpdir, "com", "example", "UserService.java"), "w") as f:
            f.write("package com.example;\n\npublic class UserService {}\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources)

        assert graph.has_edge("App.java", "com/example/UserService.java")


def test_ruby_require_relative_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "lib"))
        with open(os.path.join(tmpdir, "app.rb"), "w") as f:
            f.write("require_relative 'lib/auth'\n")
        with open(os.path.join(tmpdir, "lib", "auth.rb"), "w") as f:
            f.write("class Auth; end\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources)

        assert graph.has_edge("app.rb", "lib/auth.rb")


def test_c_include_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "main.c"), "w") as f:
            f.write('#include "utils.h"\n\nint main() { return 0; }\n')
        with open(os.path.join(tmpdir, "utils.h"), "w") as f:
            f.write("int helper(int x);\n")

        sources = index_repo(tmpdir)
        graph = build_dependency_graph(sources)

        assert graph.has_edge("main.c", "utils.h")
