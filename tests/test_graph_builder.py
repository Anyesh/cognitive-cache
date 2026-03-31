import os
import tempfile

from cognitive_cache.indexer.graph_builder import build_dependency_graph
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
            f.write("const utils = require('./utils');\nimport config from './config';\n")
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
