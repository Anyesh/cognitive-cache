import os
import tempfile

from cognitive_cache.indexer.token_counter import count_tokens
from cognitive_cache.indexer.repo_indexer import index_repo


def test_count_tokens_simple():
    text = "def hello_world():\n    print('hello')"
    count = count_tokens(text)
    assert isinstance(count, int)
    assert count > 0
    assert count < 100


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_index_repo_finds_python_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write("def main():\n    pass\n")
        with open(os.path.join(tmpdir, "src", "utils.py"), "w") as f:
            f.write("def helper(x):\n    return x + 1\n")
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# Hello")

        sources = index_repo(tmpdir)

        paths = {s.path for s in sources}
        assert "src/main.py" in paths
        assert "src/utils.py" in paths
        assert "README.md" not in paths


def test_index_repo_source_has_correct_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "app.py"), "w") as f:
            f.write("class UserService:\n    def get_user(self, uid):\n        pass\n")

        sources = index_repo(tmpdir)

        assert len(sources) == 1
        s = sources[0]
        assert s.path == "app.py"
        assert "class UserService" in s.content
        assert s.token_count > 0
        assert s.language == "python"
        assert "UserService" in s.symbols
        assert "get_user" in s.symbols


def test_index_repo_skips_vendored():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "node_modules", "pkg"))
        with open(os.path.join(tmpdir, "node_modules", "pkg", "index.js"), "w") as f:
            f.write("module.exports = {}")
        with open(os.path.join(tmpdir, "app.js"), "w") as f:
            f.write("const x = require('./pkg')")

        sources = index_repo(tmpdir)

        paths = {s.path for s in sources}
        assert "app.js" in paths
        assert "node_modules/pkg/index.js" not in paths
