import os
import tempfile

from cognitive_cache.indexer.token_counter import count_tokens
from cognitive_cache.indexer.repo_indexer import (
    index_repo,
    _extract_symbols,
    _is_test_file,
)


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


def test_index_repo_marks_test_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "tests"))
        with open(os.path.join(tmpdir, "tests", "test_auth.py"), "w") as f:
            f.write("def test_login(): pass\n")
        with open(os.path.join(tmpdir, "auth.py"), "w") as f:
            f.write("def login(): pass\n")

        sources = index_repo(tmpdir)
        by_path = {s.path: s for s in sources}

        assert by_path["tests/test_auth.py"].is_test is True
        assert by_path["auth.py"].is_test is False


def test_index_repo_finds_go_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "main.go"), "w") as f:
            f.write(
                "package main\n\nfunc main() {\n}\n\nfunc handleRequest(w http.ResponseWriter) {\n}\n"
            )

        sources = index_repo(tmpdir)

        assert len(sources) == 1
        s = sources[0]
        assert s.language == "go"
        assert "main" in s.symbols
        assert "handleRequest" in s.symbols
        assert s.is_test is False


def test_index_repo_finds_rust_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "lib.rs"), "w") as f:
            f.write("pub struct Config {\n}\n\npub fn parse_config() -> Config {\n}\n")

        sources = index_repo(tmpdir)

        assert len(sources) == 1
        s = sources[0]
        assert s.language == "rust"
        assert "Config" in s.symbols
        assert "parse_config" in s.symbols


def test_index_repo_finds_java_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "App.java"), "w") as f:
            f.write("public class App {\n    public void run() {\n    }\n}\n")

        sources = index_repo(tmpdir)

        assert len(sources) == 1
        s = sources[0]
        assert s.language == "java"
        assert "App" in s.symbols
        assert "run" in s.symbols


def test_index_repo_finds_ruby_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "app.rb"), "w") as f:
            f.write("class UserController\n  def index\n  end\nend\n")

        sources = index_repo(tmpdir)

        assert len(sources) == 1
        s = sources[0]
        assert s.language == "ruby"
        assert "UserController" in s.symbols
        assert "index" in s.symbols


def test_index_repo_finds_c_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "main.c"), "w") as f:
            f.write(
                "#define MAX_SIZE 100\n\nint main(int argc, char *argv[])\n{\n    return 0;\n}\n"
            )

        sources = index_repo(tmpdir)

        assert len(sources) == 1
        s = sources[0]
        assert s.language == "c"
        assert "MAX_SIZE" in s.symbols
        assert "main" in s.symbols


def test_extract_symbols_go():
    content = """package auth

type AuthService struct {
    db *Database
}

func (s *AuthService) Authenticate(token string) bool {
    return true
}

func NewAuthService(db *Database) *AuthService {
    return &AuthService{db: db}
}

var DefaultTimeout = 30

const MaxRetries = 3
"""
    symbols = _extract_symbols(content, "go")
    assert "AuthService" in symbols
    assert "Authenticate" in symbols
    assert "NewAuthService" in symbols
    assert "DefaultTimeout" in symbols
    assert "MaxRetries" in symbols


def test_extract_symbols_rust():
    content = """pub struct Server {
    port: u16,
}

pub enum Status {
    Active,
    Inactive,
}

pub trait Handler {
    fn handle(&self);
}

impl Server {
    pub fn new(port: u16) -> Self {
        Server { port }
    }
}

pub fn start_server() {
}

pub const MAX_CONNECTIONS: usize = 100;
"""
    symbols = _extract_symbols(content, "rust")
    assert "Server" in symbols
    assert "Status" in symbols
    assert "Handler" in symbols
    assert "handle" in symbols
    assert "new" in symbols
    assert "start_server" in symbols
    assert "MAX_CONNECTIONS" in symbols


def test_extract_symbols_java():
    content = """public class UserService {
    private final UserRepository repo;

    public UserService(UserRepository repo) {
        this.repo = repo;
    }

    public User findById(Long id) {
        return repo.findById(id);
    }

    private void validate(User user) {
    }
}

public interface UserRepository {
    User findById(Long id);
}

public enum Role {
    ADMIN, USER
}
"""
    symbols = _extract_symbols(content, "java")
    assert "UserService" in symbols
    assert "findById" in symbols
    assert "UserRepository" in symbols
    assert "Role" in symbols


def test_extract_symbols_ruby():
    content = """class ApplicationController
  def index
    render json: items
  end

  def self.configure
    yield config
  end
end

module Authentication
end

BASE_URL = "https://api.example.com"
"""
    symbols = _extract_symbols(content, "ruby")
    assert "ApplicationController" in symbols
    assert "index" in symbols
    assert "configure" in symbols
    assert "Authentication" in symbols
    assert "BASE_URL" in symbols


def test_extract_symbols_typescript_interfaces():
    content = """interface UserProps {
    name: string;
    age: number;
}

export type Config = {
    debug: boolean;
};

enum Direction {
    Up,
    Down,
}

const API_URL = "https://api.com";
"""
    symbols = _extract_symbols(content, "typescript")
    assert "UserProps" in symbols
    assert "Config" in symbols
    assert "Direction" in symbols
    assert "API_URL" in symbols


def test_extract_symbols_c():
    content = """#define BUFFER_SIZE 1024

struct Connection {
    int fd;
    char *host;
};

typedef struct Connection conn_t;

int handle_connection(int fd)
{
    return 0;
}
"""
    symbols = _extract_symbols(content, "c")
    assert "BUFFER_SIZE" in symbols
    assert "Connection" in symbols
    assert "conn_t" in symbols
    assert "handle_connection" in symbols


def test_is_test_file_python():
    assert _is_test_file("test_auth.py", "python") is True
    assert _is_test_file("auth_test.py", "python") is True
    assert _is_test_file("tests/test_login.py", "python") is True
    assert _is_test_file("auth.py", "python") is False


def test_is_test_file_javascript():
    assert _is_test_file("auth.test.js", "javascript") is True
    assert _is_test_file("auth.spec.ts", "typescript") is True
    assert _is_test_file("__tests__/auth.js", "javascript") is True
    assert _is_test_file("auth.js", "javascript") is False


def test_is_test_file_go():
    assert _is_test_file("auth_test.go", "go") is True
    assert _is_test_file("auth.go", "go") is False


def test_is_test_file_rust():
    assert _is_test_file("tests.rs", "rust") is True
    assert _is_test_file("tests/integration.rs", "rust") is True
    assert _is_test_file("lib.rs", "rust") is False


def test_is_test_file_java():
    assert _is_test_file("UserServiceTest.java", "java") is True
    assert _is_test_file("src/test/java/App.java", "java") is True
    assert _is_test_file("App.java", "java") is False


def test_is_test_file_ruby():
    assert _is_test_file("auth_spec.rb", "ruby") is True
    assert _is_test_file("spec/auth_spec.rb", "ruby") is True
    assert _is_test_file("auth.rb", "ruby") is False
