"""Repo indexer: turns a directory into a list of Source objects.

Walks the file tree, reads source files, counts tokens, and extracts
symbols (identifiers like function/class names). Skips vendored code,
generated files, and non-source files.
"""

import os
import re

from cognitive_cache.models import Source
from cognitive_cache.indexer.token_counter import count_tokens

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".c",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
}

SKIP_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".tox",
    "egg-info",
    "vendor",
    ".mypy_cache",
    ".pytest_cache",
    "target",
    "pkg",
    "_build",
    "Pods",
    ".gradle",
    ".cxx",
}

_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}


def _detect_language(path: str) -> str:
    ext = os.path.splitext(path)[1]
    return _LANGUAGE_MAP.get(ext, "unknown")


def _extract_symbols(content: str, language: str) -> frozenset[str]:
    """Extract identifier names from source code. Regex-based, precision over recall:
    catches common declaration patterns per language without requiring a full parser.
    """
    symbols = set()

    if language == "python":
        symbols.update(re.findall(r"(?:def|class)\s+(\w+)", content))
        symbols.update(re.findall(r"^(\w+)\s*=", content, re.MULTILINE))

    elif language in ("javascript", "typescript"):
        symbols.update(re.findall(r"(?:function|class)\s+(\w+)", content))
        symbols.update(re.findall(r"(?:const|let|var)\s+(\w+)", content))
        symbols.update(
            re.findall(r"export\s+(?:default\s+)?(?:function|class)\s+(\w+)", content)
        )
        symbols.update(re.findall(r"(?:interface|enum)\s+(\w+)", content))
        symbols.update(re.findall(r"(?:export\s+)?type\s+(\w+)\s*=", content))

    elif language == "go":
        symbols.update(
            re.findall(r"^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(", content, re.MULTILINE)
        )
        symbols.update(
            re.findall(r"^type\s+(\w+)\s+(?:struct|interface)\b", content, re.MULTILINE)
        )
        symbols.update(re.findall(r"^type\s+(\w+)\s+=?\s*\w", content, re.MULTILINE))
        symbols.update(re.findall(r"^var\s+(\w+)\s", content, re.MULTILINE))
        symbols.update(re.findall(r"^const\s+(\w+)\s", content, re.MULTILINE))

    elif language == "rust":
        symbols.update(re.findall(r"(?:pub\s+)?fn\s+(\w+)", content))
        symbols.update(re.findall(r"(?:pub\s+)?struct\s+(\w+)", content))
        symbols.update(re.findall(r"(?:pub\s+)?enum\s+(\w+)", content))
        symbols.update(re.findall(r"(?:pub\s+)?trait\s+(\w+)", content))
        symbols.update(re.findall(r"impl(?:<[^>]*>)?\s+(\w+)", content))
        symbols.update(re.findall(r"(?:pub\s+)?type\s+(\w+)\s*=", content))
        symbols.update(re.findall(r"(?:pub\s+)?(?:static|const)\s+(\w+)\s*:", content))

    elif language == "java":
        symbols.update(
            re.findall(
                r"(?:public|private|protected|static|abstract|final|\s)*class\s+(\w+)",
                content,
            )
        )
        symbols.update(
            re.findall(
                r"(?:public|private|protected|static|abstract|final|\s)*interface\s+(\w+)",
                content,
            )
        )
        symbols.update(
            re.findall(
                r"(?:public|private|protected|static|abstract|final|\s)*enum\s+(\w+)",
                content,
            )
        )
        symbols.update(
            re.findall(
                r"(?:public|private|protected|static|final|abstract|synchronized|native|\s)+"
                r"[\w<>\[\],\s]+\s+(\w+)\s*\(",
                content,
            )
        )

    elif language == "ruby":
        symbols.update(re.findall(r"(?:def)\s+(?:self\.)?(\w+)", content))
        symbols.update(re.findall(r"(?:class|module)\s+(\w+)", content))
        symbols.update(re.findall(r"^\s*(\w+)\s*=", content, re.MULTILINE))

    elif language in ("c", "cpp"):
        symbols.update(re.findall(r"(?:class|struct|enum|union)\s+(\w+)", content))
        symbols.update(re.findall(r"#define\s+(\w+)", content))
        symbols.update(
            re.findall(r"^[\w*\s]+\s+(\w+)\s*\([^;]*$", content, re.MULTILINE)
        )
        symbols.update(re.findall(r"typedef\s+.*\s+(\w+)\s*;", content))

    symbols.discard("")
    return frozenset(symbols)


def _is_test_file(path: str, language: str) -> bool:
    basename = os.path.basename(path)
    dirname = path.lower()

    dir_parts = set(os.path.dirname(dirname).split("/"))
    if dir_parts & {"test", "tests", "__tests__"}:
        return True

    if language == "python":
        return basename.startswith("test_") or basename.endswith("_test.py")

    if language in ("javascript", "typescript"):
        return (
            basename.endswith(".test.js")
            or basename.endswith(".test.ts")
            or basename.endswith(".test.jsx")
            or basename.endswith(".test.tsx")
            or basename.endswith(".spec.js")
            or basename.endswith(".spec.ts")
            or basename.endswith(".spec.jsx")
            or basename.endswith(".spec.tsx")
        )

    if language == "go":
        return basename.endswith("_test.go")

    if language == "rust":
        return basename == "tests.rs" or "/tests/" in path

    if language == "java":
        return basename.endswith("Test.java") or "src/test/" in path

    if language == "ruby":
        return basename.endswith("_spec.rb") or "/spec/" in path

    return False


def index_repo(repo_path: str) -> list[Source]:
    sources = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext not in SOURCE_EXTENSIONS:
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, PermissionError):
                continue

            language = _detect_language(filename)
            symbols = _extract_symbols(content, language)
            token_count = count_tokens(content)
            is_test = _is_test_file(rel_path, language)

            sources.append(
                Source(
                    path=rel_path,
                    content=content,
                    token_count=token_count,
                    language=language,
                    symbols=symbols,
                    is_test=is_test,
                )
            )

    return sources
