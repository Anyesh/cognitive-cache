"""Repo indexer: turns a directory into a list of Source objects.

Walks the file tree, reads source files, counts tokens, and extracts
symbols (identifiers like function/class names). Skips vendored code,
generated files, and non-source files.
"""

import os
import re

from cognitive_cache.models import Source
from cognitive_cache.indexer.token_counter import count_tokens

# File extensions we consider "source code"
SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx"}

# Directories to always skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".tox", "egg-info",
    "vendor", ".mypy_cache", ".pytest_cache",
}


def _detect_language(path: str) -> str:
    """Map file extension to language name."""
    ext = os.path.splitext(path)[1]
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    }.get(ext, "unknown")


def _extract_symbols(content: str, language: str) -> frozenset[str]:
    """Extract identifier names (functions, classes, variables) from source code.

    This is intentionally simple — regex-based, not a full parser.
    It catches the most common patterns: def/class in Python, function/class/const in JS/TS.
    Precision > recall here: we'd rather miss some symbols than extract noise.
    """
    symbols = set()

    if language == "python":
        # def function_name, class ClassName
        symbols.update(re.findall(r"(?:def|class)\s+(\w+)", content))
        # top-level assignments: CONSTANT = ... or variable = ...
        symbols.update(re.findall(r"^(\w+)\s*=", content, re.MULTILINE))
    elif language in ("javascript", "typescript"):
        # function name, class Name
        symbols.update(re.findall(r"(?:function|class)\s+(\w+)", content))
        # const/let/var name
        symbols.update(re.findall(r"(?:const|let|var)\s+(\w+)", content))
        # export default function/class
        symbols.update(re.findall(r"export\s+(?:default\s+)?(?:function|class)\s+(\w+)", content))

    return frozenset(symbols)


def index_repo(repo_path: str) -> list[Source]:
    """Index all source files in a repository.

    Args:
        repo_path: Absolute path to the repo root.

    Returns:
        List of Source objects, one per source file.
    """
    sources = []

    for root, dirs, files in os.walk(repo_path):
        # Prune skipped directories in-place (prevents os.walk from descending)
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

            sources.append(Source(
                path=rel_path,
                content=content,
                token_count=token_count,
                language=language,
                symbols=symbols,
            ))

    return sources
