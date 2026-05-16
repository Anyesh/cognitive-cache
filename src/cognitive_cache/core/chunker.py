"""Smart file chunking: extract relevant portions of large files.

When a file is too large for the context budget, we don't skip it entirely.
Instead, we extract the most relevant chunks — functions, classes, or
regions that contain task-mentioned symbols.

This is critical for large core files like app.py (13K tokens) that
can't fit whole but contain the exact function we need.
"""

import re

from cognitive_cache.models import Source, Task
from cognitive_cache.indexer.token_counter import count_tokens


def _find_relevant_regions(
    content: str, task_symbols: frozenset[str], language: str
) -> list[tuple[int, int]]:
    """Find line ranges in the file that contain task-relevant symbols.

    Returns list of (start_line, end_line) tuples.
    """
    lines = content.split("\n")
    regions = []
    task_words = {s.lower() for s in task_symbols}

    if language == "python":
        # Find function/class definitions that match task symbols
        block_start = None
        block_indent = 0

        for i, line in enumerate(lines):
            # Check if this is a def/class line
            match = re.match(r"^(\s*)(def |class )", line)
            if match:
                # If we were tracking a block, close it
                if block_start is not None:
                    regions.append((block_start, i - 1))
                    block_start = None

                # Check if this block is relevant
                line_lower = line.lower()
                if any(w in line_lower for w in task_words):
                    block_start = max(0, i - 2)  # include 2 lines of context before
                    block_indent = len(match.group(1))

            elif block_start is not None:
                # Check if we've left the block (dedented)
                stripped = line.strip()
                if (
                    stripped
                    and not line.startswith(" " * (block_indent + 1))
                    and not stripped.startswith("#")
                ):
                    # Check indentation — if we're at same or less indent, block ended
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= block_indent and stripped:
                        regions.append((block_start, i - 1))
                        block_start = None

        # Close final block
        if block_start is not None:
            regions.append((block_start, len(lines) - 1))

    elif language in ("javascript", "typescript"):
        # Simpler: find lines with task words and grab surrounding context
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(w in line_lower for w in task_words):
                start = max(0, i - 5)
                end = min(len(lines) - 1, i + 10)
                regions.append((start, end))

    # Also find any line mentioning task symbols (catch things outside functions)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for w in task_words:
            if len(w) >= 4 and w in line_lower:
                start = max(0, i - 2)
                end = min(len(lines) - 1, i + 2)
                regions.append((start, end))
                break

    return regions


def _merge_regions(
    regions: list[tuple[int, int]], margin: int = 3
) -> list[tuple[int, int]]:
    """Merge overlapping or nearby regions."""
    if not regions:
        return []
    sorted_regions = sorted(regions)
    merged = [sorted_regions[0]]
    for start, end in sorted_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + margin:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def chunk_source(source: Source, task: Task, max_tokens: int) -> Source:
    """Extract relevant portions of a large source file.

    If the file fits within max_tokens, return it unchanged.
    Otherwise, extract functions/regions that contain task-relevant symbols.

    Args:
        source: The full source file.
        task: The task with symbols to look for.
        max_tokens: Maximum token budget for this file.

    Returns:
        A new Source with truncated content, or the original if it fits.
    """
    if source.token_count <= max_tokens:
        return source

    lines = source.content.split("\n")
    regions = _find_relevant_regions(source.content, task.symbols, source.language)
    regions = _merge_regions(regions)

    if not regions:
        chunks = []
        for line in lines:
            chunks.append(line)
            if count_tokens("\n".join(chunks)) > max_tokens:
                chunks.pop()
                break
        content = "\n".join(chunks)
    else:
        # Build content from relevant regions
        chunks = [f"# [Cognitive Cache: showing relevant sections of {source.path}]"]
        for start, end in regions:
            chunks.append(f"\n# ... (line {start + 1})")
            chunks.extend(lines[start : end + 1])

        content = "\n".join(chunks)

        if count_tokens(content) > max_tokens:
            truncated = []
            for line in content.split("\n"):
                truncated.append(line)
                if count_tokens("\n".join(truncated)) > max_tokens:
                    truncated.pop()
                    break
            content = "\n".join(truncated)

    new_token_count = count_tokens(content)
    return Source(
        path=source.path,
        content=content,
        token_count=new_token_count,
        language=source.language,
        symbols=source.symbols,
        is_test=source.is_test,
    )
