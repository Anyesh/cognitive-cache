"""Standard prompt template for benchmark LLM calls.

CRITICAL: This must be identical across all strategies and models.
The ONLY variable is the context files section.
"""


def build_prompt(title: str, body: str, context_files: dict[str, str]) -> str:
    context_section = ""
    for path, content in context_files.items():
        context_section += f"\n### {path}\n```\n{content}\n```\n"

    return f"""You are a senior software engineer fixing a bug. Read the issue and the relevant source files, then produce a patch that fixes the issue.

## Issue

**{title}**

{body}

## Relevant Source Files
{context_section}

## Instructions

1. Analyze the issue and the source files above.
2. Identify the root cause of the bug.
3. Produce a minimal patch that fixes the issue.
4. Output ONLY the patch in unified diff format. Do not include explanations.

## Output Format

For each file you modify, output:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```
"""
