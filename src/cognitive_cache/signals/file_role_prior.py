"""Signal 6: File Role Prior.

Assigns a baseline score based on what kind of file this is.
Some file types are inherently more informative:

- Test files: reveal expected behavior and edge cases
- Type definitions: reveal contracts and interfaces
- Config files: reveal setup and environment
- Init files: usually just re-exports, less useful
- Regular source files: default baseline

These priors are hand-tuned. They're not meant to be the dominant signal —
just a tiebreaker when other signals are equal.
"""

import os
import re

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal

ROLE_PRIORS = {
    "test": 0.6,
    "type_definition": 0.7,
    "config": 0.5,
    "init": 0.2,
    "source": 0.4,
}


def _classify_file_role(path: str, content: str) -> str:
    """Classify a file into a role category."""
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)

    if "test" in dirname or basename.startswith("test_") or basename.endswith("_test.py"):
        return "test"
    if basename.endswith(".test.js") or basename.endswith(".test.ts") or basename.endswith(".spec.ts"):
        return "test"

    if basename in ("__init__.py", "index.js", "index.ts"):
        return "init"

    if basename.endswith(".d.ts") or basename.endswith("types.py") or basename.endswith("types.ts"):
        return "type_definition"
    if re.search(r"(typing|Protocol|TypedDict|interface\s+\w+)", content):
        return "type_definition"

    if basename in ("config.py", "settings.py", "config.js", "config.ts"):
        return "config"
    if "conftest" in basename:
        return "config"

    return "source"


class FileRolePriorSignal(Signal):
    def score(self, source: Source, task: Task, selected: list[Source], **kwargs) -> float:
        role = _classify_file_role(source.path, source.content)
        return ROLE_PRIORS.get(role, 0.4)
