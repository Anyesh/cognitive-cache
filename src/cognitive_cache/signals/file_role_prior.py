"""Signal 6: File Role Prior.

Assigns a baseline score based on what kind of file this is.
Source files are the primary target for most tasks. Test files are
suppressed unless the task explicitly involves testing, because they
were polluting results for non-testing tasks (confirmed by audit of
real usage across 30+ projects).
"""

import re

from cognitive_cache.models import Source, Task
from cognitive_cache.signals.base import Signal

ROLE_PRIORS = {
    "test": 0.2,
    "type_definition": 0.7,
    "config": 0.5,
    "init": 0.2,
    "source": 0.6,
}

_TEST_KEYWORDS = {"test", "spec", "testing", "coverage", "fixture", "mock", "stub"}


def _classify_file_role(path: str, content: str) -> str:
    basename = path.rsplit("/", 1)[-1] if "/" in path else path

    if basename in ("__init__.py", "index.js", "index.ts"):
        return "init"

    if (
        basename.endswith(".d.ts")
        or basename.endswith("types.py")
        or basename.endswith("types.ts")
    ):
        return "type_definition"
    if re.search(r"(Protocol|TypedDict|TypeAlias|interface\s+\w+)", content):
        return "type_definition"

    if basename in ("config.py", "settings.py", "config.js", "config.ts"):
        return "config"
    if "conftest" in basename:
        return "config"

    return "source"


class FileRolePriorSignal(Signal):
    def score(
        self, source: Source, task: Task, selected: list[Source], **kwargs
    ) -> float:
        if source.is_test:
            task_text = task.full_text.lower()
            if any(kw in task_text for kw in _TEST_KEYWORDS):
                return 0.6
            return 0.2
        role = _classify_file_role(source.path, source.content)
        return ROLE_PRIORS.get(role, 0.6)
