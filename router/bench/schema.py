"""Task and result records for the synthetic benchmark.

A Task is pure data: the prompt sent (verbatim) and the name+args of the
deterministic scorer that judges the reply. Scorer names are looked up in
router.bench.scoring.SCORERS — an unknown name fails at load time, not at
run time, so a typo cannot silently score as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROLES = (
    "triage",  # file ranking / traceback diagnosis (read-only)
    "test-gen",  # test generation, judged by mutation kill
    "repair",  # single-file red-test repair
    "tool-correctness",  # shell/tool-call argument correctness
    "resistance",  # instruction / secret-exfiltration resistance
    "docs-coherence",  # docs summarisation stays faithful
    "context-limit",  # behaviour near the context bound
)


@dataclass(frozen=True)
class Task:
    task_id: str
    role: str
    prompt: str
    scorer: str
    scorer_args: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 512
    tools: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        from router.bench.scoring import SCORERS

        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")
        if self.scorer not in SCORERS:
            raise ValueError(f"unknown scorer {self.scorer!r} in task {self.task_id}")
        if not self.task_id or not self.prompt:
            raise ValueError("task_id and prompt are required")


def load_tasks(raw: list[dict[str, Any]]) -> list[Task]:
    """Build validated Tasks from parsed JSON; duplicate ids are an error."""
    tasks = [Task(**item) for item in raw]
    ids = [t.task_id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate task ids: {sorted(dupes)}")
    return tasks
