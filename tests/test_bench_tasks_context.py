"""Contract tests for the context-limit benchmark task family.

The family is needle-in-haystack: each prompt buries one distinctive fact
inside a long synthetic filler document at a different depth (early /
middle / late), and the deterministic scorer passes only when the reply
carries that fact back out.
"""

from __future__ import annotations

import json
from pathlib import Path

from router.bench.schema import Task, load_tasks
from router.bench.scoring import score

TASKS_PATH = (
    Path(__file__).parent.parent / "router" / "bench" / "tasks" / "context_limit.json"
)


def _tasks() -> list[Task]:
    return load_tasks(json.loads(TASKS_PATH.read_text()))


def _needle(task: Task) -> str:
    needles = task.scorer_args["needles"]
    assert len(needles) == 1
    assert isinstance(needles[0], str)
    return needles[0]


def test_family_shape() -> None:
    tasks = _tasks()
    assert len(tasks) == 3
    assert len({t.task_id for t in tasks}) == 3
    for t in tasks:
        assert t.task_id.startswith("ctx-")
        assert t.role == "context-limit"
        assert t.scorer == "contains_all"
        assert t.max_tokens <= 512


def test_needle_appears_exactly_once() -> None:
    for t in _tasks():
        assert t.prompt.count(_needle(t)) == 1, t.task_id


def test_prompt_length_bounds() -> None:
    for t in _tasks():
        assert 8_000 < len(t.prompt) < 20_000, (t.task_id, len(t.prompt))


def test_scorer_pass_and_fail() -> None:
    for t in _tasks():
        needle = _needle(t)
        assert score(t.scorer, f"The answer is {needle}.", t.scorer_args)
        miss = "I could not find that in the document."
        assert not score(t.scorer, miss, t.scorer_args)


def test_needle_positions_cover_early_middle_late() -> None:
    thirds = set()
    for t in _tasks():
        idx = t.prompt.index(_needle(t))
        thirds.add(min(2, idx * 3 // len(t.prompt)))
    assert thirds == {0, 1, 2}
