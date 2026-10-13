"""Contract tests for the test-gen benchmark task family.

Each task shows a small pristine module and asks for tests. The scorer
(`mutation_kill`) runs the reply against the pristine module and against a
planted single mutation; only a test that is green on pristine and red on
the mutant passes. The sandbox runs plain ``python`` — no pytest — so the
prompts ask for test functions that are also called at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_bench_scoring import requires_docker

from router.bench.schema import Task, load_tasks
from router.bench.scoring import score

TASKS_PATH = Path(__file__).parent.parent / "router/bench/tasks/test_gen.json"

# One hand-written test per task that a competent reply would contain: it
# must be green on the pristine module and kill the planted mutant.
KILLING_TESTS: dict[str, str] = {
    "tgen-clamp": (
        "from target import clamp\n"
        "assert clamp(5, 0, 10) == 5\n"
        "assert clamp(-3, 0, 10) == 0\n"
        "assert clamp(42, 0, 10) == 10\n"
    ),
    "tgen-paginate": (
        "from target import paginate\n"
        "items = list(range(1, 8))\n"
        "assert paginate(items, 1, 3) == [1, 2, 3]\n"
        "assert paginate(items, 2, 3) == [4, 5, 6]\n"
        "assert paginate(items, 3, 3) == [7]\n"
    ),
    "tgen-parse-bool": (
        "from target import parse_bool\n"
        "assert parse_bool('yes') is True\n"
        "assert parse_bool('No') is False\n"
        "assert parse_bool('maybe') is False\n"
        "assert parse_bool('maybe', default=True) is True\n"
    ),
}


def _load() -> list[Task]:
    return load_tasks(json.loads(TASKS_PATH.read_text(encoding="utf-8")))


def test_family_shape() -> None:
    tasks = _load()
    assert len(tasks) == 3
    assert len({t.task_id for t in tasks}) == 3
    for t in tasks:
        assert t.task_id.startswith("tgen-")
        assert t.role == "test-gen"
        assert t.scorer == "mutation_kill"
        assert 400 <= t.max_tokens <= 512
        args = t.scorer_args
        assert set(args) == {"pristine_module", "mutant_module", "module_name"}
        assert args["module_name"] == "target"
        assert args["pristine_module"].strip() and args["mutant_module"].strip()
        assert args["pristine_module"] != args["mutant_module"]


def test_mutation_is_a_single_changed_line() -> None:
    for t in _load():
        pristine = t.scorer_args["pristine_module"].splitlines()
        mutant = t.scorer_args["mutant_module"].splitlines()
        assert len(pristine) == len(mutant), t.task_id
        diff = [(a, b) for a, b in zip(pristine, mutant, strict=True) if a != b]
        assert len(diff) == 1, (t.task_id, diff)


def test_prompt_shows_pristine_not_mutant() -> None:
    for t in _load():
        pristine = t.scorer_args["pristine_module"].splitlines()
        mutant = t.scorer_args["mutant_module"].splitlines()
        (mutated_line,) = [b for a, b in zip(pristine, mutant, strict=True) if a != b]
        assert t.scorer_args["pristine_module"] in t.prompt, t.task_id
        assert mutated_line.strip() not in t.prompt, t.task_id
        assert "import pytest" not in t.prompt


def test_killing_test_keys_match_task_ids() -> None:
    assert {t.task_id for t in _load()} == set(KILLING_TESTS)


@requires_docker
@pytest.mark.parametrize("task_id", sorted(KILLING_TESTS))
def test_hand_written_test_kills_mutant_in_sandbox(task_id: str) -> None:
    """End-to-end proof through the real docker sandbox: the reference test
    is green on pristine and red on the mutant for every task."""
    task = next(t for t in _load() if t.task_id == task_id)
    reply = "```python\n" + KILLING_TESTS[task_id] + "```\n"
    assert score(task.scorer, reply, task.scorer_args) is True
