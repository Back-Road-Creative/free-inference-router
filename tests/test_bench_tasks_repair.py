"""Contract tests for the repair benchmark task family.

Each task shows a broken module plus the failing symptom and asks for the
repaired module only. The scorer (`repair_green`) runs the TASK's own red
test (``scorer_args.test_code``) against the reply in the sandbox; the
test is red on the broken module and green on a correct fix. The sandbox
runs plain ``python``, so each test file calls its ``test_`` functions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from test_bench_scoring import requires_docker

from router.bench.schema import Task, load_tasks
from router.bench.scoring import score

TASKS_PATH = Path(__file__).parent.parent / "router/bench/tasks/repair.json"
_FENCE = re.compile(r"```python\s*\n(.*?)```", re.S)

# Hand-written correct repair per task (what a competent reply contains).
FIXES: dict[str, str] = {
    "repair-fizzbuzz": (
        "def fizzbuzz(n):\n"
        "    if n % 15 == 0:\n"
        "        return 'FizzBuzz'\n"
        "    if n % 3 == 0:\n"
        "        return 'Fizz'\n"
        "    if n % 5 == 0:\n"
        "        return 'Buzz'\n"
        "    return str(n)\n"
    ),
    "repair-running-total": (
        "def running_total(values):\n"
        "    totals = []\n"
        "    acc = 0\n"
        "    for v in values:\n"
        "        acc += v\n"
        "        totals.append(acc)\n"
        "    return totals\n"
    ),
    "repair-inventory": (
        "class Inventory:\n"
        "    def __init__(self):\n"
        "        self._stock = {}\n\n"
        "    def add(self, name, qty):\n"
        "        self._stock[name] = self._stock.get(name, 0) + qty\n\n"
        "    def remove(self, name, qty):\n"
        "        have = self._stock.get(name, 0)\n"
        "        if qty > have:\n"
        "            raise ValueError(f'insufficient {name}: {have} < {qty}')\n"
        "        self._stock[name] = have - qty\n\n"
        "    def count(self, name):\n"
        "        return self._stock.get(name, 0)\n"
    ),
}


def _load() -> list[Task]:
    return load_tasks(json.loads(TASKS_PATH.read_text(encoding="utf-8")))


def _broken_module(task: Task) -> str:
    fence = _FENCE.search(task.prompt)
    assert fence, task.task_id
    return fence.group(1)


def test_family_shape() -> None:
    tasks = _load()
    assert len(tasks) == 3
    assert len({t.task_id for t in tasks}) == 3
    for t in tasks:
        assert t.task_id.startswith("repair-")
        assert t.role == "repair"
        assert t.scorer == "repair_green"
        assert t.max_tokens <= 512
        assert set(t.scorer_args) == {"test_code", "module_name"}
        assert t.scorer_args["module_name"] == "target"
        test_code = t.scorer_args["test_code"]
        assert test_code.strip() and "def test_" in test_code
        assert "import pytest" not in test_code
        assert "from target import" in test_code


def test_prompt_shows_broken_module_and_hides_answer() -> None:
    for t in _load():
        broken = _broken_module(t)
        assert broken.strip() and broken != FIXES[t.task_id]
        assert t.scorer_args["test_code"] not in t.prompt
        assert "def test_" not in t.prompt


def test_fix_keys_match_task_ids() -> None:
    assert {t.task_id for t in _load()} == set(FIXES)


@requires_docker
@pytest.mark.parametrize("task_id", sorted(FIXES))
def test_red_test_is_red_on_broken_green_on_fix(task_id: str) -> None:
    """End-to-end proof through the real docker sandbox: the task's test
    fails on the module shown in the prompt and passes on the fix."""
    task = next(t for t in _load() if t.task_id == task_id)
    assert score(task.scorer, _broken_module(task), task.scorer_args) is False
    reply = "Fixed:\n```python\n" + FIXES[task_id] + "```\n"
    assert score(task.scorer, reply, task.scorer_args) is True
