"""Contract tests for the triage benchmark task family."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from router.bench.schema import Task, load_tasks
from router.bench.scoring import score

TASKS_PATH = Path(__file__).parent.parent / "router" / "bench" / "tasks" / "triage.json"

# Hand-written replies per task: one that ranks the buggy file first,
# one that names a decoy instead.
REPLIES: dict[str, tuple[str, str]] = {
    "triage-ledger-keyerror": (
        "billing/ledger.py — balance_for reads self._by_account directly, so "
        "add_entry is likely keying accounts differently (or not populating "
        "the index). billing/cli.py just forwards the argument.",
        "billing/cli.py — the argument parsing probably mangles the account "
        "name before it reaches billing/ledger.py.",
    ),
    "triage-render-typeerror": (
        "invoicing/render.py — line 27 concatenates an int onto a string "
        "without formatting it; total_cents being an int is correct per "
        "invoicing/models.py, so the render code must convert it.",
        "invoicing/models.py — total_cents should have been declared as a "
        "string so invoicing/render.py could concatenate it.",
    ),
    "triage-cli-offbyone": (
        "pantry/cli.py — the slice items[1:limit] starts at index 1, "
        "skipping the soonest-expiring item; it should be items[:limit]. "
        "pantry/store.py is verified correct.",
        "pantry/store.py — the sort in items() probably drops the first "
        "element before pantry/cli.py ever sees it.",
    ),
}


def _load() -> list[Task]:
    with TASKS_PATH.open() as f:
        return load_tasks(json.load(f))


def test_family_shape() -> None:
    tasks = _load()
    assert len(tasks) == 3
    assert all(t.role == "triage" for t in tasks)
    assert all(t.task_id.startswith("triage-") for t in tasks)
    assert len({t.task_id for t in tasks}) == 3
    assert all(t.scorer == "file_rank_top1" for t in tasks)
    assert all(t.max_tokens <= 512 for t in tasks)


def test_every_task_has_replies() -> None:
    assert {t.task_id for t in _load()} == set(REPLIES)


@pytest.mark.parametrize("task_id", sorted(REPLIES))
def test_scorer_passes_on_correct_reply(task_id: str) -> None:
    task = next(t for t in _load() if t.task_id == task_id)
    correct, _ = REPLIES[task_id]
    assert score(task.scorer, correct, task.scorer_args) is True


@pytest.mark.parametrize("task_id", sorted(REPLIES))
def test_scorer_fails_on_wrong_reply(task_id: str) -> None:
    task = next(t for t in _load() if t.task_id == task_id)
    _, wrong = REPLIES[task_id]
    assert score(task.scorer, wrong, task.scorer_args) is False
