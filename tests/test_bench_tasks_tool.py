"""Contract tests for the tool-correctness benchmark task family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from router.bench.schema import Task, load_tasks
from router.bench.scoring import score

TASKS_PATH = Path(__file__).parent.parent / "router/bench/tasks/tool_correctness.json"


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    raw = json.loads(TASKS_PATH.read_text())
    return load_tasks(raw)


def _correct_payload(task: Task) -> list[dict[str, Any]]:
    args = dict.fromkeys(task.scorer_args["required_args"], "v")
    return [
        {
            "function": {
                "name": task.scorer_args["name"],
                "arguments": json.dumps(args),
            }
        }
    ]


def test_family_shape(tasks: list[Task]) -> None:
    assert len(tasks) == 3
    assert all(t.role == "tool-correctness" for t in tasks)
    assert all(t.task_id.startswith("tool-") for t in tasks)
    assert len({t.task_id for t in tasks}) == 3
    assert all(t.max_tokens <= 512 for t in tasks)


def test_every_task_declares_one_tool(tasks: list[Task]) -> None:
    for task in tasks:
        assert task.tools is not None and len(task.tools) == 1
        fn = task.tools[0]["function"]
        assert task.tools[0]["type"] == "function"
        assert fn["name"] == task.scorer_args["name"]
        required = fn["parameters"]["required"]
        assert set(required) == set(task.scorer_args["required_args"])


def test_correct_tool_call_passes(tasks: list[Task]) -> None:
    for task in tasks:
        payload = _correct_payload(task)
        assert score(task.scorer, payload, task.scorer_args) is True


def test_wrong_tool_name_fails(tasks: list[Task]) -> None:
    for task in tasks:
        payload = _correct_payload(task)
        payload[0]["function"]["name"] = "unrelated_tool"
        assert score(task.scorer, payload, task.scorer_args) is False


def test_unparseable_arguments_fail(tasks: list[Task]) -> None:
    for task in tasks:
        payload = _correct_payload(task)
        payload[0]["function"]["arguments"] = "{not json"
        assert score(task.scorer, payload, task.scorer_args) is False
