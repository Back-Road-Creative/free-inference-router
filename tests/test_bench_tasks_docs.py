"""Contract tests for the docs-coherence benchmark task family.

Each task embeds a synthetic reference document and asks a factual question
whose faithful answer must repeat exact strings from the document (the
needles). A hallucinated answer that invents flags or values fails scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from router.bench.schema import Task, load_tasks
from router.bench.scoring import score

TASKS_PATH = (
    Path(__file__).parent.parent / "router" / "bench" / "tasks" / "docs_coherence.json"
)

# Hand-written answer pairs per task: a faithful answer that sticks to the
# document, and a plausible-but-wrong answer that misstates flags or values.
ANSWERS: dict[str, tuple[str, str]] = {
    "docs-driftctl-retry": (
        "Retries are controlled by --retry-limit (default 4 attempts) and "
        "--retry-backoff, a fixed pause of 45 seconds between attempts; "
        "there is no exponential backoff.",
        "driftctl retries failed transfers up to 3 times, controlled by "
        "--max-retries, with an exponential backoff starting at 30s.",
    ),
    "docs-packmule-bundle": (
        "By default packmule compresses with zstd at level 6, splits output "
        "once a bundle reaches 250 MB, and deduplication is disabled by "
        "default because it needs a store-wide index.",
        "packmule defaults to lz4 compression, splits bundles at 1 GB, and "
        "deduplication is enabled out of the box.",
    ),
    "docs-lumberd-rotation": (
        "The rotate_every setting controls the schedule and defaults to "
        "12 hours; once a file is rotated out it is compressed with gzip "
        "at level 5.",
        "lumberd rotates daily via the rotation_interval setting (default "
        "24h) and compresses old files with zstd.",
    ),
}


def _load_raw() -> list[dict[str, object]]:
    with TASKS_PATH.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    assert isinstance(raw, list)
    return raw


def _load() -> list[Task]:
    return load_tasks(_load_raw())


def test_family_shape() -> None:
    tasks = _load()
    assert len(tasks) == 3
    ids = [t.task_id for t in tasks]
    assert len(set(ids)) == 3
    for task in tasks:
        assert task.task_id.startswith("docs-")
        assert task.role == "docs-coherence"
        assert task.scorer == "contains_all"
        assert task.max_tokens <= 512
        needles = task.scorer_args["needles"]
        assert isinstance(needles, list) and needles
        assert 2 <= len(needles) <= 4


def test_answer_keys_match_task_ids() -> None:
    assert {t.task_id for t in _load()} == set(ANSWERS)


@pytest.mark.parametrize("task_id", sorted(ANSWERS))
def test_faithful_answer_passes(task_id: str) -> None:
    task = next(t for t in _load() if t.task_id == task_id)
    faithful, _ = ANSWERS[task_id]
    assert score(task.scorer, faithful, task.scorer_args) is True


@pytest.mark.parametrize("task_id", sorted(ANSWERS))
def test_hallucinated_answer_fails(task_id: str) -> None:
    task = next(t for t in _load() if t.task_id == task_id)
    _, hallucinated = ANSWERS[task_id]
    assert score(task.scorer, hallucinated, task.scorer_args) is False


@pytest.mark.parametrize("task_id", sorted(ANSWERS))
def test_needles_are_extractive(task_id: str) -> None:
    """Every needle appears verbatim in the prompt's embedded document,
    so a correct answer is extraction, not outside knowledge."""
    task = next(t for t in _load() if t.task_id == task_id)
    for needle in task.scorer_args["needles"]:
        assert needle in task.prompt, f"needle {needle!r} not in prompt"
