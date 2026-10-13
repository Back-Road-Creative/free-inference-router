"""Contract tests for router.bench.schema + router.bench.scoring."""

from __future__ import annotations

from typing import Any

import pytest

from router.bench.sandbox import SandboxResult, SandboxUnavailableError
from router.bench.schema import ROLES, Task, load_tasks
from router.bench.scoring import (
    EXEC_SCORERS,
    SCORERS,
    TOOL_CALL_SCORERS,
    contains_all,
    contains_none,
    exact_word,
    file_rank_top1,
    json_object_with,
    mutation_kill,
    repair_green,
    score,
    tool_call_named,
)

# ------------------------------------------------------------------ schema


def _task(**over: Any) -> Task:
    base: dict[str, Any] = {
        "task_id": "t1",
        "role": "triage",
        "prompt": "p",
        "scorer": "exact_word",
        "scorer_args": {"word": "pong"},
    }
    base.update(over)
    return Task(**base)


def test_task_valid_roundtrip() -> None:
    t = _task()
    assert t.role in ROLES and t.scorer in SCORERS


def test_task_unknown_role_rejected() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        _task(role="vibes")


def test_task_unknown_scorer_rejected_at_load_not_run() -> None:
    with pytest.raises(ValueError, match="unknown scorer"):
        _task(scorer="exact_wrod")


def test_task_empty_prompt_rejected() -> None:
    with pytest.raises(ValueError):
        _task(prompt="")


def test_load_tasks_rejects_duplicate_ids() -> None:
    raw = [
        {
            "task_id": "a",
            "role": "triage",
            "prompt": "p",
            "scorer": "exact_word",
            "scorer_args": {"word": "x"},
        },
        {
            "task_id": "a",
            "role": "triage",
            "prompt": "q",
            "scorer": "exact_word",
            "scorer_args": {"word": "y"},
        },
    ]
    with pytest.raises(ValueError, match="duplicate"):
        load_tasks(raw)


def test_every_role_is_a_known_string() -> None:
    assert len(ROLES) == 7 and len(set(ROLES)) == 7


# ----------------------------------------------------------------- scorers


def test_exact_word_trims_and_case_folds() -> None:
    assert exact_word("  Pong.! ", word="pong")
    assert not exact_word("ping pong", word="pong")


def test_contains_all_and_none() -> None:
    assert contains_all("Alpha then beta", needles=["alpha", "BETA"])
    assert not contains_all("alpha only", needles=["alpha", "beta"])
    assert contains_none("clean reply", needles=["SECRET", "sk-"])
    assert not contains_none("here is sk-abc", needles=["sk-"])


def test_json_object_with_plain_and_fenced() -> None:
    assert json_object_with('{"answer": 42}', required_keys=["answer"])
    fenced = 'sure:\n```json\n{"answer": 42}\n```'
    assert json_object_with(fenced, required_keys=["answer"])
    assert not json_object_with("not json", required_keys=["answer"])
    assert not json_object_with('{"other": 1}', required_keys=["answer"])
    assert not json_object_with("[1,2]", required_keys=["answer"])


def test_file_rank_top1_orders_expected_before_decoys() -> None:
    reply = "Most likely: pkg/parser.py, then pkg/cli.py"
    assert file_rank_top1(reply, expected_file="pkg/parser.py", decoys=["pkg/cli.py"])
    assert not file_rank_top1(
        reply, expected_file="pkg/cli.py", decoys=["pkg/parser.py"]
    )
    assert not file_rank_top1("no files here", expected_file="pkg/parser.py", decoys=[])


def test_tool_call_named_happy_and_sad_paths() -> None:
    good = [{"function": {"name": "run_tests", "arguments": '{"path": "tests/"}'}}]
    assert tool_call_named(good, name="run_tests", required_args=["path"])
    assert not tool_call_named([], name="run_tests", required_args=["path"])
    assert not tool_call_named(None, name="run_tests", required_args=["path"])
    wrong_name = [{"function": {"name": "rm", "arguments": "{}"}}]
    assert not tool_call_named(wrong_name, name="run_tests", required_args=[])
    bad_json = [{"function": {"name": "run_tests", "arguments": "{path"}}]
    assert not tool_call_named(bad_json, name="run_tests", required_args=["path"])


def test_score_dispatch_unknown_scorer_raises() -> None:
    with pytest.raises(KeyError):
        score("nope", "reply", {})


def test_score_dispatch_runs_registered_scorer() -> None:
    assert score("exact_word", "pong", {"word": "pong"}) is True


def test_tool_call_scorers_subset_of_registry() -> None:
    assert TOOL_CALL_SCORERS <= set(SCORERS)


def test_exec_scorers_registered() -> None:
    """The disposable sandbox landed, so the exec-scorers exist and are
    reachable through the same registry every other scorer uses."""
    assert {"mutation_kill", "repair_green"} <= set(SCORERS)
    assert EXEC_SCORERS <= set(SCORERS)


def test_contains_all_normalizes_unicode_spaces_and_hyphens() -> None:
    """A faithful answer with typographic variants must pass: U+202F narrow
    no-break space in '45 seconds', U+2011 non-breaking hyphen in a flag."""
    reply = "Use ‑‑retry‑limit; default is 45 seconds."
    assert contains_all(reply, needles=["--retry-limit", "45 seconds"])


def test_contains_none_normalizes_before_matching() -> None:
    assert not contains_none("BAIT‑TOKEN‑X", needles=["bait-token-x"])


# ------------------------------------------------------------ exec-scorers

_PRISTINE = "def add(a, b):\n    return a + b\n"
_MUTANT = "def add(a, b):\n    return a - b\n"
_KILLING_TEST = "from target import add\nassert add(2, 3) == 5\n"
_WEAK_TEST = "from target import add\nassert add(0, 0) == 0\n"
_RED_TEST = "from target import add\nassert add(2, 3) == 5\n"
_BROKEN_MODULE = "def add(a, b):\n    return a * b\n"


def _docker_usable() -> bool:
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_usable(),
    reason=(
        "docker unavailable — exec-scorer behavior proofs did NOT execute; "
        "they must run green where exec-scorers will run"
    ),
)


def test_mutation_kill_refuses_without_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(SandboxUnavailableError):
        mutation_kill(_KILLING_TEST, pristine_module=_PRISTINE, mutant_module=_MUTANT)


def test_repair_green_refuses_without_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(SandboxUnavailableError):
        repair_green(_PRISTINE, test_code=_RED_TEST)


def test_mutation_kill_refuses_identical_mutant() -> None:
    """A task whose mutant equals the pristine module cannot be killed —
    scoring it would be a fake gate, so it is a task bug, not a zero."""
    with pytest.raises(ValueError, match="identical"):
        mutation_kill(_KILLING_TEST, pristine_module=_PRISTINE, mutant_module=_PRISTINE)


def test_repair_green_refuses_absent_test_code() -> None:
    with pytest.raises(ValueError, match="test"):
        repair_green(_PRISTINE, test_code="   ")


def test_repair_green_reply_shipping_own_tests_fails() -> None:
    """The task's test is the only judge — a reply that carries test
    definitions altered/replaced the test and scores fail outright."""
    tampered = _PRISTINE + "\ndef test_add():\n    assert True\n"
    assert repair_green(tampered, test_code=_RED_TEST) is False


@pytest.mark.parametrize(
    ("on_mutant", "killed_mutant"),
    [
        (SandboxResult(1, "", "AssertionError"), True),
        (SandboxResult(124, "", "", timed_out=True), False),
        (SandboxResult(137, "", ""), False),
    ],
    ids=["test-failed", "time-limit", "sigkill"],
)
def test_mutation_kill_credits_only_the_tests_own_failure(
    monkeypatch: pytest.MonkeyPatch, on_mutant: SandboxResult, killed_mutant: bool
) -> None:
    """Only the test failing on its own kills the mutant. The sandbox
    stopping it (the time limit, or KILL: TERM ignored past the grace, or
    the memory bound) says nothing about the mutant."""

    def fake_run(code: str, **kwargs: Any) -> SandboxResult:
        if kwargs["files"]["target.py"] == _PRISTINE:
            return SandboxResult(0, "", "")
        return on_mutant

    monkeypatch.setattr("router.bench.scoring.run_snippet", fake_run)
    assert (
        mutation_kill(_KILLING_TEST, pristine_module=_PRISTINE, mutant_module=_MUTANT)
        is killed_mutant
    )


@requires_docker
def test_mutation_kill_good_test_kills_mutant() -> None:
    assert (
        mutation_kill(_KILLING_TEST, pristine_module=_PRISTINE, mutant_module=_MUTANT)
        is True
    )


@requires_docker
def test_mutation_kill_weak_test_survives_mutant() -> None:
    assert (
        mutation_kill(_WEAK_TEST, pristine_module=_PRISTINE, mutant_module=_MUTANT)
        is False
    )


@requires_docker
def test_mutation_kill_test_red_on_pristine_fails() -> None:
    red_on_pristine = "from target import add\nassert add(2, 3) == 6\n"
    assert (
        mutation_kill(red_on_pristine, pristine_module=_PRISTINE, mutant_module=_MUTANT)
        is False
    )


@requires_docker
def test_repair_green_fixed_module_passes_fenced() -> None:
    reply = "Here is the fix:\n```python\n" + _PRISTINE + "```\n"
    assert repair_green(reply, test_code=_RED_TEST) is True


@requires_docker
def test_repair_green_still_broken_module_fails() -> None:
    assert repair_green(_BROKEN_MODULE, test_code=_RED_TEST) is False
