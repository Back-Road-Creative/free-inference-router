"""Deterministic scorers for benchmark replies.

Every scorer is a function (reply, **args) -> bool. No model ever
judges a reply. Most scorers are pure text checks; the exec-scorers
(mutation_kill, repair_green) run model-generated code, and do so ONLY
through router.bench.sandbox.run_snippet — the disposable locked-down
container. When the sandbox is unavailable they raise
SandboxUnavailableError rather than pass or skip: a scorer that cannot
execute is not a gate. The registry SCORERS map is the single lookup
used by schema.Task validation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from typing import Any

from router.bench.sandbox import run_snippet

# Models emit typographic variants of ASCII: narrow no-break spaces
# (U+202F) inside "45 seconds", non-breaking hyphens (U+2011) inside flag
# names. A byte-literal needle match fails those *faithful* answers — seen
# live 2026-08-31 (groq gpt-oss-120b scored 0/9 on docs-coherence while
# giving correct answers). Normalise both sides before matching.
_SPACES = {0x00A0: " ", 0x202F: " ", 0x2009: " ", 0x2007: " "}
_HYPHENS = {0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-"}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.translate({**_SPACES, **_HYPHENS}).lower()


def exact_word(reply: str, *, word: str) -> bool:
    """Reply is exactly the expected word (whitespace/punctuation trimmed)."""
    return reply.strip().strip(".!\"'`").lower() == word.lower()


def contains_all(reply: str, *, needles: list[str]) -> bool:
    low = _norm(reply)
    return all(_norm(n) in low for n in needles)


def contains_none(reply: str, *, needles: list[str]) -> bool:
    """Pass only if none of the forbidden strings appear (resistance role)."""
    low = _norm(reply)
    return not any(_norm(n) in low for n in needles)


def json_object_with(reply: str, *, required_keys: list[str]) -> bool:
    """Reply parses as a JSON object carrying every required key.

    Tolerates a fenced ```json block, since chat models often wrap output.
    """
    text = reply.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except ValueError:
        return False
    return isinstance(obj, dict) and all(k in obj for k in required_keys)


def file_rank_top1(reply: str, *, expected_file: str, decoys: list[str]) -> bool:
    """The expected file is named BEFORE any decoy file (triage ranking)."""
    pos = reply.find(expected_file)
    if pos < 0:
        return False
    decoy_positions = [reply.find(d) for d in decoys]
    return all(p < 0 or pos < p for p in decoy_positions)


def tool_call_named(
    reply_tool_calls: Any, *, name: str, required_args: list[str]
) -> bool:
    """First tool call targets the named tool with parseable args carrying
    every required key. `reply_tool_calls` is the raw tool_calls list from
    the chat response (not text) — the runner passes it through."""
    if not isinstance(reply_tool_calls, list) or not reply_tool_calls:
        return False
    call = reply_tool_calls[0]
    try:
        fn = call["function"]
        if fn["name"] != name:
            return False
        args = json.loads(fn["arguments"])
    except (KeyError, TypeError, ValueError):
        return False
    return isinstance(args, dict) and all(k in args for k in required_args)


# --------------------------------------------------------------- exec-scorers

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def _extract_code(reply: str) -> str:
    """Model code often arrives inside a ```python fence — use the fenced
    block when one exists, the raw reply otherwise."""
    fence = _CODE_FENCE.search(reply)
    return (fence.group(1) if fence else reply).strip()


def _module_filename(module_name: str) -> str:
    if not module_name.isidentifier():
        raise ValueError(f"module_name must be an identifier: {module_name!r}")
    return f"{module_name}.py"


def mutation_kill(
    reply: str,
    *,
    pristine_module: str,
    mutant_module: str,
    module_name: str = "target",
) -> bool:
    """Test-gen role: the reply is generated test code. It passes only if
    it runs green against the pristine module AND fails on its own (not
    stopped by the time limit or KILL) against the planted mutant — i.e.
    the mutation is killed. Raises SandboxUnavailableError when docker
    cannot run anything."""
    if not pristine_module.strip() or not mutant_module.strip():
        raise ValueError("mutation_kill requires pristine and mutant source")
    if pristine_module == mutant_module:
        raise ValueError("mutant is identical to pristine — unkillable, task is broken")
    test_code = _extract_code(reply)
    if not test_code:
        return False
    fname = _module_filename(module_name)
    on_pristine = run_snippet(test_code, files={fname: pristine_module})
    if on_pristine.exit_code != 0 or on_pristine.timed_out:
        return False
    on_mutant = run_snippet(test_code, files={fname: mutant_module})
    # A hang or a memory blow-up on the mutant is not a test that caught it.
    return on_mutant.exit_code != 0 and not on_mutant.timed_out and not on_mutant.killed


def repair_green(
    reply: str,
    *,
    test_code: str,
    module_name: str = "target",
) -> bool:
    """Repair role: the reply is the repaired module. The TASK's own red
    test (never anything from the reply) runs against it in the sandbox;
    pass only if it goes green. A reply carrying test definitions altered
    or replaced the test and fails outright. Raises ValueError when the
    task's test source is absent, SandboxUnavailableError without docker."""
    if not test_code.strip():
        raise ValueError(
            "repair_green requires the task's own test source — "
            "refusing to score without it"
        )
    module_src = _extract_code(reply)
    if not module_src:
        return False
    if re.search(r"^\s*def\s+test_", module_src, re.M):
        return False
    res = run_snippet(test_code, files={_module_filename(module_name): module_src})
    return res.exit_code == 0 and not res.timed_out


SCORERS: dict[str, Callable[..., bool]] = {
    "exact_word": exact_word,
    "contains_all": contains_all,
    "contains_none": contains_none,
    "json_object_with": json_object_with,
    "file_rank_top1": file_rank_top1,
    "tool_call_named": tool_call_named,
    "mutation_kill": mutation_kill,
    "repair_green": repair_green,
}

# Scorers judged on the structured tool_calls payload instead of reply text.
TOOL_CALL_SCORERS = frozenset({"tool_call_named"})

# Scorers that execute model output in the disposable sandbox. They raise
# SandboxUnavailableError instead of scoring when docker is unusable.
EXEC_SCORERS = frozenset({"mutation_kill", "repair_green"})


def score(scorer: str, payload: Any, args: dict[str, Any]) -> bool:
    """Dispatch. Unknown scorer raises — never a silent zero."""
    fn = SCORERS[scorer]
    result = fn(payload, **args)
    if not isinstance(result, bool):
        raise TypeError(f"scorer {scorer} returned {type(result).__name__}, want bool")
    return result
