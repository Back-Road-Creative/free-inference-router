"""Retry classification and request-shape validation.

Enforces docs/POLICY.md:45-52: retry only on 429, a pre-tool-call timeout,
transport failure, or a retryable 5xx (502/503/504); hard stop on 401/403,
unknown/non-zero price, invalid tool schema, context overflow,
classification denial, or malformed output after a tool has run. A
returned ``model_id`` differing from the request is a hard stop plus
quarantine. Once a tool has executed, a model switch is a new attempt from
a clean disposable worktree, never a resumed session — this module only
classifies the outcome; it does not perform the switch.

``classify`` also returns ``Action.ACCEPT`` for a clean success (no
exception, a missing or 2xx status, matching model, valid schema, known
zero price, no overflow/denial/malformed output) — every hard-stop and
quarantine check runs first, so ACCEPT is only reachable once none of
them fired. An outcome that matches neither a known retry/hard-stop
pattern nor a clean success still fails closed to HARD_STOP.

Also enforces Plan §5: provider retry budgets are 0 or 1, nested fallback
fields in a request/config are rejected, and OpenRouter gets one explicit
model ID, never ``openrouter/free`` or a bare ``:free`` wildcard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

_RETRYABLE_5XX = {502, 503, 504}
_HARD_STOP_STATUS = {401, 403, 500}
_RETRYABLE_TRANSIENT_EXCEPTIONS = {"timeout", "transport"}


class Action(Enum):
    ACCEPT = auto()
    RETRY = auto()
    HARD_STOP = auto()
    HARD_STOP_QUARANTINE = auto()


@dataclass(frozen=True)
class Outcome:
    status_code: int | None
    exception_kind: str | None
    tools_executed: bool
    requested_model: str
    returned_model: str | None
    price_known: bool
    price_zero: bool
    schema_valid: bool
    context_overflow: bool
    classification_denied: bool
    output_malformed: bool


def classify(outcome: Outcome) -> Action:
    # Belt: a model substitution is always a hard stop plus quarantine,
    # regardless of any other field (docs/THREAT-MODEL.md:14-17).
    if (
        outcome.returned_model is not None
        and outcome.returned_model != outcome.requested_model
    ):
        return Action.HARD_STOP_QUARANTINE

    if outcome.classification_denied:
        return Action.HARD_STOP
    if outcome.context_overflow:
        return Action.HARD_STOP
    if not outcome.schema_valid:
        return Action.HARD_STOP
    if not outcome.price_known:
        return Action.HARD_STOP
    if not outcome.price_zero:
        return Action.HARD_STOP
    if outcome.output_malformed and outcome.tools_executed:
        return Action.HARD_STOP

    if outcome.status_code is not None:
        if outcome.status_code in _HARD_STOP_STATUS:
            return Action.HARD_STOP
        if outcome.status_code == 429 or outcome.status_code in _RETRYABLE_5XX:
            return Action.RETRY
        if outcome.status_code == 408:
            return Action.HARD_STOP if outcome.tools_executed else Action.RETRY

    if outcome.exception_kind in _RETRYABLE_TRANSIENT_EXCEPTIONS:
        return Action.HARD_STOP if outcome.tools_executed else Action.RETRY

    # Accept only once every hard-stop and quarantine rule above has
    # cleared: no exception, a missing or 2xx status, matching model,
    # valid schema, known zero price, no overflow/denial/malformed output.
    status_ok = outcome.status_code is None or 200 <= outcome.status_code < 300
    if status_ok and outcome.exception_kind is None:
        return Action.ACCEPT

    # Fail closed: an outcome that tripped none of the known retry paths,
    # none of the known hard-stop conditions, and isn't a clean success is
    # unrecognized, not safe.
    return Action.HARD_STOP


_NESTED_FALLBACK_KEYS = ("models", "fallbacks", "route")


def validate_request_shape(payload: dict[str, Any]) -> None:
    """Reject nested fallback fields and an over-budget retry_budget.

    Plan §5: no ``models``/``fallbacks``/``route`` list, no
    ``provider.order``, no literal ``openrouter/free``, no ``:free``
    wildcard without a concrete model before it, and ``retry_budget`` must
    be 0 or 1.
    """
    for key in _NESTED_FALLBACK_KEYS:
        if key in payload:
            raise ValueError(f"nested fallback field not allowed: {key!r}")

    provider = payload.get("provider")
    if isinstance(provider, dict) and "order" in provider:
        raise ValueError("nested fallback field not allowed: provider.order")

    model = payload.get("model")
    if isinstance(model, str):
        if model == "openrouter/free":
            raise ValueError("openrouter/free is not a concrete model id")
        if model.endswith(":free"):
            prefix = model[: -len(":free")]
            if not prefix or "*" in prefix:
                raise ValueError(":free wildcard requires a concrete model before it")

    retry_budget = payload.get("retry_budget", 0)
    if retry_budget is not None and retry_budget > 1:
        raise ValueError("retry_budget must be 0 or 1")
