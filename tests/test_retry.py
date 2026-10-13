"""Tests for router.retry — outcome classification and request-shape guard.

Binding rules: docs/POLICY.md:45-52 (retry rules, model-mismatch hard-stop
plus quarantine) and Plan §5 (nested fallback fields rejected, one explicit
OpenRouter model ID, retry budget 0 or 1).
"""

from __future__ import annotations

import pytest

from router.retry import Action, Outcome, classify, validate_request_shape


def _outcome(**overrides: object) -> Outcome:
    base: dict[str, object] = {
        "status_code": None,
        "exception_kind": None,
        "tools_executed": False,
        "requested_model": "openai/gpt-oss-120b",
        "returned_model": "openai/gpt-oss-120b",
        "price_known": True,
        "price_zero": True,
        "schema_valid": True,
        "context_overflow": False,
        "classification_denied": False,
        "output_malformed": False,
    }
    base.update(overrides)
    return Outcome(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"status_code": 429}, Action.RETRY),
        ({"status_code": 408, "exception_kind": "timeout"}, Action.RETRY),
        (
            {
                "status_code": 408,
                "exception_kind": "timeout",
                "tools_executed": True,
            },
            Action.HARD_STOP,
        ),
        ({"status_code": 502}, Action.RETRY),
        ({"status_code": 503}, Action.RETRY),
        ({"status_code": 504}, Action.RETRY),
        ({"status_code": 500}, Action.HARD_STOP),
        ({"status_code": 401}, Action.HARD_STOP),
        ({"status_code": 403}, Action.HARD_STOP),
        ({"price_known": False}, Action.HARD_STOP),
        ({"price_zero": False}, Action.HARD_STOP),
        ({"schema_valid": False}, Action.HARD_STOP),
        ({"context_overflow": True}, Action.HARD_STOP),
        ({"classification_denied": True}, Action.HARD_STOP),
        (
            {"output_malformed": True, "tools_executed": True},
            Action.HARD_STOP,
        ),
    ],
)
def test_classify_contract_table(
    overrides: dict[str, object], expected: Action
) -> None:
    """Contract tests 3 and 4 — the full retry/hard-stop decision table."""
    assert classify(_outcome(**overrides)) is expected


def test_clean_success_is_accepted() -> None:
    """A 200 with matching model, valid schema, zero price, and
    well-formed output must be distinguishable from a hard stop."""
    outcome = _outcome(status_code=200)
    assert classify(outcome) is Action.ACCEPT


def test_clean_success_with_no_status_code_is_accepted() -> None:
    outcome = _outcome(status_code=None)
    assert classify(outcome) is Action.ACCEPT


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"returned_model": "openai/gpt-oss-20b"},
            Action.HARD_STOP_QUARANTINE,
        ),
        ({"price_known": False}, Action.HARD_STOP),
        ({"price_zero": False}, Action.HARD_STOP),
        ({"schema_valid": False}, Action.HARD_STOP),
        ({"context_overflow": True}, Action.HARD_STOP),
        ({"classification_denied": True}, Action.HARD_STOP),
        (
            {"output_malformed": True, "tools_executed": True},
            Action.HARD_STOP,
        ),
        ({"status_code": 401}, Action.HARD_STOP),
    ],
)
def test_clean_success_with_one_field_flipped_is_not_accepted(
    overrides: dict[str, object], expected: Action
) -> None:
    merged: dict[str, object] = {"status_code": 200, **overrides}
    outcome = _outcome(**merged)
    result = classify(outcome)
    assert result is not Action.ACCEPT
    assert result is expected


def test_200_with_model_mismatch_still_quarantines() -> None:
    """A 2xx status must not shortcut past the quarantine check."""
    outcome = _outcome(
        status_code=200,
        requested_model="openai/gpt-oss-120b",
        returned_model="openai/gpt-oss-20b",
    )
    assert classify(outcome) is Action.HARD_STOP_QUARANTINE


def test_transport_failure_before_tools_retries() -> None:
    outcome = _outcome(exception_kind="transport", tools_executed=False)
    assert classify(outcome) is Action.RETRY


def test_transport_failure_after_tools_hard_stops() -> None:
    outcome = _outcome(exception_kind="transport", tools_executed=True)
    assert classify(outcome) is Action.HARD_STOP


def test_returned_model_mismatch_hard_stops_and_quarantines() -> None:
    """Contract test 15."""
    outcome = _outcome(
        requested_model="openai/gpt-oss-120b",
        returned_model="openai/gpt-oss-20b",
    )
    assert classify(outcome) is Action.HARD_STOP_QUARANTINE


def test_returned_model_equal_is_not_quarantined() -> None:
    outcome = _outcome(
        requested_model="openai/gpt-oss-120b",
        returned_model="openai/gpt-oss-120b",
    )
    assert classify(outcome) is not Action.HARD_STOP_QUARANTINE


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "openai/gpt-oss-120b", "models": ["a", "b"]},
        {"model": "openai/gpt-oss-120b", "fallbacks": ["a"]},
        {"model": "openai/gpt-oss-120b", "route": "fallback"},
        {"model": "openai/gpt-oss-120b", "provider": {"order": ["a", "b"]}},
        {"model": "openrouter/free"},
        {"model": ":free"},
        {"model": "*:free"},
        {"model": "openai/gpt-oss-120b", "retry_budget": 2},
    ],
)
def test_validate_request_shape_rejects(payload: dict[str, object]) -> None:
    """Contract test 9 — nested fallback shapes and retry_budget > 1."""
    with pytest.raises(ValueError):
        validate_request_shape(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "openai/gpt-oss-120b"},
        {"model": "openai/gpt-oss-120b", "retry_budget": 0},
        {"model": "openai/gpt-oss-120b", "retry_budget": 1},
        {"model": "meta-llama/llama-3-8b-instruct:free", "retry_budget": 1},
    ],
)
def test_validate_request_shape_accepts_clean_request(
    payload: dict[str, object],
) -> None:
    validate_request_shape(payload)


def test_openrouter_explicit_model_free_suffix_pinned() -> None:
    """Pins the interpretation: a concrete ``org/model:free`` id IS the
    explicit-model form OpenRouter uses and must pass, while the literal
    ``openrouter/free`` and any bare/wildcard ``:free`` must not."""
    validate_request_shape({"model": "meta-llama/llama-3.3-70b-instruct:free"})
    with pytest.raises(ValueError):
        validate_request_shape({"model": "openrouter/free"})
    with pytest.raises(ValueError):
        validate_request_shape({"model": ":free"})
    with pytest.raises(ValueError):
        validate_request_shape({"model": "*:free"})
