"""Tests for router.selector — the deterministic lane selector.

Grounded in docs/POLICY.md:
- Lanes and task-shape -> lane mapping: POLICY.md:6-17.
- Data classification matrix (public/synthetic/private x
  no-training/training/unknown, unknown fails closed as training):
  POLICY.md:30-43.
- Groq 8k TPM bounded-artifact-only limit: POLICY.md:26-28.
- Default-deny classification.json: POLICY.md and router/classification.json.
"""

from __future__ import annotations

from typing import Any

import pytest

from router import load_providers
from router.selector import (
    Task,
    is_routable_to_r1,
    provider_eligible,
    repo_permits_remote,
    select_lane,
)

ALL_SHAPES = (
    "deterministic",
    "judgment",
    "bounded-diagnosis",
    "small-diff-review",
    "full-repo-write",
)


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "provider": "fixture-provider",
        "model_id": "fixture-model",
        "status": "admitted",
        "data_policy_class": "no-training",
        "card_required": "no",
        "price_usd_per_mtok_in": 0,
        "price_usd_per_mtok_out": 0,
        "rpm": 30,
        "rpd": 1000,
        "tpm": 100_000,
        "tpd": 200_000,
        "reset_boundary": None,
        "allowed_roles": ["diagnosis", "small-diff-review"],
        "max_attempts": 3,
        "timeout_s": 60,
        "last_smoke": None,
        "source_url": "https://example.invalid",
    }
    base.update(overrides)
    return base


def _classification(repo: str, **entry: Any) -> dict[str, Any]:
    return {"default": "deny", "repos": {repo: entry}}


def _task(**overrides: Any) -> Task:
    base: dict[str, Any] = {
        "shape": "bounded-diagnosis",
        "data_class": "public",
        "repo": "example-org/private-repo",
        "role": "diagnosis",
        "estimated_tokens": 500,
        "needs_write": False,
    }
    base.update(overrides)
    return Task(**base)


# ---------------------------------------------------------------- contract 1


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_private_task_never_selects_r1(shape: str) -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [_entry()]
    task = _task(shape=shape, data_class="private")
    decision = select_lane(task, classification, providers)
    assert decision.lane != "R1"
    assert decision.lane in {"D0", "L1", "C1"}


# ---------------------------------------------------------------- contract 2


def test_public_task_with_admitted_no_training_provider_selects_it() -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [_entry(provider="groq", model_id="openai/gpt-oss-120b")]
    task = _task(data_class="public", role="diagnosis")
    decision = select_lane(task, classification, providers)
    assert decision.lane == "R1"
    assert decision.eligible == (("groq", "openai/gpt-oss-120b"),)


def test_real_registry_selects_only_the_admitted_groq_pair_for_public() -> None:
    """Phase 2 (2026-08-31): groq's two models are admitted after a 14/14
    smoke. A public task may now reach R1, but ONLY those exact pairs."""
    classification = _classification("example-org/private-repo", remote=True)
    providers = load_providers()
    task = _task(data_class="public", role="diagnosis")
    decision = select_lane(task, classification, providers)
    assert set(decision.eligible) <= {
        ("groq", "openai/gpt-oss-120b"),
        ("groq", "openai/gpt-oss-20b"),
    }
    assert decision.eligible, (
        "admitted groq models should be eligible for a public diagnosis task"
    )


def test_real_registry_private_still_never_selects_r1() -> None:
    """The invariant that must survive every admission: private data never
    routes remotely, no matter what the registry admits."""
    classification = _classification("example-org/private-repo", remote=True)
    providers = load_providers()
    task = _task(data_class="private", role="diagnosis")
    decision = select_lane(task, classification, providers)
    assert decision.eligible == ()
    assert decision.lane != "R1"


# ------------------------------------------------------------ training class


def test_training_provider_public_not_routable() -> None:
    assert is_routable_to_r1("public", "training") is False


def test_training_provider_synthetic_benchmark_role_routable() -> None:
    entry = _entry(data_policy_class="training", allowed_roles=["synthetic-benchmark"])
    task = _task(data_class="synthetic", role="synthetic-benchmark")
    assert provider_eligible(entry, task) is True


def test_training_provider_synthetic_non_benchmark_role_not_routable() -> None:
    entry = _entry(data_policy_class="training", allowed_roles=["diagnosis"])
    task = _task(data_class="synthetic", role="diagnosis")
    assert provider_eligible(entry, task) is False


def test_unknown_provider_class_behaves_like_training() -> None:
    assert is_routable_to_r1("public", "unknown") is False
    assert is_routable_to_r1("public", "training") is False

    entry_unknown = _entry(
        data_policy_class="unknown", allowed_roles=["synthetic-benchmark"]
    )
    entry_training = _entry(
        data_policy_class="training", allowed_roles=["synthetic-benchmark"]
    )
    synthetic_task = _task(data_class="synthetic", role="synthetic-benchmark")
    assert provider_eligible(entry_unknown, synthetic_task) is True
    assert provider_eligible(entry_training, synthetic_task) is True

    bad_role_task = _task(data_class="synthetic", role="diagnosis")
    entry_unknown_bad = _entry(data_policy_class="unknown", allowed_roles=["diagnosis"])
    assert provider_eligible(entry_unknown_bad, bad_role_task) is False
    # unknown provider class fails closed as training: public is not routable
    assert is_routable_to_r1("public", "unknown") is False


# ------------------------------------------------------------------- repos


def test_unlisted_repo_not_routable_to_r1() -> None:
    classification = {"default": "deny", "repos": {}}
    assert repo_permits_remote(classification, "example-org/private-repo") is False


def test_repo_listed_remote_false_not_routable() -> None:
    classification = _classification("example-org/private-repo", remote=False)
    assert repo_permits_remote(classification, "example-org/private-repo") is False


def test_repo_listed_remote_true_routable() -> None:
    classification = _classification("example-org/private-repo", remote=True)
    assert repo_permits_remote(classification, "example-org/private-repo") is True


def test_unlisted_repo_falls_through_to_c1_not_r1() -> None:
    classification = {"default": "deny", "repos": {}}
    providers = [_entry()]
    task = _task(data_class="public", role="diagnosis")
    decision = select_lane(task, classification, providers)
    assert decision.lane == "C1"
    assert decision.eligible == ()


# ------------------------------------------------------------------- budget


def test_estimated_tokens_over_tpm_excludes_provider() -> None:
    # Groq's real binding limit: 8,000 TPM on the free gpt-oss models
    # (POLICY.md:26-28) -- bounded-artifact roles only, never whole-file work.
    entry = _entry(provider="groq", model_id="openai/gpt-oss-120b", tpm=8000)
    task = _task(data_class="public", role="diagnosis", estimated_tokens=9000)
    assert provider_eligible(entry, task) is False

    task_within_budget = _task(
        data_class="public", role="diagnosis", estimated_tokens=4000
    )
    assert provider_eligible(entry, task_within_budget) is True


def test_null_tpm_never_excludes_on_budget() -> None:
    entry = _entry(tpm=None)
    task = _task(data_class="public", role="diagnosis", estimated_tokens=10_000_000)
    assert provider_eligible(entry, task) is True


# ------------------------------------------------------------------ lanes


@pytest.mark.parametrize("data_class", ["public", "synthetic", "private"])
def test_deterministic_shape_always_d0(data_class: str) -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [_entry()]
    task = _task(shape="deterministic", data_class=data_class)
    decision = select_lane(task, classification, providers)
    assert decision.lane == "D0"


@pytest.mark.parametrize("data_class", ["public", "synthetic", "private"])
def test_judgment_shape_always_l1(data_class: str) -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [_entry()]
    task = _task(shape="judgment", data_class=data_class)
    decision = select_lane(task, classification, providers)
    assert decision.lane == "L1"


def test_full_repo_write_shape_c1() -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [_entry()]
    task = _task(shape="full-repo-write", data_class="public")
    decision = select_lane(task, classification, providers)
    assert decision.lane == "C1"


def test_needs_write_on_private_c1() -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [_entry()]
    task = _task(shape="bounded-diagnosis", data_class="private", needs_write=True)
    decision = select_lane(task, classification, providers)
    assert decision.lane == "C1"


def test_over_budget_bounded_task_falls_back_to_c1() -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [_entry(provider="groq", model_id="openai/gpt-oss-120b", tpm=8000)]
    task = _task(
        shape="small-diff-review",
        data_class="public",
        role="small-diff-review",
        estimated_tokens=50_000,
    )
    decision = select_lane(task, classification, providers)
    assert decision.lane == "C1"
    assert decision.eligible == ()


# ------------------------------------------------------------------ status


@pytest.mark.parametrize("status", ["not-admitted", "quarantined", "excluded"])
def test_non_admitted_status_never_in_decision(status: str) -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [_entry(status=status)]
    task = _task(data_class="public", role="diagnosis")
    decision = select_lane(task, classification, providers)
    assert decision.eligible == ()


def test_decision_never_contains_non_admitted_provider_mixed() -> None:
    classification = _classification("example-org/private-repo", remote=True)
    providers = [
        _entry(provider="admitted-one", status="admitted"),
        _entry(provider="quarantined-one", status="quarantined"),
        _entry(provider="excluded-one", status="excluded"),
        _entry(provider="not-admitted-one", status="not-admitted"),
    ]
    task = _task(data_class="public", role="diagnosis")
    decision = select_lane(task, classification, providers)
    providers_in_decision = {p for p, _m in decision.eligible}
    assert providers_in_decision == {"admitted-one"}
