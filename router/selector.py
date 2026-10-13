"""Deterministic lane selector.

Binding rules from docs/POLICY.md:

- Lanes and task-shape -> lane mapping (POLICY.md:6-17): `D0` no model;
  `L1` sealed local agent; `R1` free remote; `C1` paid/human. Deterministic
  -> D0; repo code needing judgment (any privacy class) -> L1; bounded
  public/synthetic diagnosis or small-diff review -> R1; private,
  full-repo-write, or over-budget -> C1.
- Data classification matrix (POLICY.md:30-43): data classes
  `public|synthetic|private`; provider classes `no-training|training|unknown`
  where `unknown` fails closed as `training`. public+no-training and
  synthetic+no-training are routable; public+training is never routable;
  synthetic+training is routable only for benchmark/read-only roles.
  private is never routable to R1 under any provider class.
- Groq's binding limit is TPM (8,000 on the free gpt-oss models), so a
  provider whose `tpm` cap is below a task's `estimated_tokens` is excluded
  (POLICY.md:26-28).
- `classification.json` is default-deny (POLICY.md, router/classification.json):
  an unlisted repo, or one listed without an explicit `remote: true`, is not
  routable to R1.
- Only `status == "admitted"` registry entries may ever appear in a Decision
  (not-admitted/quarantined/excluded are all excluded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Shape = Literal[
    "deterministic",
    "judgment",
    "bounded-diagnosis",
    "small-diff-review",
    "full-repo-write",
]
DataClass = Literal["public", "synthetic", "private"]
ProviderClass = Literal["no-training", "training", "unknown"]
Lane = Literal["D0", "L1", "R1", "C1"]

# Roles a training/unknown provider may serve for synthetic data
# (POLICY.md:38: "routable, benchmark/read-only roles only"), taken from the
# only training-class registry entry that carries them (router/providers.json,
# google/gemini-free-tier: ["synthetic-benchmark", "public-repo-read"]).
BENCHMARK_READONLY_ROLES = frozenset({"synthetic-benchmark", "public-repo-read"})


@dataclass(frozen=True)
class Task:
    shape: Shape
    data_class: DataClass
    repo: str
    role: str
    estimated_tokens: int
    needs_write: bool = False


@dataclass(frozen=True)
class Decision:
    lane: Lane
    reason: str
    eligible: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def is_routable_to_r1(data_class: DataClass, provider_class: ProviderClass) -> bool:
    """Data-class x provider-class routability, ignoring role.

    private: never (POLICY.md:39). public: only no-training providers
    (POLICY.md:35-37). synthetic: no-training always routable; training and
    unknown are routable only for benchmark/read-only roles, checked
    separately by `provider_eligible` (POLICY.md:38, 32-33 fail-closed).
    """
    if data_class == "private":
        return False
    if provider_class == "no-training":
        return True
    # training or unknown (unknown fails closed as training, POLICY.md:33)
    return data_class == "synthetic"


def repo_permits_remote(classification: dict[str, Any], repo: str) -> bool:
    """classification.json is default-deny: only an explicit allow permits.

    An unlisted repo, or one listed without `"remote": true`, is not
    routable to R1.
    """
    entry = classification.get("repos", {}).get(repo)
    if not isinstance(entry, dict):
        return False
    return entry.get("remote") is True


def provider_eligible(entry: dict[str, Any], task: Task) -> bool:
    """Whether one registry entry may serve one task on R1."""
    if entry.get("status") != "admitted":
        return False
    if task.role not in entry.get("allowed_roles", []):
        return False
    provider_class: ProviderClass = entry["data_policy_class"]
    if not is_routable_to_r1(task.data_class, provider_class):
        return False
    if (
        task.data_class == "synthetic"
        and provider_class in ("training", "unknown")
        and task.role not in BENCHMARK_READONLY_ROLES
    ):
        return False
    tpm = entry.get("tpm")
    if tpm is not None and tpm < task.estimated_tokens:
        return False
    return True


def select_lane(
    task: Task,
    classification: dict[str, Any],
    providers: list[dict[str, Any]],
) -> Decision:
    """Select a lane for `task`. Pure function; no I/O."""
    if task.shape == "deterministic":
        return Decision("D0", "deterministic-shape")
    if task.shape == "judgment":
        return Decision("L1", "judgment-shape-sealed-local")
    if task.shape == "full-repo-write":
        return Decision("C1", "full-repo-write-shape")

    # Remaining shapes: bounded-diagnosis, small-diff-review.
    if task.needs_write and task.data_class == "private":
        return Decision("C1", "needs-write-on-private")
    if task.data_class == "private":
        return Decision("C1", "private-never-routable-to-r1")
    if not repo_permits_remote(classification, task.repo):
        return Decision("C1", "repo-not-allowlisted-for-remote")

    eligible = tuple(
        (entry["provider"], entry["model_id"])
        for entry in providers
        if provider_eligible(entry, task)
    )
    if eligible:
        return Decision("R1", "bounded-task-eligible-provider-found", eligible)
    return Decision("C1", "no-eligible-r1-provider-fallback-to-c1")
