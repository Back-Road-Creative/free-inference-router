"""Attempt semantics: one model per attempt, disposable worktrees.

Per docs/POLICY.md:50-53 and plan Sec.5: one model per attempt; once any tool
has executed or a file has changed, a model switch is a NEW attempt from a
clean disposable worktree, never a resumed session; the first attempt that
passes every deterministic gate is kept, rejected attempts leave the target
tree byte-identical.

``sanitized_env`` replicates the shape of the local-agent lane's
``Policy.sanitized_env`` with a router-specific key list -- not imported
across services, per the plan: this lane must never weaken that boundary by
depending on it.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

#: Keys must never enter the sealed local agent's environment
#: (docs/POLICY.md:66-70). Router-specific list per the task brief.
_FORBIDDEN_KEY_NAMES = frozenset(
    {
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "NVIDIA_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    }
)

#: Value shapes, same set as router.bundle._KEY_VALUE_PATTERNS, kept as one
#: combined pattern here since sanitized_env only needs "does this value
#: look like a key", not which provider it belongs to.
_KEY_SHAPE_PATTERN = re.compile(
    r"(gsk_|AIzaSy|sk-or-|nvapi-|sk-ant-|sk-)[A-Za-z0-9_-]{10,}"
)


class AttemptError(RuntimeError):
    """Base class for attempt-lifecycle violations."""


class TreeChanged(AttemptError):
    """Raised when a worktree that should be untouched is not."""


class AttemptInProgress(AttemptError):
    """Raised when a model switch is requested mid-attempt.

    The caller must open a new ``Attempt`` from a clean disposable worktree
    instead of resuming this one.
    """


class KeyPresent(AttemptError):
    """Raised by ``sanitized_env(forbid_keys=True)`` on any key material."""


@dataclass
class Attempt:
    repo: Path
    ref: str
    worktree: Path
    tools_executed: bool = False
    files_changed: bool = False
    model: str | None = None
    _kept: bool = field(default=False, repr=False)
    _tmp_root: Path | None = field(default=None, repr=False)

    @classmethod
    def open(
        cls, repo: Path | str, ref: str, *, base_dir: Path | str | None = None
    ) -> Attempt:
        repo = Path(repo)
        tmp_root = Path(tempfile.mkdtemp(prefix="router-attempt-", dir=base_dir))
        worktree = tmp_root / "worktree"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), ref],
            check=True,
            capture_output=True,
        )
        return cls(repo=repo, ref=ref, worktree=worktree, _tmp_root=tmp_root)

    def __enter__(self) -> Attempt:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None or not self._kept:
            self.discard()

    def keep(self) -> None:
        """Mark this attempt to survive its context exit (caller's gates passed)."""
        self._kept = True

    def discard(self) -> None:
        """Remove the disposable worktree; safe to call more than once."""
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "worktree",
                "remove",
                "--force",
                str(self.worktree),
            ],
            check=False,
            capture_output=True,
        )
        if self._tmp_root is not None:
            subprocess.run(
                ["rm", "-rf", str(self._tmp_root)], check=False, capture_output=True
            )


def assert_tree_identical(before_ref: str, worktree: Path | str) -> None:
    """Raise ``TreeChanged`` unless ``worktree`` matches ``before_ref`` exactly,
    including having no stray untracked files."""
    worktree = Path(worktree)
    diff = subprocess.run(
        ["git", "-C", str(worktree), "diff", "--quiet", before_ref],
        capture_output=True,
    )
    status = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if diff.returncode != 0 or status.strip():
        raise TreeChanged(
            f"worktree {worktree} diverges from {before_ref}: {status.strip()!r}"
        )


def switch_model(attempt: Attempt, model: str) -> None:
    """Assign ``model`` to ``attempt``. Refuses once the attempt is no longer
    pristine -- the caller must open a new ``Attempt`` instead."""
    if attempt.tools_executed or attempt.files_changed:
        raise AttemptInProgress(
            "attempt has executed tools or changed files; open a new Attempt"
        )
    attempt.model = model


def sanitized_env(
    env: Mapping[str, str], *, forbid_keys: bool = False
) -> dict[str, str]:
    """Reduce ``env`` to values with no paid-provider key material.

    With ``forbid_keys=True`` the mere presence of a forbidden key name, or a
    value shaped like one of the provider key formats, aborts by raising
    ``KeyPresent`` -- a leaked key is a billing/privacy event even if this
    process would not have forwarded it. Without it, key names are stripped
    and everything else passes through unchanged.
    """
    if forbid_keys:
        offenders = sorted(
            {k for k in env if k in _FORBIDDEN_KEY_NAMES}
            | {k for k, v in env.items() if _KEY_SHAPE_PATTERN.search(v)}
        )
        if offenders:
            raise KeyPresent(f"forbidden key material present: {offenders}")
        return dict(env)
    return {k: v for k, v in env.items() if k not in _FORBIDDEN_KEY_NAMES}
