"""Shared fixtures: a real temp git repo with a tracked file, an untracked
``.env`` carrying a fake key (built at runtime so no key-shaped literal sits
in this source file), and a variant with a fake key shape committed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", cwd=repo)
    # The operator's global core.hooksPath points at host pre-commit hooks
    # (detect-secrets among them), which would block this fixture's deliberate
    # fake-secret commit. Hooks are the host's concern, not the fixture's.
    hooks = tmp_path / "nohooks"
    hooks.mkdir()
    _git("config", "core.hooksPath", str(hooks), cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "committed.txt").write_text("hello world\n")
    _git("add", "committed.txt", cwd=repo)
    _git("commit", "-m", "init", cwd=repo)

    groq_key = "gsk_" + "a" * 20
    (repo / ".env").write_text(f"GROQ_API_KEY={groq_key}\n")
    return repo


@pytest.fixture
def git_repo_with_committed_secret(git_repo: Path) -> Path:
    google_key = "AIzaSy" + "b" * 33
    (git_repo / "leaked.txt").write_text(f"key={google_key}\n")
    _git("add", "leaked.txt", cwd=git_repo)
    _git("commit", "-m", "oops", cwd=git_repo)
    return git_repo
