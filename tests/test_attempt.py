from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from router import attempt as attempt_mod


def test_attempt_open_creates_worktree(git_repo: Path, tmp_path: Path) -> None:
    a = attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path)
    try:
        assert a.worktree.is_dir()
        assert (a.worktree / "committed.txt").read_text() == "hello world\n"
    finally:
        a.discard()


def test_discard_removes_worktree_and_registration(
    git_repo: Path, tmp_path: Path
) -> None:
    a = attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path)
    wt = a.worktree
    a.discard()
    assert not wt.exists()
    listing = subprocess.run(
        ["git", "-C", str(git_repo), "worktree", "list"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert str(wt) not in listing


def test_context_manager_discards_by_default(git_repo: Path, tmp_path: Path) -> None:
    with attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path) as a:
        wt = a.worktree
        assert wt.is_dir()
    assert not wt.exists()


def test_context_manager_keeps_when_told(git_repo: Path, tmp_path: Path) -> None:
    with attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path) as a:
        wt = a.worktree
        a.keep()
    assert wt.is_dir()
    subprocess.run(
        ["git", "-C", str(git_repo), "worktree", "remove", "--force", str(wt)],
        check=True,
        capture_output=True,
    )


def test_rejected_attempt_leaves_no_stray_worktree_or_untracked_noise(
    git_repo: Path, tmp_path: Path
) -> None:
    baseline = subprocess.run(
        ["git", "-C", str(git_repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    with attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path) as a:
        (a.worktree / "scratch.txt").write_text("noise\n")
        a.files_changed = True
        a.tools_executed = True
        # not kept -> discarded on exit

    after = subprocess.run(
        ["git", "-C", str(git_repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert after == baseline

    listing = subprocess.run(
        ["git", "-C", str(git_repo), "worktree", "list"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert len([line for line in listing.splitlines() if line.strip()]) == 1


def test_assert_tree_identical_passes_when_unmodified(
    git_repo: Path, tmp_path: Path
) -> None:
    a = attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path)
    try:
        attempt_mod.assert_tree_identical("HEAD", a.worktree)
    finally:
        a.discard()


def test_assert_tree_identical_raises_on_modified_file(
    git_repo: Path, tmp_path: Path
) -> None:
    a = attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path)
    try:
        (a.worktree / "committed.txt").write_text("changed\n")
        with pytest.raises(attempt_mod.TreeChanged):
            attempt_mod.assert_tree_identical("HEAD", a.worktree)
    finally:
        a.discard()


def test_assert_tree_identical_raises_on_stray_untracked_file(
    git_repo: Path, tmp_path: Path
) -> None:
    a = attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path)
    try:
        (a.worktree / "stray.txt").write_text("noise\n")
        with pytest.raises(attempt_mod.TreeChanged):
            attempt_mod.assert_tree_identical("HEAD", a.worktree)
    finally:
        a.discard()


def test_switch_model_refuses_after_tools_executed(
    git_repo: Path, tmp_path: Path
) -> None:
    a = attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path)
    try:
        a.tools_executed = True
        with pytest.raises(attempt_mod.AttemptInProgress):
            attempt_mod.switch_model(a, "other-model")
    finally:
        a.discard()


def test_switch_model_refuses_after_files_changed(
    git_repo: Path, tmp_path: Path
) -> None:
    a = attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path)
    try:
        a.files_changed = True
        with pytest.raises(attempt_mod.AttemptInProgress):
            attempt_mod.switch_model(a, "other-model")
    finally:
        a.discard()


def test_switch_model_allowed_before_execution(git_repo: Path, tmp_path: Path) -> None:
    a = attempt_mod.Attempt.open(git_repo, "HEAD", base_dir=tmp_path)
    try:
        attempt_mod.switch_model(a, "model-b")
        assert a.model == "model-b"
    finally:
        a.discard()


def test_sanitized_env_forbid_keys_raises_on_key_name() -> None:
    env = {"GROQ_API_KEY": "x", "PATH": "/usr/bin"}
    with pytest.raises(attempt_mod.KeyPresent):
        attempt_mod.sanitized_env(env, forbid_keys=True)


def test_sanitized_env_forbid_keys_raises_on_key_shaped_value() -> None:
    key = "sk-ant-" + "c" * 20
    env = {"SOME_OTHER_VAR": key}
    with pytest.raises(attempt_mod.KeyPresent):
        attempt_mod.sanitized_env(env, forbid_keys=True)


def test_sanitized_env_without_forbid_strips_and_passes_through() -> None:
    env = {"GROQ_API_KEY": "x", "PATH": "/usr/bin", "HOME": "/home/x"}
    out = attempt_mod.sanitized_env(env, forbid_keys=False)
    assert "GROQ_API_KEY" not in out
    assert out["PATH"] == "/usr/bin"
    assert out["HOME"] == "/home/x"
