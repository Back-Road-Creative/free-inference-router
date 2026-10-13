"""Tests for router.killswitch — unreadable-is-engaged semantics.

Mirrors bin/autoland.py:334-345 (AUTOLAND_KILL_FILE handling): existence OR
any OSError on stat/open counts as engaged; only a clean "does not exist"
means disengaged. See docs/POLICY.md:54-58.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from router.killswitch import KillSwitchEngaged, engaged, require_disengaged


def test_absent_path_is_not_engaged(tmp_path: Path) -> None:
    path = tmp_path / "disabled"
    assert path.exists() is False
    assert engaged(path) is False


def test_present_path_is_engaged(tmp_path: Path) -> None:
    path = tmp_path / "disabled"
    path.write_text("stop")
    assert engaged(path) is True


def test_directory_at_path_is_engaged(tmp_path: Path) -> None:
    # Path.exists() on a directory returns True regardless of type; a switch
    # implemented as a directory is still "present" and must still engage.
    path = tmp_path / "disabled"
    path.mkdir()
    assert engaged(path) is True


def test_unreadable_file_is_engaged_via_permission_error(
    tmp_path: Path,
) -> None:
    if os.geteuid() == 0:
        pytest.skip("running as root: chmod 000 does not deny root reads")
    path = tmp_path / "disabled"
    path.write_text("stop")
    os.chmod(path, 0o000)
    try:
        # A permission-denied stat on the file itself raises OSError out of
        # Path.exists() (pathlib only swallows ENOENT/ENOTDIR/ELOOP/EBADF),
        # which must be treated as engaged, never as "absent, proceed".
        assert engaged(path) is True
    finally:
        os.chmod(path, 0o644)


def test_stat_oserror_injected_is_engaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "disabled"

    def _raise_stat(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("injected")

    monkeypatch.setattr(Path, "stat", _raise_stat)
    assert engaged(path) is True


def test_require_disengaged_raises_when_engaged(tmp_path: Path) -> None:
    path = tmp_path / "disabled"
    path.write_text("stop")
    with pytest.raises(KillSwitchEngaged):
        require_disengaged(path)


def test_require_disengaged_ok_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "disabled"
    require_disengaged(path)  # must not raise
