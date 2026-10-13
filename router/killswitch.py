"""The free-inference-router kill switch.

Checked before every dispatch (docs/POLICY.md:54-58): `/etc/free-router/disabled`.
An **unreadable** file counts as engaged -- mirrors `bin/autoland.py`'s
`AUTOLAND_KILL_FILE` handling (bin/autoland.py:330-350), which treats
`path.exists()` raising `OSError` (e.g. a permission-denied stat when the
switch lives in a directory this account can no longer traverse) the same
as the switch being present. Only a clean "does not exist" means
disengaged -- never assume a switch is clear because it could not be read.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT = Path("/etc/free-router/disabled")


class KillSwitchEngaged(RuntimeError):
    """Raised by `require_disengaged` when the kill switch is engaged."""


def engaged(path: Path = DEFAULT) -> bool:
    """Return True if the kill switch is engaged.

    Engaged means: the path exists, OR checking it raised any OSError
    (permission denied, unreadable parent directory, etc). Only a clean
    "the path does not exist" result means disengaged.
    """
    try:
        return path.exists()
    except OSError:
        return True


def require_disengaged(path: Path = DEFAULT) -> None:
    """Raise KillSwitchEngaged if the kill switch is engaged."""
    if engaged(path):
        raise KillSwitchEngaged(f"kill switch engaged: {path}")
