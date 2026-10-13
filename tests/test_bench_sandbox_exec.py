"""Contract tests for router.bench.sandbox — the disposable-container exec.

The docker-backed tests skip when docker is unusable. A SKIPPED
no-network proof is NOT a pass: it must execute (green, not skipped)
on the machine where exec-scorers will run before any model output is
ever executed there.

The command-level tests at the bottom run everywhere: they fake
``subprocess.run`` and pin the ``docker`` commands ``run_snippet`` issues.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from router.bench.sandbox import (
    SANDBOX_CPUS,
    SANDBOX_IMAGE,
    SANDBOX_KILL_AFTER_S,
    SANDBOX_MEMORY,
    SANDBOX_PIDS_LIMIT,
    SANDBOX_STARTUP_ALLOWANCE_S,
    SANDBOX_TMPFS,
    SandboxResult,
    SandboxUnavailableError,
    run_snippet,
)


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_usable(),
    reason=(
        "docker unavailable — sandbox proofs (incl. NO-NETWORK) did NOT "
        "execute; they must run green where exec-scorers will run"
    ),
)


@requires_docker
def test_trivial_snippet_captures_output() -> None:
    res = run_snippet("import sys; print('ok-42'); print('err-7', file=sys.stderr)")
    assert res.exit_code == 0
    assert not res.timed_out
    assert "ok-42" in res.stdout
    assert "err-7" in res.stderr


@requires_docker
def test_timeout_kills_sleeping_container() -> None:
    start = time.monotonic()
    res = run_snippet("import time; time.sleep(120)", timeout_s=8)
    elapsed = time.monotonic() - start
    assert res.timed_out
    assert not res.killed
    assert res.exit_code != 0
    assert 8 <= elapsed < 60


@requires_docker
def test_snippet_ignoring_sigterm_is_killed_after_grace() -> None:
    """The in-container limit sends TERM, then KILL after the grace. A
    snippet that ignores TERM still ends long before the host backstop
    (which would refuse to score the run as a docker stall)."""
    snippet = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(120)\n"
    )
    start = time.monotonic()
    res = run_snippet(snippet, timeout_s=3)
    elapsed = time.monotonic() - start
    assert res.exit_code == 137  # 128 + KILL
    assert res.killed
    assert not res.timed_out
    # KILL waited out the grace after the limit, and still beat the backstop.
    assert 3 + SANDBOX_KILL_AFTER_S <= elapsed < 60


@requires_docker
def test_no_network_proof_connects_all_fail() -> None:
    """A sandboxed connect to a public IP and to the docker host gateway
    must fail. This is the safety property exec-scorers rely on."""
    snippet = (
        "import socket\n"
        "for host in ('1.1.1.1', '172.17.0.1'):\n"
        "    try:\n"
        "        socket.create_connection((host, 80), timeout=3)\n"
        "        print('REACHED', host)\n"
        "        raise SystemExit(1)\n"
        "    except OSError:\n"
        "        print('BLOCKED', host)\n"
    )
    res = run_snippet(snippet, timeout_s=30)
    assert res.exit_code == 0, res.stdout + res.stderr
    assert "REACHED" not in res.stdout
    assert res.stdout.count("BLOCKED") == 2


@requires_docker
def test_no_env_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_CANARY_SECRET", "leak-me")
    res = run_snippet(
        "import os; print(sorted(k for k in os.environ if 'CANARY' in k))"
    )
    assert res.exit_code == 0
    assert "leak-me" not in res.stdout
    assert "SANDBOX_CANARY_SECRET" not in res.stdout


def test_docker_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(SandboxUnavailableError, match="docker"):
        run_snippet("print('never runs')")


# --- docker-free: the commands run_snippet issues ---
# The proofs above need a docker daemon, so a runner without one (the CI
# canary) skips them. These pin the same bounds at the command level with a
# fake `subprocess.run`, so every runner checks them. They prove what is
# requested, not that the kernel enforces it: the docker proofs above stay
# the authority for that.

_FAKE_DOCKER = "/fake/bin/docker"


def _opts(cmd: list[str]) -> list[str]:
    """The docker options: everything before the image (the rest runs inside)."""
    return cmd[: cmd.index(SANDBOX_IMAGE)]


def _flag(cmd: list[str], name: str) -> str:
    opts = _opts(cmd)
    return opts[opts.index(name) + 1]


class _FakeDocker:
    """Stands in for `subprocess.run`; records each docker command."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.mounted: dict[str, str] = {}
        self.info_result: BaseException | subprocess.CompletedProcess[bytes] = (
            subprocess.CompletedProcess([_FAKE_DOCKER, "info"], 0, b"", b"")
        )
        self.run_raises: BaseException | None = None
        self.run_returncode = 3
        self.rm_result: BaseException | subprocess.CompletedProcess[str] = (
            subprocess.CompletedProcess([_FAKE_DOCKER, "rm"], 0, "", "")
        )

    def __call__(
        self, cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[Any]:
        self.calls.append((list(cmd), kwargs))
        if cmd[1] == "info":
            if isinstance(self.info_result, BaseException):
                raise self.info_result
            return self.info_result
        if cmd[1] == "run":
            # The source dir is deleted when run_snippet returns: read it now.
            src = Path(_flag(cmd, "-v").removesuffix(":/src:ro"))
            self.mounted = {
                p.relative_to(src).as_posix(): p.read_text()
                for p in src.rglob("*")
                if p.is_file()
            }
            if self.run_raises is not None:
                raise self.run_raises
            return subprocess.CompletedProcess(
                cmd, self.run_returncode, "out-text", "err-text"
            )
        if cmd[1] == "rm":
            if isinstance(self.rm_result, BaseException):
                raise self.rm_result
            return self.rm_result
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def commands(self, verb: str) -> list[list[str]]:
        return [cmd for cmd, _ in self.calls if cmd[1] == verb]

    def kwargs(self, verb: str) -> list[dict[str, Any]]:
        return [kw for cmd, kw in self.calls if cmd[1] == verb]


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> _FakeDocker:
    fake = _FakeDocker()
    monkeypatch.setattr("shutil.which", lambda cmd: _FAKE_DOCKER)
    monkeypatch.setattr("subprocess.run", fake)
    return fake


def test_command_carries_every_sandbox_bound(fake_docker: _FakeDocker) -> None:
    run_snippet("print('hi')", timeout_s=7)

    assert fake_docker.commands("info") == [[_FAKE_DOCKER, "info"]]
    [cmd] = fake_docker.commands("run")
    opts = _opts(cmd)
    assert opts[:2] == [_FAKE_DOCKER, "run"]
    assert "--rm" in opts
    assert _flag(cmd, "--network") == "none"
    assert "--read-only" in opts
    assert _flag(cmd, "--tmpfs") == SANDBOX_TMPFS
    assert _flag(cmd, "--workdir") == "/work"
    assert _flag(cmd, "--cpus") == SANDBOX_CPUS
    assert _flag(cmd, "--memory") == SANDBOX_MEMORY
    assert _flag(cmd, "--pids-limit") == SANDBOX_PIDS_LIMIT
    assert _flag(cmd, "-v").endswith(":/src:ro")
    # The 7 s limit is enforced INSIDE the container, so docker start-up
    # never counts against it: TERM at 7 s, KILL after the grace.
    assert cmd[len(opts) :] == [
        SANDBOX_IMAGE,
        "timeout",
        "-k",
        str(SANDBOX_KILL_AFTER_S),
        "7",
        "python",
        "/src/main.py",
    ]
    assert "@sha256:" in SANDBOX_IMAGE
    # The host-side timeout is only a backstop for a stalled docker.
    assert fake_docker.kwargs("run")[0]["timeout"] == (
        7 + SANDBOX_KILL_AFTER_S + SANDBOX_STARTUP_ALLOWANCE_S
    )


def test_backstop_leaves_room_for_a_slow_docker_start_up() -> None:
    # Docker start-up alone took 30 s on a loaded CI runner (2026-09-29) and
    # 17 s locally. A smaller allowance would refuse healthy runs as stalls.
    assert SANDBOX_STARTUP_ALLOWANCE_S >= 60


def test_command_adds_no_env_mount_network_or_privilege(
    fake_docker: _FakeDocker,
) -> None:
    run_snippet("print('hi')")

    [cmd] = fake_docker.commands("run")
    names = [arg.split("=", 1)[0] for arg in _opts(cmd)]
    for banned in (
        "-e",
        "--env",
        "--env-file",
        "--mount",
        "--volume",
        "--volumes-from",
        "--privileged",
        "--cap-add",
        "--device",
    ):
        assert banned not in names
    assert names.count("-v") == 1
    assert [n for n in names if n in ("--network", "--net")] == ["--network"]


def test_snippet_and_files_are_what_gets_mounted(fake_docker: _FakeDocker) -> None:
    res = run_snippet("print('hi')\n", files={"pkg/helper.py": "X = 1\n"})

    assert fake_docker.mounted == {
        "main.py": "print('hi')\n",
        "pkg/helper.py": "X = 1\n",
    }
    assert res == SandboxResult(exit_code=3, stdout="out-text", stderr="err-text")


@pytest.mark.parametrize(
    "rel", ["../outside.py", "pkg/../../outside.py", "/tmp/outside.py"]
)
def test_file_escaping_the_source_dir_is_refused(
    fake_docker: _FakeDocker, rel: str
) -> None:
    with pytest.raises(ValueError, match="escapes sandbox source dir"):
        run_snippet("print('hi')", files={rel: "X = 1\n"})
    assert fake_docker.commands("run") == []


@pytest.mark.parametrize(
    ("returncode", "timed_out", "killed"),
    [(124, True, False), (137, False, True), (1, False, False), (0, False, False)],
    ids=["limit-hit", "killed", "failed", "passed"],
)
def test_only_the_in_container_limit_reads_as_timed_out(
    fake_docker: _FakeDocker, returncode: int, timed_out: bool, killed: bool
) -> None:
    """`timeout` exits 124 when the limit stops the snippet. 137 is KILL:
    TERM ignored past the grace, or the memory bound. The two cannot be
    told apart, so it is `killed`, not `timed_out`."""
    fake_docker.run_returncode = returncode

    res = run_snippet("import time; time.sleep(120)", timeout_s=5)

    assert res == SandboxResult(
        exit_code=returncode, stdout="out-text", stderr="err-text", timed_out=timed_out
    )
    assert res.killed is killed
    assert fake_docker.commands("rm") == []


def test_stalled_docker_is_removed_and_refused_not_scored(
    fake_docker: _FakeDocker,
) -> None:
    """The backstop firing means docker itself stalled (the in-container
    limit ends every snippet first), so the run says nothing about the
    snippet: remove the named container and refuse, never report a timeout.
    `rm -f` also clears a container that was created but never started,
    which `--rm` would leave behind."""
    fake_docker.run_raises = subprocess.TimeoutExpired(
        ["docker"], 5 + SANDBOX_KILL_AFTER_S + SANDBOX_STARTUP_ALLOWANCE_S
    )

    with pytest.raises(SandboxUnavailableError, match="docker stalled") as exc:
        run_snippet("print('hi')", timeout_s=5)

    [run_cmd] = fake_docker.commands("run")
    name = _flag(run_cmd, "--name")
    assert name.startswith("bench-sandbox-")
    assert fake_docker.commands("rm") == [[_FAKE_DOCKER, "rm", "-f", name]]
    assert f"removed {name}" in str(exc.value)


@pytest.mark.parametrize(
    ("rm_result", "reason"),
    [
        (OSError("no socket"), "no socket"),
        (subprocess.TimeoutExpired(["docker", "rm"], 30), "timed out"),
        (
            subprocess.CompletedProcess(
                ["docker", "rm"], 1, "", "Error response from daemon: busy\n"
            ),
            "Error response from daemon: busy$",
        ),
    ],
    ids=["oserror", "rm-stalls-too", "rm-refused"],
)
def test_stall_is_refused_even_when_removal_fails(
    fake_docker: _FakeDocker, rm_result: BaseException | Any, reason: str
) -> None:
    """A stalled docker is the premise here, so the clean-up may fail too.
    The caller still gets the refusal, saying the container may be left."""
    fake_docker.run_raises = subprocess.TimeoutExpired(["docker"], 130)
    fake_docker.rm_result = rm_result

    with pytest.raises(SandboxUnavailableError, match="docker stalled") as exc:
        run_snippet("print('hi')", timeout_s=5)

    [run_cmd] = fake_docker.commands("run")
    name = _flag(run_cmd, "--name")
    message = str(exc.value)
    assert f"could not remove {name}" in message
    assert re.search(reason, message.removesuffix(" — refusing to score"))


@pytest.mark.parametrize("timeout_s", [0, -1])
def test_timeout_below_one_second_is_refused(
    fake_docker: _FakeDocker, timeout_s: int
) -> None:
    # `timeout 0` means NO limit, so a zero would silently unbound the run.
    with pytest.raises(ValueError, match="timeout_s"):
        run_snippet("print('hi')", timeout_s=timeout_s)
    assert fake_docker.calls == []


@pytest.mark.parametrize(
    ("info_result", "message"),
    [
        (OSError("no socket"), "docker daemon unreachable: no socket"),
        (
            subprocess.TimeoutExpired(["docker", "info"], 15),
            "docker daemon unreachable",
        ),
        (
            subprocess.CompletedProcess(
                ["docker", "info"], 1, b"", b"Cannot connect to the Docker daemon\n"
            ),
            "docker daemon unreachable: Cannot connect to the Docker daemon$",
        ),
    ],
    ids=["oserror", "timeout", "nonzero"],
)
def test_unreachable_daemon_refuses_before_running(
    fake_docker: _FakeDocker,
    info_result: BaseException | subprocess.CompletedProcess[bytes],
    message: str,
) -> None:
    fake_docker.info_result = info_result

    with pytest.raises(SandboxUnavailableError, match=message):
        run_snippet("print('never runs')")
    assert fake_docker.commands("run") == []
