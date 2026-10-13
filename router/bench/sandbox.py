"""Disposable-container exec for exec-scorers (mutation kill, red-test
repair) — the only sanctioned way to run model-generated code.

Every run is a fresh ``docker run`` with, non-negotiably: an image pinned
by tag AND digest (a bare tag is a moving target), ``--network none``
(model output must never reach a socket), a read-only rootfs with a tmpfs
workdir, cpu/memory/pids bounds, a hard wall-clock limit enforced INSIDE
the container by ``timeout`` (so docker's own start-up never counts
against the snippet), and no host environment passthrough (docker passes
no env by default; nothing here adds ``-e``/``--env-file``). Inputs are
mounted read-only at /src; the snippet runs from the tmpfs /work.

When docker is unusable this module raises SandboxUnavailableError —
callers must refuse to score, never silently pass (a scorer that cannot
execute is not a gate). The same goes for a docker that stalls past the
host-side backstop: the run says nothing about the snippet.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Digest of docker.io/library/python:3.12-slim as pulled on this host
# (verified 2026-09-01 via `docker inspect --format '{{.RepoDigests}}'`).
# Bumping the image = pull the new tag, update this constant, and re-run
# the sandbox suite green (the no-network proof included) in one change.
SANDBOX_IMAGE = (
    "python:3.12-slim"
    "@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab"
)
SANDBOX_CPUS = "1"
SANDBOX_MEMORY = "256m"
SANDBOX_PIDS_LIMIT = "64"
SANDBOX_TMPFS = "/work:rw,size=64m,mode=1777"
DEFAULT_TIMEOUT_S = 30
# A snippet that ignores TERM at the limit gets KILL this much later.
SANDBOX_KILL_AFTER_S = 5
# Room for docker's own start-up and teardown on a loaded host (start-up
# alone has run past 30 s). Only the host-side backstop uses it.
SANDBOX_STARTUP_ALLOWANCE_S = 120
_ENTRY = "main.py"
# GNU `timeout` exits 124 when TERM at the limit stopped the command.
_LIMIT_HIT_RC = 124
# KILL ends the snippet with 137: it ignored TERM past the grace, or it hit
# the memory bound. The two cannot be told apart from the exit code.
_KILLED_RC = 137


class SandboxUnavailableError(RuntimeError):
    """Docker is missing, its daemon is unreachable, or it stalled past the
    host-side backstop. Callers must refuse to score — treating this as a
    pass would be a fake gate."""


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def killed(self) -> bool:
        """KILL ended the snippet: it ignored TERM past the limit's grace,
        or it hit the memory bound. Either way the sandbox stopped it; it
        did not fail on its own."""
        return self.exit_code == _KILLED_RC


def _require_docker() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise SandboxUnavailableError(
            "docker binary not found — exec-scoring must refuse, not pass"
        )
    try:
        proc = subprocess.run([docker, "info"], capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SandboxUnavailableError(f"docker daemon unreachable: {e}") from e
    if proc.returncode != 0:
        raise SandboxUnavailableError(
            "docker daemon unreachable: " + proc.stderr.decode(errors="replace").strip()
        )
    return docker


def _remove_container(docker: str, name: str) -> str:
    """Force-remove a stalled run's container, which also clears one that
    was created but never started (``--rm`` leaves that behind). Docker is
    already stalling here, so this may fail too: report it, never raise."""
    try:
        proc = subprocess.run(
            [docker, "rm", "-f", name], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"could not remove {name}: {e}"
    if proc.returncode != 0:
        return f"could not remove {name}: {proc.stderr.strip()}"
    return f"removed {name}"


def run_snippet(
    code: str,
    *,
    files: Mapping[str, str] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> SandboxResult:
    """Run a python snippet in a disposable locked-down container.

    ``files`` are extra sources placed read-only next to the snippet in
    /src (relative names only). Returns exit code and captured output.
    ``timeout_s`` is counted inside the container, from the snippet's own
    start: TERM at the limit ends it with 124 and ``timed_out`` is True. A
    snippet that ignores TERM gets KILL ``SANDBOX_KILL_AFTER_S`` later and
    ends with 137, read as ``killed`` (not ``timed_out``), the same as one
    that hits the memory bound. A snippet that itself exits 124 or 137
    reads the same way, which only ever counts against it. Raises
    SandboxUnavailableError when docker cannot run anything, or stalls
    past the host-side backstop.
    """
    if timeout_s < 1:
        # `timeout 0` means no limit at all.
        raise ValueError(f"timeout_s must be at least 1, got {timeout_s}")
    docker = _require_docker()
    name = f"bench-sandbox-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="bench-sandbox-") as tmp:
        src = Path(tmp)
        (src / _ENTRY).write_text(code)
        for rel, content in (files or {}).items():
            dest = (src / rel).resolve()
            if not dest.is_relative_to(src.resolve()):
                raise ValueError(f"file escapes sandbox source dir: {rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        cmd = [
            docker,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            SANDBOX_TMPFS,
            "--workdir",
            "/work",
            "--cpus",
            SANDBOX_CPUS,
            "--memory",
            SANDBOX_MEMORY,
            "--pids-limit",
            SANDBOX_PIDS_LIMIT,
            "-v",
            f"{src}:/src:ro",
            SANDBOX_IMAGE,
            "timeout",
            "-k",
            str(SANDBOX_KILL_AFTER_S),
            str(timeout_s),
            "python",
            f"/src/{_ENTRY}",
        ]
        backstop_s = timeout_s + SANDBOX_KILL_AFTER_S + SANDBOX_STARTUP_ALLOWANCE_S
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=backstop_s
            )
        except subprocess.TimeoutExpired as e:
            # The in-container limit ends every snippet long before this,
            # so docker itself stalled (start-up or teardown).
            cleanup = _remove_container(docker, name)
            raise SandboxUnavailableError(
                f"docker stalled: no result within {backstop_s}s for a "
                f"{timeout_s}s snippet limit; {cleanup} — refusing to score"
            ) from e
    return SandboxResult(
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        timed_out=proc.returncode == _LIMIT_HIT_RC,
    )
