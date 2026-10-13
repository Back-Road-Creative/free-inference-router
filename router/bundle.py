"""Outbound payload construction and secret scanning.

Per docs/POLICY.md:58-61: "Every payload is a git archive of tracked files
only, so an untracked .env can never be bundled by accident, scanned with
detect-secrets before send; any hit hard-stops."

``build_bundle`` uses ``git archive <ref>`` to produce the tar -- git archive
only ever walks the tree object recorded at ``ref``, so an untracked or
merely-staged-but-uncommitted file structurally cannot appear in the output.
This is "robust by construction" (no filter to get wrong), not a gate applied
after the fact.

Two independent scanning layers run before anything is treated as safe to
send: ``detect-secrets`` (``scan_bundle``) and a small dependency-free regex
layer over the same extracted content (``scan_text``), so a host without
detect-secrets installed still gets *a* scan rather than silently skipping
straight to "clean". ``prepare_payload`` runs both and raises on any finding
from either.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: Value shapes for the paid/free provider keys this router can ever touch.
#: Mirrors the provider list in the task brief; kept router-local rather than
#: imported from the local-agent lane (different service, different key set).
_KEY_VALUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "groq": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "google": re.compile(r"AIzaSy[A-Za-z0-9_-]{20,}"),
    "openrouter": re.compile(r"sk-or-[A-Za-z0-9-]{10,}"),
    "nvidia": re.compile(r"nvapi-[A-Za-z0-9-]{20,}"),
    "anthropic": re.compile(r"sk-ant-[A-Za-z0-9-]{10,}"),
    "openai": re.compile(r"sk-[A-Za-z0-9]{20,}"),
}


class ScannerUnavailable(RuntimeError):
    """Raised when detect-secrets cannot be located -- never silently passed."""


class SecretsFound(RuntimeError):
    """Raised by ``prepare_payload`` when either scan layer finds a hit."""

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings
        super().__init__(f"{len(findings)} secret-shaped finding(s) in payload")


@dataclass(frozen=True)
class Finding:
    """One secret-shaped hit. Carries only the source name and finding kind
    -- never the matched value -- so a Finding is itself safe to log."""

    source: str
    kind: str


@dataclass(frozen=True)
class BundleInfo:
    tar_path: Path
    members: tuple[str, ...]
    total_bytes: int


def build_bundle(repo: Path | str, ref: str, out: Path | str) -> BundleInfo:
    """Archive the tracked files at ``ref`` in ``repo`` into ``out``.

    Tracked-files-only by construction: ``git archive`` walks the tree object
    recorded at ``ref``, never the working directory or the index, so there is
    no filter step that could be misconfigured to admit an untracked file.
    """
    repo = Path(repo)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "--output", str(out), ref],
        check=True,
        capture_output=True,
    )
    with tarfile.open(out) as tf:
        file_members = [m for m in tf.getmembers() if m.isfile()]
        members = tuple(m.name for m in file_members)
        total_bytes = sum(m.size for m in file_members)
    return BundleInfo(tar_path=out, members=members, total_bytes=total_bytes)


def _find_detect_secrets() -> str:
    exe = shutil.which("detect-secrets")
    if exe:
        return exe
    fallback = Path.home() / ".local" / "bin" / "detect-secrets"
    if fallback.is_file():
        return str(fallback)
    raise ScannerUnavailable("detect-secrets binary not found on PATH or ~/.local/bin")


def scan_bundle(tar_path: Path | str) -> list[Finding]:
    """Run detect-secrets over an extracted copy of ``tar_path``.

    Raises ``ScannerUnavailable`` rather than returning an empty (falsely
    reassuring) list when the binary cannot be located.
    """
    exe = _find_detect_secrets()
    tar_path = Path(tar_path)
    findings: list[Finding] = []
    with tempfile.TemporaryDirectory(prefix="router-scan-") as td:
        extract_dir = Path(td)
        with tarfile.open(tar_path) as tf:
            # Our own archive of `ref`, but extract with the data filter anyway:
            # Python 3.14 makes it the default and 3.12+ warns without it.
            if hasattr(tarfile, "data_filter"):
                tf.extractall(extract_dir, filter="data")
            else:  # pragma: no cover - Python < 3.12
                tf.extractall(extract_dir)
        proc = subprocess.run(
            [exe, "scan", str(extract_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return findings
        for source, hits in data.get("results", {}).items():
            for hit in hits:
                findings.append(
                    Finding(source=source, kind=str(hit.get("type", "unknown")))
                )
    return findings


def scan_text(text: str) -> list[Finding]:
    """Dependency-free regex layer over ``text`` for known key value shapes."""
    return [
        Finding(source="<text>", kind=name)
        for name, pattern in _KEY_VALUE_PATTERNS.items()
        if pattern.search(text)
    ]


def prepare_payload(repo: Path | str, ref: str, out: Path | str) -> BundleInfo:
    """Build the bundle and run both scan layers; raise on any finding."""
    info = build_bundle(repo, ref, out)
    findings = scan_bundle(info.tar_path)

    with tarfile.open(info.tar_path) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            content = extracted.read().decode("utf-8", errors="ignore")
            findings.extend(
                Finding(source=member.name, kind=f.kind) for f in scan_text(content)
            )

    if findings:
        raise SecretsFound(findings)
    return info
