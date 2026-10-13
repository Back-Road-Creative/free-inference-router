from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from router import bundle


def _detect_secrets_available() -> bool:
    if shutil.which("detect-secrets"):
        return True
    return (Path.home() / ".local" / "bin" / "detect-secrets").is_file()


def test_bundle_contains_committed_not_untracked_env(
    git_repo: Path, tmp_path: Path
) -> None:
    out = tmp_path / "payload.tar"
    info = bundle.build_bundle(git_repo, "HEAD", out)
    assert "committed.txt" in info.members
    assert not any(m.endswith(".env") for m in info.members)


def test_bundle_excludes_uncommitted_working_tree_additions(
    git_repo: Path, tmp_path: Path
) -> None:
    (git_repo / "staged_not_committed.txt").write_text("x\n")
    subprocess.run(
        ["git", "add", "staged_not_committed.txt"],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )
    out = tmp_path / "payload2.tar"
    info = bundle.build_bundle(git_repo, "HEAD", out)
    assert "staged_not_committed.txt" not in info.members
    assert info.total_bytes > 0


def test_scan_text_finds_each_key_shape() -> None:
    samples = [
        "gsk_" + "a" * 20,
        "AIzaSy" + "b" * 33,
        "sk-or-" + "c" * 20,
        "nvapi-" + "d" * 20,
        "sk-ant-" + "e" * 20,
        "sk-" + "f" * 20,
    ]
    for value in samples:
        assert bundle.scan_text(f"key={value}"), f"missed shape in {value!r}"


def test_scan_text_clean_text_finds_nothing() -> None:
    assert bundle.scan_text("hello world, nothing secret here") == []


def test_prepare_payload_raises_secrets_found_on_committed_secret(
    git_repo_with_committed_secret: Path, tmp_path: Path
) -> None:
    if not _detect_secrets_available():
        pytest.skip("detect-secrets not installed on this host")
    out = tmp_path / "payload.tar"
    with pytest.raises(bundle.SecretsFound):
        bundle.prepare_payload(git_repo_with_committed_secret, "HEAD", out)


def test_prepare_payload_clean_repo_returns_bundle_info(
    git_repo: Path, tmp_path: Path
) -> None:
    if not _detect_secrets_available():
        pytest.skip("detect-secrets not installed on this host")
    out = tmp_path / "clean.tar"
    info = bundle.prepare_payload(git_repo, "HEAD", out)
    assert isinstance(info, bundle.BundleInfo)
    assert "committed.txt" in info.members


def test_scan_bundle_raises_scanner_unavailable_when_missing(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, tmp_path: Path
) -> None:
    out = tmp_path / "p.tar"
    info = bundle.build_bundle(git_repo, "HEAD", out)

    monkeypatch.setattr("router.bundle.shutil.which", lambda name: None)
    fake_home = tmp_path / "no-such-home"
    monkeypatch.setattr("router.bundle.Path.home", classmethod(lambda cls: fake_home))

    with pytest.raises(bundle.ScannerUnavailable):
        bundle.scan_bundle(info.tar_path)
