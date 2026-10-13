"""The service tree reads as a standalone project.

It is cut to its own public repository, so nothing in it may point at the
monorepo layout, one machine's paths or any monorepo service directory
name (its own included: the public checkout is not named ``svc-``), operator-local
configs ship as ``*.example.json``, and the licence the package metadata declares
is present.
"""

from __future__ import annotations

import re
from pathlib import Path

SERVICE = Path(__file__).resolve().parent.parent
MACHINE_PATHS = re.compile(r"\b_active/|/home/[a-z]+/")
# A monorepo service directory name says where the tree came from. The tree's
# own is no exception: the public checkout is named after the package, so a
# self-reference reads as another service there.
SERVICE_NAMES = re.compile(r"\bsvc-[a-z][a-z0-9-]*")
# pyproject.toml is edited only by a human here; the public cut rewrites its
# comments with a patch, and the cut's own scan re-checks the result.
CUT_PATCHED = {"pyproject.toml"}
SKIP_DIRS = {
    ".venv",
    ".git",
    ".hypothesis",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
}


def _tracked_text_files() -> list[Path]:
    files = []
    for path in SERVICE.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(SERVICE).parts):
            continue
        if path.name == Path(__file__).name:
            continue
        try:
            path.read_text()
        except UnicodeDecodeError:
            continue
        files.append(path)
    return files


def test_no_monorepo_or_machine_paths() -> None:
    hits = [
        f"{p.relative_to(SERVICE)}:{n}"
        for p in _tracked_text_files()
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if MACHINE_PATHS.search(line)
    ]
    assert hits == []


def test_no_monorepo_service_names() -> None:
    hits = [
        f"{p.relative_to(SERVICE)}:{n}"
        for p in _tracked_text_files()
        if p.name not in CUT_PATCHED
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if SERVICE_NAMES.search(line)
    ]
    assert hits == []


def test_operator_configs_ship_as_examples() -> None:
    compose = SERVICE / "compose"
    assert sorted(p.name for p in compose.glob("*.local.json")) == []
    assert (compose / "pricing.example.json").is_file()
    assert (compose / "model-parameters.example.json").is_file()
    text = (compose / "docker-compose.yml").read_text()
    assert "./pricing.example.json:" in text
    assert "./model-parameters.example.json:" in text


def test_licence_file_matches_package_metadata() -> None:
    assert 'license = { text = "MIT" }' in (SERVICE / "pyproject.toml").read_text()
    assert (SERVICE / "LICENSE").read_text().startswith("MIT License\n")
