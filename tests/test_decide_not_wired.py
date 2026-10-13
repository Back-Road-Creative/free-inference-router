"""router.decide is an offline evaluation harness (JP decision D1,
2026-09-22) and must NEVER be wired into selection, permission, or release
decisions. This asserts the one-way import boundary: router.selector (and,
transitively, anything selection depends on) never imports router.decide."""

from __future__ import annotations

import ast
from pathlib import Path


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_selector_does_not_import_decide() -> None:
    selector_path = Path(__file__).resolve().parents[1] / "router" / "selector.py"
    imported = _imported_module_names(selector_path)
    assert not any("decide" in name for name in imported)


def test_no_module_in_router_selection_path_imports_decide() -> None:
    """Belt-and-suspenders: no file under router/ other than decide.py
    itself and its test imports router.decide."""
    router_dir = Path(__file__).resolve().parents[1] / "router"
    offenders = []
    for path in router_dir.rglob("*.py"):
        if path.name == "decide.py":
            continue
        imported = _imported_module_names(path)
        if any(
            name == "router.decide" or name.endswith(".decide") for name in imported
        ):
            offenders.append(str(path))
    assert offenders == []
