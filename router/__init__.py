"""free-inference-router: deterministic lane selection for free LLM providers.

The router is code and policy, never an LLM prompt. Every module here enforces a
rule written down first in ``docs/POLICY.md``; nothing dispatches to a real
provider until Phase 2, and nothing in this package holds a credential.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

DATA_DIR = Path(__file__).resolve().parent

#: Directory holding the operator's private inputs. Set to a path: that dir.
#: Set to "" (the test suite's default): no overlay. Unset: the first
#: ``_private/<PRIVATE_DIR_NAME>`` beside an ancestor of the checkout — a fixed
#: name, never the checkout's own folder name, so a clone finds the operator's
#: layer whatever it is called and no launcher has to export the variable. No
#: overlay means every billing-gated row stays refused.
INPUTS_DIR_ENV = "FIR_INPUTS_DIR"
#: The operator's private-inputs dir name, searched beside each ancestor.
PRIVATE_DIR_NAME = "free-inference-router"
#: The attestation contract (registry.schema.json): ``YYYY-MM-DD JP: <text>``,
#: no surrounding whitespace, non-blank text. Anything else fails closed.
ATTESTATION_RE = re.compile(r"\d{4}-\d{2}-\d{2} JP: \S(?:.*\S)?")
#: The private overlay: a list of ``{provider, model_id, <private field>...}``
#: rows merged onto the tracked registry by (provider, model_id).
OVERLAY_NAME = "providers.local.json"
#: The only keys the overlay may set. It attests; it never admits, re-caps or
#: re-classes a row — those stay policy decisions in the tracked file.
OVERLAY_FIELDS = frozenset({"billing_disabled_attested"})


def load_json(name: str) -> Any:
    """Load one of the tracked policy data files (``providers.json`` etc.)."""
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def load_providers() -> list[dict[str, Any]]:
    """The tracked registry with the private overlay (if any) merged in."""
    data = load_json("providers.json")
    if not isinstance(data, list):
        raise ValueError("providers.json must be a list of registry entries")
    return _apply_overlay(data)


def private_overlay_path() -> Path | None:
    """``$FIR_INPUTS_DIR/providers.local.json``; see ``INPUTS_DIR_ENV``."""
    inputs_dir = os.environ.get(INPUTS_DIR_ENV)
    if inputs_dir is not None:
        return Path(inputs_dir) / OVERLAY_NAME if inputs_dir else None
    for ancestor in DATA_DIR.parent.parents:
        candidate = ancestor / "_private" / PRIVATE_DIR_NAME
        if candidate.is_dir():
            return candidate / OVERLAY_NAME
    return None


def valid_attestation(value: object) -> bool:
    """True only for a string on the attestation contract."""
    return isinstance(value, str) and ATTESTATION_RE.fullmatch(value) is not None


def _apply_overlay(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = private_overlay_path()
    if path is None or not path.is_file():
        return data
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"{OVERLAY_NAME} must be a list of partial rows")
    by_key = {(e["provider"], e["model_id"]): e for e in data}
    seen: set[tuple[Any, Any]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{OVERLAY_NAME} rows must be objects, got {row!r}")
        key = (row.get("provider"), row.get("model_id"))
        if key in seen:
            raise ValueError(f"{OVERLAY_NAME} names {key[0]}/{key[1]} twice")
        seen.add(key)
        extra = set(row) - {"provider", "model_id"} - OVERLAY_FIELDS
        if extra:
            raise ValueError(
                f"{OVERLAY_NAME} may only set {sorted(OVERLAY_FIELDS)}; "
                f"{key[0]}/{key[1]} sets {sorted(extra)}"
            )
        if key not in by_key:
            raise ValueError(
                f"{OVERLAY_NAME} names {key[0]}/{key[1]}, not in the registry"
            )
        if not valid_attestation(row.get("billing_disabled_attested")):
            raise ValueError(
                f"{OVERLAY_NAME}: {key[0]}/{key[1]} attestation is not"
                " 'YYYY-MM-DD JP: <text>'"
            )
        by_key[key].update({k: row[k] for k in OVERLAY_FIELDS if k in row})
    return data


def billing_attested(entry: dict[str, Any]) -> bool:
    """A billing-gated row (a free tier that exists only while the key's cloud
    project has no billing account) counts only once a human attested it.
    Ungated rows need no attestation."""
    if not entry.get("billing_gated"):
        return True
    return valid_attestation(entry.get("billing_disabled_attested"))


def load_classification() -> dict[str, Any]:
    data = load_json("classification.json")
    if not isinstance(data, dict) or data.get("default") != "deny":
        raise ValueError("classification.json must be a default-deny mapping")
    return data
