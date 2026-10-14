"""free-inference-router: deterministic lane selection for free LLM providers.

The router is code and policy, never an LLM prompt. Every module here enforces a
rule written down first in ``docs/POLICY.md``; nothing dispatches to a real
provider until Phase 2, and nothing in this package holds a credential.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

DATA_DIR = Path(__file__).resolve().parent

#: Directory holding the operator's private inputs. Unset (the public checkout,
#: CI) means no overlay: every billing-gated row stays unattested and refused.
INPUTS_DIR_ENV = "FIR_INPUTS_DIR"
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
    """``$FIR_INPUTS_DIR/providers.local.json`` when the variable is set."""
    inputs_dir = os.environ.get(INPUTS_DIR_ENV)
    return Path(inputs_dir) / OVERLAY_NAME if inputs_dir else None


def _apply_overlay(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = private_overlay_path()
    if path is None or not path.is_file():
        return data
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"{OVERLAY_NAME} must be a list of partial rows")
    by_key = {(e["provider"], e["model_id"]): e for e in data}
    for row in rows:
        key = (row.get("provider"), row.get("model_id"))
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
        by_key[key].update({k: row[k] for k in OVERLAY_FIELDS if k in row})
    return data


def billing_attested(entry: dict[str, Any]) -> bool:
    """A billing-gated row (a free tier that exists only while the key's cloud
    project has no billing account) counts only once a human attested it.
    Ungated rows need no attestation."""
    if not entry.get("billing_gated"):
        return True
    return bool(entry.get("billing_disabled_attested"))


def load_classification() -> dict[str, Any]:
    data = load_json("classification.json")
    if not isinstance(data, dict) or data.get("default") != "deny":
        raise ValueError("classification.json must be a default-deny mapping")
    return data
