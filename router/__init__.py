"""svc-free-inference-router: deterministic lane selection for free LLM providers.

The router is code and policy, never an LLM prompt. Every module here enforces a
rule written down first in ``docs/POLICY.md``; nothing dispatches to a real
provider until Phase 2, and nothing in this package holds a credential.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

DATA_DIR = Path(__file__).resolve().parent


def load_json(name: str) -> Any:
    """Load one of the tracked policy data files (``providers.json`` etc.)."""
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def load_providers() -> list[dict[str, Any]]:
    data = load_json("providers.json")
    if not isinstance(data, list):
        raise ValueError("providers.json must be a list of registry entries")
    return data


def load_classification() -> dict[str, Any]:
    data = load_json("classification.json")
    if not isinstance(data, dict) or data.get("default") != "deny":
        raise ValueError("classification.json must be a default-deny mapping")
    return data
