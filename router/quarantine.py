"""Append-only quarantine ledger.

POLICY.md:45-52: a returned ``model_id`` differing from the request is a
hard stop plus quarantine. POLICY.md:60-62: a provider terms change also
quarantines it until JP re-approves. The selector will consult
``is_quarantined`` later (not built here). Records carry only identifiers
and a reason string — never prompt or response content
(docs/THREAT-MODEL.md:23-30: logs never carry raw provider content).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Quarantine:
    """JSONL of ``(provider, model_id, reason, at)``, append-only."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, provider: str, model_id: str, reason: str, at: datetime) -> None:
        entry = {
            "provider": provider,
            "model_id": model_id,
            "reason": reason,
            "at": at.isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def is_quarantined(self, provider: str, model_id: str) -> bool:
        try:
            with open(self.path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            return False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("provider") == provider and entry.get("model_id") == model_id:
                return True
        return False
