"""Content-free audit log: provider/model/latency/status only, never prompts.

Per plan Sec.6: keep provider/model, latency, tokens, status, attempt IDs
only -- never prompts, source, tool args/results, or secret values. Enforced
two ways here, both upstream of the write (Code Quality tier 2 in
CLAUDE.md -- validate before the irreversible append, not after):

1. An explicit field allowlist -- any other key raises ``ValueError`` before
   a single byte is written, so a caller cannot smuggle e.g. ``prompt=`` in.
2. Every string value is run through ``router.bundle.scan_text`` -- a
   key-shaped value anywhere in a record raises and nothing is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from router.bundle import scan_text

_ALLOWED_FIELDS = frozenset(
    {
        "attempt_id",
        "provider",
        "requested_model",
        "returned_model",
        "status",
        "latency_ms",
        "tokens_in",
        "tokens_out",
        "lane",
        "reason",
        "at",
    }
)


class AuditLog:
    """Appends allowlisted JSONL records to ``path``."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, **fields: Any) -> None:
        unknown = set(fields) - _ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"disallowed audit field(s): {sorted(unknown)}")

        for key, value in fields.items():
            if isinstance(value, str) and scan_text(value):
                raise ValueError(f"key-shaped value in audit field {key!r}")

        line = json.dumps(fields, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
