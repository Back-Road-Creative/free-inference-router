"""Tests for router.quarantine — append-only quarantine ledger.

Binding rule: docs/POLICY.md:45-52 (model-mismatch is a hard stop plus
quarantine) and docs/THREAT-MODEL.md:23-30 (logs never carry raw provider
content).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from router.quarantine import Quarantine

UTC = timezone.utc


def test_record_then_is_quarantined_true(tmp_path: Path) -> None:
    q = Quarantine(tmp_path / "quarantine.jsonl")
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
    q.record("openrouter", "openai/gpt-oss-20b", "model_mismatch", at=now)
    assert q.is_quarantined("openrouter", "openai/gpt-oss-20b") is True


def test_different_model_id_same_provider_not_quarantined(tmp_path: Path) -> None:
    q = Quarantine(tmp_path / "quarantine.jsonl")
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
    q.record("openrouter", "openai/gpt-oss-20b", "model_mismatch", at=now)
    assert q.is_quarantined("openrouter", "openai/gpt-oss-120b") is False


def test_unknown_provider_model_not_quarantined(tmp_path: Path) -> None:
    q = Quarantine(tmp_path / "quarantine.jsonl")
    assert q.is_quarantined("groq", "openai/gpt-oss-120b") is False


def test_jsonl_line_contains_no_content_keys(tmp_path: Path) -> None:
    path = tmp_path / "quarantine.jsonl"
    q = Quarantine(path)
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
    q.record("openrouter", "openai/gpt-oss-20b", "model_mismatch", at=now)

    forbidden = {"prompt", "messages", "content", "input", "output"}
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert forbidden.isdisjoint(record.keys())
    assert record["provider"] == "openrouter"
    assert record["model_id"] == "openai/gpt-oss-20b"
    assert record["reason"] == "model_mismatch"
    assert "at" in record


def test_append_only_multiple_records(tmp_path: Path) -> None:
    path = tmp_path / "quarantine.jsonl"
    q = Quarantine(path)
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
    q.record("openrouter", "openai/gpt-oss-20b", "model_mismatch", at=now)
    q.record("groq", "openai/gpt-oss-120b", "terms_change", at=now)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert q.is_quarantined("openrouter", "openai/gpt-oss-20b") is True
    assert q.is_quarantined("groq", "openai/gpt-oss-120b") is True
